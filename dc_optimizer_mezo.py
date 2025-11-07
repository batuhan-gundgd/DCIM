#!/usr/bin/env python
# coding: utf-8

"""
dc_optimizer_mezo.py

[cite_start]Bu betik, 'yeni batarya.pdf' [cite: 1-139] [cite_start]dosyasında belirtilen "Mezo" (Zone) [cite: 49-109]
[cite_start]ve "Makro" (ESD) [cite: 1-48] hiyerarşisini kullanarak veri merkezi operasyonunu
optimize eder.

[cite_start]Toplam 5 batarya (1 ESD + 4 Zone) [cite: 1-139] yönetir ve tüm kısıtları
(DC Kapasitesi, C-Rate, SoC Limitleri) uygular.

GÜNCELLEME: NameError (idx_grid) düzeltildi.
"""

import argparse
import pandas as pd
import numpy as np
from scipy.optimize import linprog
import sys
from dataclasses import dataclass, asdict
from typing import Optional

# --- Proje Dosyalarının Yüklenmesi ---
try:
    # config_loader, HVAC (COP) ve DC (Zone/Rack sayısı) parametrelerini sağlar
    from config_loader import load_dc, HVACParams, DCParams
    # datacenter_core, COP hesaplama fonksiyonunu sağlar
    from datacenter_core import cop_from_ambient
    # battery_core, varsayılan batarya parametrelerini (örn. kapasite) sağlar
    from battery_core import BatteryParams
    
    try:
        from load_shift_price import QueueParams
    except ImportError:
        print("Uyarı: 'load_shift_price.py' içinde 'QueueParams' bulunamadı.")
        print("Varsayılan Gecikme Parametreleri kullanılıyor.")
        @dataclass
        class QueueParams:
            max_delay_hours: int = 24
            delay_penalty_eur_per_mwh: float = 10.0
            drop_penalty_eur_per_mwh: float = 100.0

except ImportError as e:
    print(f"Hata: Gerekli proje modülleri yüklenemedi: {e}")
    sys.exit(1)


# --- Veri Yükleme ve Hazırlama Fonksiyonları ---

def ensure_timestamp(df, col_name=None, use_dayfirst=True):
    if col_name is None:
        for col in df.columns:
            if "time" in col.lower() or "date" in col.lower():
                col_name = col
                break
    if col_name is None:
        raise ValueError("DataFrame'de zaman damgası sütunu bulunamadı.")
    df = df.rename(columns={col_name: "timestamp"})
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", dayfirst=use_dayfirst)
    except Exception:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df

def create_synthetic_ambient_temp(timestamps: pd.Series) -> np.ndarray:
    print("Sentetik dış ortam sıcaklığı (ambient_temp_c) oluşturuluyor...")
    day_of_year = timestamps.dt.dayofyear
    hour_of_day = timestamps.dt.hour
    avg_temp = 10.0
    seasonal_variation = -7 * np.cos(2 * np.pi * day_of_year / 365.25)
    daily_variation = -4 * np.cos(2 * np.pi * hour_of_day / 24)
    noise = np.random.normal(0, 1.0, size=len(timestamps))
    ambient_temp_c = avg_temp + seasonal_variation + daily_variation + noise
    return ambient_temp_c.values

def load_data_and_params(workload_file, price_file, dc_config_file, year=2025):
    print(f"Veriler yükleniyor: {workload_file}, {price_file}")
    try:
        workload = pd.read_csv(workload_file)
        price = pd.read_csv(price_file)
    except FileNotFoundError as e:
        print(f"Hata: Dosya bulunamadı -> {e}")
        sys.exit(1)

    workload = ensure_timestamp(workload, use_dayfirst=True)
    price = ensure_timestamp(price, col_name="Datetime (Local)", use_dayfirst=False)
    price = price.rename(columns={"Price (EUR/MWhe)": "price_eur_per_mwh"})

    workload = workload[workload["timestamp"].dt.year == year].reset_index(drop=True)
    price = price[price["timestamp"].dt.year == year].reset_index(drop=True)

    price['timestamp'] = price['timestamp'].dt.floor('h')
    workload['timestamp'] = workload['timestamp'].dt.floor('h')
    price_hourly = price.groupby('timestamp')['price_eur_per_mwh'].mean().reset_index()
    data = pd.merge(workload, price_hourly, on="timestamp", how="inner")
    
    req_cols = ["timestamp", "critical_load_mw", "flexible_load_mw", "price_eur_per_mwh"]
    if not all(col in data.columns for col in req_cols):
        print(f"Hata: Birleştirilmiş veride eksik kolonlar. Gerekenler: {req_cols}")
        sys.exit(1)

    # Parametreleri Yükle
    print(f"Parametreler yükleniyor: {dc_config_file}")
    dc_conf_reader, hvac_params, dc_params = load_dc(dc_config_file)
    n_zones = dc_conf_reader.NUM_ROWS # 4 Zone
    print(f"Fiziksel yapı: {n_zones} Zone (Satır) bulundu.")

    # Sentetik Sıcaklık ve COP Hesapla
    data["ambient_temp_c"] = create_synthetic_ambient_temp(data["timestamp"])
    data["cop"] = data["ambient_temp_c"].apply(lambda temp: cop_from_ambient(hvac_params, temp))
    
    print(f"Veri yükleme tamamlandı. Sentetik sıcaklık ve COP eklendi. {len(data)} saatlik zaman adımı bulundu.")
    return data, hvac_params, dc_params, n_zones, QueueParams(), BatteryParams()


# --- MEZO OPTİMİZASYON ÇEKİRDEĞİ ---

def solve_mezo_optimization_window(
    window_data: pd.DataFrame,
    n_zones: int,
    dc_capacity_mw: float,
    queue_params: QueueParams,
    # 5 Batarya için parametre setleri
    esd_bat_params: dict,
    zone_bat_params: dict,
    # Başlangıç durumları
    soc_0_esd: float,
    soc_0_zones: np.ndarray,
    queue_0: float
) -> Optional[object]:
    """
    Verilen 'N_window' uzunluğundaki pencere için Mezo (5 Batarya) LP problemini kurar ve çözer.
    """
    
    # --- 1. Pencere Parametrelerini Hazırla ---
    N_window = len(window_data)
    if N_window == 0:
        return None

    P_crit_total_in = window_data["critical_load_mw"].values
    P_flex_in = window_data["flexible_load_mw"].values
    Price = window_data["price_eur_per_mwh"].values
    COP = window_data["cop"].values
    
    # Kritik yükü Zone'lara eşit dağıt (Varsayım)
    P_crit_zone_in = P_crit_total_in / n_zones

    DELAY_PENALTY = queue_params.delay_penalty_eur_per_mwh
    MAX_IT_CAPACITY_MW = dc_capacity_mw

    # 5 Batarya Parametresini Ayarla
    B_ESD = esd_bat_params
    B_Z = zone_bat_params

    # --- 2. Optimizasyon Değişkenlerini Tanımla (N_window * 11 adet) ---
    n_vars = N_window * (1 + 1 + 1 + 1 + n_zones + n_zones) # 11 * N
    
    # Index dilimleri
    idx_grid = slice(0 * N_window, 1 * N_window)
    idx_flex = slice(1 * N_window, 2 * N_window)
    idx_esd_c = slice(2 * N_window, 3 * N_window)
    idx_esd_d = slice(3 * N_window, 4 * N_window)
    idx_zone_c = slice(4 * N_window, (4 + n_zones) * N_window) # 4*N adet
    idx_zone_d = slice((4 + n_zones) * N_window, (4 + 2 * n_zones) * N_window) # 4*N adet

    # --- 3. Amaç Fonksiyonu (c) ---
    c = np.zeros(n_vars)
    
    # 1. Grid Maliyeti
    c[idx_grid] = Price
    
    # 2. Gecikme Cezası Maliyeti (Tasarrufu)
    c[idx_flex] = np.array([- (N_window - t) * DELAY_PENALTY for t in range(N_window)])

    # --- 4. Sınırlar (Bounds) - (GÜÇ Limitleri) ---
    bounds = np.zeros((n_vars, 2))
    bounds[idx_grid, :] = (0.0, np.inf)
    bounds[idx_flex, :] = (0.0, np.inf) # Kısıtlarla yönetilecek
    bounds[idx_esd_c, :] = (0.0, B_ESD['max_c_mw']) # ESD Şarj Güç Limiti
    bounds[idx_esd_d, :] = (0.0, B_ESD['max_d_mw']) # ESD Deşarj Güç Limiti
    
    # Zone bataryalarının güç limitlerini ayarla
    bounds_zone_c = np.tile([0.0, B_Z['max_c_mw']], (N_window * n_zones, 1))
    bounds_zone_d = np.tile([0.0, B_Z['max_d_mw']], (N_window * n_zones, 1))
    
    bounds[idx_zone_c, :] = bounds_zone_c.reshape(-1, 2)
    bounds[idx_zone_d, :] = bounds_zone_d.reshape(-1, 2)

    # --- 5. Eşitlik Kısıtları (A_eq, b_eq) - Güç Dengesi (N_window adet) ---
    A_eq = np.zeros((N_window, n_vars))
    b_eq = np.zeros(N_window)
    
    # Zone şarj/deşarj için katsayılar (sum)
    zone_c_coeffs = np.zeros(n_vars)
    zone_d_coeffs = np.zeros(n_vars)
    zone_c_coeffs[idx_zone_c] = -1.0
    zone_d_coeffs[idx_zone_d] = 1.0

    for t in range(N_window):
        dc_load_factor = 1.0 + (1.0 / COP[t])
        
        A_eq[t, idx_grid.start + t] = 1.0     # P_grid
        A_eq[t, idx_esd_c.start + t] = -1.0    # -P_ESD_charge
        A_eq[t, idx_esd_d.start + t] = 1.0     # +P_ESD_discharge
        
        A_eq[t, idx_flex.start + t] = -dc_load_factor # -P_flex_exec * (1 + 1/COP)
        
        # Zone'ları topla
        A_eq[t, idx_zone_c.start + t*n_zones : idx_zone_c.start + (t+1)*n_zones] = -1.0 # -sum(P_Z_c)
        A_eq[t, idx_zone_d.start + t*n_zones : idx_zone_d.start + (t+1)*n_zones] = 1.0 # +sum(P_Z_d)

        
        b_eq[t] = P_crit_total_in[t] * dc_load_factor

    # --- 6. Eşitsizlik Kısıtları (A_ub, b_ub) ---
    n_ub_constraints = N_window * (1 + 1 + 2 * (1 + n_zones)) # 12*N
    A_ub = np.zeros((n_ub_constraints, n_vars))
    b_ub = np.zeros(n_ub_constraints)
    row_idx = 0

    # 6.1 DC Kapasite Kısıtı (N adet)
    for t in range(N_window):
        A_ub[row_idx, idx_flex.start + t] = 1.0
        b_ub[row_idx] = MAX_IT_CAPACITY_MW - P_crit_total_in[t]
        row_idx += 1
        
    # 6.2 Esnek Yük Kuyruk Kısıtı (N adet)
    tril_matrix_N = np.tril(np.ones((N_window, N_window)))
    A_ub[row_idx : row_idx + N_window, idx_flex] = tril_matrix_N
    b_ub[row_idx : row_idx + N_window] = queue_0 + np.cumsum(P_flex_in)
    row_idx += N_window

    # 6.3 Batarya SoC Kısıtları (Toplam 5 batarya * 2 * N = 10*N adet)
    
    # 6.3.1 ESD Batarya SoC (2*N adet)
    # SoC[t+1] <= MAX
    A_ub[row_idx : row_idx + N_window, idx_esd_c] = tril_matrix_N * B_ESD['eff_c']
    A_ub[row_idx : row_idx + N_window, idx_esd_d] = tril_matrix_N * (-1.0 / B_ESD['eff_d'])
    b_ub[row_idx : row_idx + N_window] = B_ESD['max_e_mwh'] - soc_0_esd
    row_idx += N_window
    # SoC[t+1] >= MIN
    A_ub[row_idx : row_idx + N_window, idx_esd_c] = tril_matrix_N * (-B_ESD['eff_c'])
    A_ub[row_idx : row_idx + N_window, idx_esd_d] = tril_matrix_N * (1.0 / B_ESD['eff_d'])
    b_ub[row_idx : row_idx + N_window] = soc_0_esd - B_ESD['min_e_mwh']
    row_idx += N_window

    # 6.3.2 Zone Batarya SoC (4 * 2*N = 8*N adet)
    tril_matrix_N_zones = np.kron(np.eye(n_zones), tril_matrix_N) # (4N x 4N)
    
    # Zone SoC <= MAX
    A_ub[row_idx : row_idx + N_window * n_zones, idx_zone_c] = tril_matrix_N_zones * B_Z['eff_c']
    A_ub[row_idx : row_idx + N_window * n_zones, idx_zone_d] = tril_matrix_N_zones * (-1.0 / B_Z['eff_d'])
    b_ub_target_max = np.zeros(N_window * n_zones)
    for i in range(n_zones):
        b_ub_target_max[i*N_window : (i+1)*N_window] = B_Z['max_e_mwh'] - soc_0_zones[i]
    b_ub[row_idx : row_idx + N_window * n_zones] = b_ub_target_max
    row_idx += N_window * n_zones

    # Zone SoC >= MIN
    A_ub[row_idx : row_idx + N_window * n_zones, idx_zone_c] = tril_matrix_N_zones * (-B_Z['eff_c'])
    A_ub[row_idx : row_idx + N_window * n_zones, idx_zone_d] = tril_matrix_N_zones * (1.0 / B_Z['eff_d'])
    b_ub_target_min = np.zeros(N_window * n_zones)
    for i in range(n_zones):
        b_ub_target_min[i*N_window : (i+1)*N_window] = soc_0_zones[i] - B_Z['min_e_mwh']
    b_ub[row_idx : row_idx + N_window * n_zones] = b_ub_target_min
    row_idx += N_window * n_zones
    
    # --- 7. Modeli Çöz ---
    sol = linprog(c=c, 
                  A_ub=A_ub, b_ub=b_ub, 
                  A_eq=A_eq, b_eq=b_eq, 
                  bounds=list(zip(bounds[:, 0], bounds[:, 1])),
                  method='highs')
    
    return sol


def run_mpc_simulation(data, n_zones, dc_capacity_mw, queue_params, 
                       esd_bat_params, zone_bat_params, out_file, window_hours=24):
    
    print(f"\n--- MEZO (5 Batarya) MPC Simülasyonu Başlatılıyor ---")
    
    # --- 1. Sabitleri ve Başlangıç Durumlarını Ayarla ---
    total_steps = len(data)
    DELAY_PENALTY = queue_params.delay_penalty_eur_per_mwh

    # Batarya Parametreleri
    B_ESD = esd_bat_params
    B_Z = zone_bat_params

    # Başlangıç durumları
    current_soc_esd_mwh = B_ESD['min_e_mwh']
    current_soc_zones_mwh = np.full(n_zones, B_Z['min_e_mwh'])
    current_queue_mwh = 0.0
    
    results_list = []

    # --- 2. Ana Simülasyon Döngüsü (t = 0 ... N-1) ---
    for t in range(total_steps):
        if (t % 100 == 0) or (t == total_steps - 1):
             print(f"  Simülasyon adımı: {t+1} / {total_steps} (Kuyruk: {current_queue_mwh:.1f} MWh, ESD SoC: {current_soc_esd_mwh:.1f} MWh)")

        start_idx = t
        end_idx = min(t + window_hours, total_steps)
        window_data = data.iloc[start_idx:end_idx].reset_index(drop=True)
        if len(window_data) == 0: break

        sol = solve_mezo_optimization_window(
            window_data, n_zones, dc_capacity_mw, queue_params,
            B_ESD, B_Z,
            soc_0_esd=current_soc_esd_mwh,
            soc_0_zones=current_soc_zones_mwh,
            queue_0=current_queue_mwh
        )
        
        N_window = len(window_data)
        data_t = data.iloc[t]
        
        # GÜNCELLEME: NameError düzeltmesi için indeksleri burada YENİDEN TANIMLA
        n_vars = N_window * (1 + 1 + 1 + 1 + n_zones + n_zones) # 11 * N
        idx_grid = slice(0 * N_window, 1 * N_window)
        idx_flex = slice(1 * N_window, 2 * N_window)
        idx_esd_c = slice(2 * N_window, 3 * N_window)
        idx_esd_d = slice(3 * N_window, 4 * N_window)
        idx_zone_c = slice(4 * N_window, (4 + n_zones) * N_window) # 4*N adet
        idx_zone_d = slice((4 + n_zones) * N_window, (4 + 2 * n_zones) * N_window) # 4*N adet
        
        # 2.3. Sadece İlk Adımı (t=0) Uygula
        if sol and sol.success:
            P_grid_decision = sol.x[idx_grid.start]
            P_flex_exec_decision = sol.x[idx_flex.start]
            P_esd_c_decision = sol.x[idx_esd_c.start]
            P_esd_d_decision = sol.x[idx_esd_d.start]
            P_zone_c_decisions = sol.x[idx_zone_c.start : idx_zone_c.start + n_zones]
            P_zone_d_decisions = sol.x[idx_zone_d.start : idx_zone_d.start + n_zones]
        
        else:
            # Güvenli Mod
            if t % 100 == 0: print(f"    Uyarı: Saat {t} için optimizasyon başarısız. Güvenli mod.")
            P_flex_exec_decision = 0.0
            P_esd_c_decision = 0.0
            P_esd_d_decision = 0.0
            P_zone_c_decisions = np.zeros(n_zones)
            P_zone_d_decisions = np.zeros(n_zones)
            # Gerekli gücü şebekeden çek
            dc_load_factor = 1.0 + (1.0 / data_t["cop"])
            P_grid_decision = (data_t["critical_load_mw"] + P_flex_exec_decision) * dc_load_factor

        # 2.4. Durum Değişkenlerini (SoC ve Kuyruk) Güncelle
        P_flex_in_t = data_t["flexible_load_mw"]
        
        # ESD SoC
        soc_change_esd = (P_esd_c_decision * B_ESD['eff_c']) - (P_esd_d_decision / B_ESD['eff_d'])
        current_soc_esd_mwh += soc_change_esd
        current_soc_esd_mwh = max(B_ESD['min_e_mwh'], min(B_ESD['max_e_mwh'], current_soc_esd_mwh))
        
        # Zone SoC
        soc_change_zones = (P_zone_c_decisions * B_Z['eff_c']) - (P_zone_d_decisions / B_Z['eff_d'])
        current_soc_zones_mwh += soc_change_zones
        current_soc_zones_mwh = np.maximum(B_Z['min_e_mwh'], np.minimum(B_Z['max_e_mwh'], current_soc_zones_mwh))

        # Kuyruk
        total_available_flex = current_queue_mwh + P_flex_in_t
        P_flex_exec_decision = max(0, min(P_flex_exec_decision, total_available_flex))
        current_queue_mwh = total_available_flex - P_flex_exec_decision
        
        # 2.5. Raporlama Metriklerini Hesapla
        P_IT_opt = data_t["critical_load_mw"] + P_flex_exec_decision
        P_HVAC_opt = P_IT_opt / data_t["cop"]
        P_total_dc_opt = P_IT_opt + P_HVAC_opt
        
        Grid_Cost = P_grid_decision * data_t["price_eur_per_mwh"]
        Delay_Cost = current_queue_mwh * DELAY_PENALTY
        Total_Cost = Grid_Cost + Delay_Cost

        # 2.6. Sonuçları Kaydet
        results_list.append({
            "timestamp": data_t["timestamp"],
            "P_grid_mw": P_grid_decision,
            "P_flex_exec_mw": P_flex_exec_decision,
            "P_IT_load_mw": P_IT_opt,
            "P_HVAC_load_mw": P_HVAC_opt,
            "P_total_dc_load_mw": P_total_dc_opt,
            "Queue_mwh_end": current_queue_mwh,
            "SoC_ESD_mwh_end": current_soc_esd_mwh,
            "SoC_Z1_mwh_end": current_soc_zones_mwh[0],
            "SoC_Z2_mwh_end": current_soc_zones_mwh[1],
            "SoC_Z3_mwh_end": current_soc_zones_mwh[2],
            "SoC_Z4_mwh_end": current_soc_zones_mwh[3],
            "P_ESD_c_mw": P_esd_c_decision,
            "P_ESD_d_mw": P_esd_d_decision,
            "P_Z_c_total_mw": np.sum(P_zone_c_decisions),
            "P_Z_d_total_mw": np.sum(P_zone_d_decisions),
            "grid_cost_eur": Grid_Cost,
            "delay_penalty_eur": Delay_Cost,
            "total_cost_eur": Total_Cost,
            "price_eur_per_mwh": data_t["price_eur_per_mwh"],
            "cop": data_t["cop"]
        })

    # --- 3. Simülasyon Sonu: Raporlama ---
    print("Simülasyon tamamlandı. Sonuçlar işleniyor...")
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(out_file, index=False)
    print(f"\n✅ Mezo optimizasyon sonuçları şuraya kaydedildi: {out_file}")

    print("\n--- ÖZET (Mezo MPC) ---")
    print(f"Toplam Şebeke Maliyeti: {results_df['grid_cost_eur'].sum():.2f} EUR")
    print(f"Toplam Gecikme Cezası:  {results_df['delay_penalty_eur'].sum():.2f} EUR")
    print(f"TOPLAM MALİYET:          {results_df['total_cost_eur'].sum():.2f} EUR")
    print(f"Son Kalan Kuyruk:        {current_queue_mwh:.2f} MWh")


def helper_parse_battery_params(capacity, c_rate_c, c_rate_d, min_soc, max_soc):
    """ Batarya parametrelerini bir sözlükte toplar """
    cap_mwh = float(capacity)
    return {
        "cap_mwh": cap_mwh,
        "eff_c": 0.95, # Varsayılan
        "eff_d": 0.95, # Varsayılan
        "max_c_mw": float(c_rate_c) * cap_mwh,
        "max_d_mw": float(c_rate_d) * cap_mwh,
        "min_e_mwh": float(min_soc) * cap_mwh,
        "max_e_mwh": float(max_soc) * cap_mwh,
    }

def main():
    parser = argparse.ArgumentParser(description="Veri Merkezi Mezo (Zone+ESD) Optimizasyon Betiği")
    # Dosyalar
    parser.add_argument("--workload", required=True, help="İş yükü (kritik+esnek) CSV dosyası (workload_flex.csv)")
    parser.add_argument("--price", required=True, help="Fiyat CSV dosyası (United Kingdom.csv)")
    parser.add_argument("--dc-config", required=True, help="Veri merkezi konfigürasyon JSON dosyası (dc_config.json)")
    parser.add_argument("--out-file", default="results_optimizer_mezo.csv", help="Çıktı CSV dosyasının adı")
    
    # DC Kısıtları
    parser.add_argument("--horizon", type=int, default=24, help="Optimizasyon ufku (pencere) (saat)")
    parser.add_argument("--capacity", type=float, default=10.0, help="Maksimum Veri Merkezi BT Yükü Kapasitesi (MW)")
    
    # ESD Batarya Parametreleri (Makro)
    parser.add_argument("--esd-cap", type=float, default=20.0, help="ESD Batarya Kapasitesi (MWh)")
    parser.add_argument("--esd-c-rate-c", type=float, default=0.3, help="ESD Şarj C-rate (örn: 0.3)")
    parser.add_argument("--esd-c-rate-d", type=float, default=0.2, help="ESD Deşarj C-rate (örn: 0.2)")
    parser.add_argument("--esd-min-soc", type=float, default=0.2, help="ESD Min SoC (örn: 0.2 = 20%%)")
    parser.add_argument("--esd-max-soc", type=float, default=0.8, help="ESD Max SoC (örn: 0.8 = 80%%)")
    
    # Zone Batarya Parametreleri (Mezo)
    parser.add_argument("--zone-cap", type=float, default=5.0, help="HER BİR Zone Batarya Kapasitesi (MWh)")
    parser.add_argument("--zone-c-rate-c", type=float, default=0.5, help="Zone Şarj C-rate (örn: 0.5)")
    parser.add_argument("--zone-c-rate-d", type=float, default=0.5, help="Zone Deşarj C-rate (örn: 0.5)")
    parser.add_argument("--zone-min-soc", type=float, default=0.2, help="Zone Min SoC (örn: 0.2 = 20%%)")
    parser.add_argument("--zone-max-soc", type=float, default=0.8, help="Zone Max SoC (örn: 0.8 = 80%%)")

    args = parser.parse_args()

    # Verileri ve temel parametreleri yükle
    data, hvac_params, dc_params, n_zones, queue_params, default_bat_params = load_data_and_params(
        args.workload, args.price, args.dc_config
    )
    
    # Batarya parametrelerini CLI'dan ayarla
    esd_bat_params = helper_parse_battery_params(
        args.esd_cap, args.esd_c_rate_c, args.esd_c_rate_d, args.esd_min_soc, args.esd_max_soc
    )
    zone_bat_params = helper_parse_battery_params(
        args.zone_cap, args.zone_c_rate_c, args.zone_c_rate_d, args.zone_min_soc, args.zone_max_soc
    )
    
    print("\n--- MEZO SİMÜLASYON PARAMETRELERİ ---")
    print(f"DC BT KAPASİTE KISITI: {args.capacity} MW")
    print(f"ESD BATARYA (1x): {esd_bat_params}")
    print(f"ZONE BATARYA (4x): {zone_bat_params}")

    # Simülasyonu çalıştır
    run_mpc_simulation(
        data, n_zones, args.capacity, queue_params, 
        esd_bat_params, zone_bat_params, 
        args.out_file, args.horizon
    )

if __name__ == "__main__":
    main()