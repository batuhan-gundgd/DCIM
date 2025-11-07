#!/usr/bin/env python
# coding: utf-8

"""
dc_optimizer_mpc.py (KAPASİTE KISITLI)

Bu betik, veri merkezi operasyonunu (batarya kullanımı ve esnek yük zamanlaması)
toplam maliyeti minimize edecek şekilde optimize etmek için "Kayan Ufuklu Model
Tahminli Kontrol" (Receding Horizon Model Predictive Control - MPC) yaklaşımını kullanır.

YENİ KISIT:
Model, veri merkezinin BT yükünü (P_IT = P_critical + P_flex_exec)
belirlenen bir maksimum kapasite (örn. 10 MW) ile sınırlar.
"""

import argparse
import pandas as pd
import numpy as np
from scipy.optimize import linprog
import sys
from dataclasses import asdict
from typing import Optional

# Proje dosyalarını sisteme tanıtma (mevcut yapıya göre)
sys.path.append(".") 

try:
    from config_loader import load_dc
    from battery_core import BatteryParams
    
    try:
        from load_shift_price import QueueParams
    except ImportError:
        print("Uyarı: 'load_shift_price.py' içinde 'QueueParams' bulunamadı.")
        print("Varsayılan Gecikme Parametreleri kullanılıyor.")
        from dataclasses import dataclass
        @dataclass
        class QueueParams:
            max_delay_hours: int = 24
            delay_penalty_eur_per_mwh: float = 10.0 # Örnek ceza
            drop_penalty_eur_per_mwh: float = 100.0

except ImportError as e:
    print(f"Hata: Gerekli modüller yüklenemedi: {e}")
    sys.exit(1)


def ensure_timestamp(df, col_name=None, use_dayfirst=True):
    """
    DataFrame'i zaman damgasına göre temizler ve sıralar.
    use_dayfirst: pd.to_datetime için 'dayfirst' parametresi.
    """
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

def load_data(workload_file, price_file, year=2025):
    """İş yükü ve fiyat verilerini yükler, birleştirir ve temizler."""
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
        print(f"Mevcut kolonlar: {data.columns.tolist()}")
        sys.exit(1)

    print(f"Veri yükleme tamamlandı. {len(data)} saatlik zaman adımı bulundu.")
    return data

def load_parameters(config_file):
    """Konfigürasyon dosyalarından parametreleri yükler."""
    print(f"Parametreler yükleniyor: {config_file}")
    try:
        _dc_conf, hvac_params, _dc_params = load_dc(config_file)
    except FileNotFoundError:
        print(f"Hata: Konfigürasyon dosyası bulunamadı: {config_file}")
        sys.exit(1)
    
    battery_params = BatteryParams()
    queue_params = QueueParams()

    print("Parametreler:")
    print(f"  HVAC: {asdict(hvac_params)}")
    print(f"  BATARYA: {asdict(battery_params)}")
    print(f"  KUYRUK: {asdict(queue_params)}")
    
    return hvac_params, battery_params, queue_params

def solve_optimization_window(
    window_data: pd.DataFrame,
    hvac_params: "HVACParams",
    battery_params: "BatteryParams",
    queue_params: "QueueParams",
    soc_0: float,
    queue_0: float,
    capacity_mw: float  # GÜNCELLEME: Yeni kapasite parametresi eklendi
) -> Optional[object]:
    """
    Verilen 'N_window' uzunluğundaki pencere için LP problemini kurar ve çözer.
    """
    
    # --- 1. Pencere Parametrelerini Hazırla ---
    N_window = len(window_data)
    if N_window == 0:
        return None

    P_crit_in = window_data["critical_load_mw"].values
    P_flex_in = window_data["flexible_load_mw"].values
    Price = window_data["price_eur_per_mwh"].values
    
    COP = hvac_params.CHILLER_COP_BASE 
    DC_LOAD_FACTOR = 1.0 + 1.0 / COP

    BAT_CAP_MWh = battery_params.capacity_mwh
    BAT_EFF_C = battery_params.eff_c
    BAT_EFF_D = battery_params.eff_d
    BAT_C_LIM_MW = battery_params.c_lim * BAT_CAP_MWh
    BAT_D_LIM_MW = battery_params.d_lim * BAT_CAP_MWh
    
    DELAY_PENALTY = queue_params.delay_penalty_eur_per_mwh
    
    # GÜNCELLEME: Yeni kapasite kısıtı parametresi
    MAX_IT_CAPACITY_MW = capacity_mw

    # --- 2. Optimizasyon Değişkenlerini Tanımla (Pencere boyutu: N_window) ---
    n_vars = 4 * N_window
    
    idx_grid = slice(0 * N_window, 1 * N_window)
    idx_charge = slice(1 * N_window, 2 * N_window)
    idx_discharge = slice(2 * N_window, 3 * N_window)
    idx_flex_exec = slice(3 * N_window, 4 * N_window)

    # --- 3. Amaç Fonksiyonu (c) ---
    c = np.zeros(n_vars)
    c[idx_grid] = Price
    c[idx_flex_exec] = np.array([- (N_window - t) * DELAY_PENALTY for t in range(N_window)])

    # --- 4. Sınırlar (Bounds) ---
    bounds = np.zeros((n_vars, 2))
    bounds[idx_grid, :] = (0.0, np.inf)
    bounds[idx_charge, :] = (0.0, BAT_C_LIM_MW)
    bounds[idx_discharge, :] = (0.0, BAT_D_LIM_MW)
    bounds[idx_flex_exec, :] = (0.0, np.inf)

    # --- 5. Eşitlik Kısıtları (A_eq, b_eq) - Güç Dengesi (N_window adet) ---
    A_eq = np.zeros((N_window, n_vars))
    b_eq = np.zeros(N_window)
    
    for t in range(N_window):
        A_eq[t, idx_grid.start + t] = 1.0       
        A_eq[t, idx_charge.start + t] = -1.0      
        A_eq[t, idx_discharge.start + t] = 1.0    
        A_eq[t, idx_flex_exec.start + t] = -DC_LOAD_FACTOR
        b_eq[t] = P_crit_in[t] * DC_LOAD_FACTOR
        
    # --- 6. Eşitsizlik Kısıtlarını (A_ub, b_ub) Kur ---
    # GÜNCELLEME: 3*N -> 4*N kısıt (Kapasite kısıtı eklendi)
    # N: SoC[t] <= MAX
    # N: SoC[t] >= MIN
    # N: Queue[t] >= MIN (yani P_flex_exec <= mevcut olan)
    # N: P_IT[t] <= MAX_CAPACITY (YENİ KISIT)
    
    A_ub = np.zeros((4 * N_window, n_vars)) # GÜNCELLEME: Boyut 3*N'den 4*N'ye çıkarıldı
    b_ub = np.zeros(4 * N_window)
    
    tril_matrix = np.tril(np.ones((N_window, N_window)))
    cum_flex_in = np.cumsum(P_flex_in)
    
    row_idx = 0
    
    # 6.1 Batarya SoC Kısıtları (2*N adet)
    # SoC[t+1] <= BAT_CAP_MWh
    A_ub[row_idx : row_idx + N_window, idx_charge] = tril_matrix * BAT_EFF_C
    A_ub[row_idx : row_idx + N_window, idx_discharge] = tril_matrix * (-1.0 / BAT_EFF_D)
    b_ub[row_idx : row_idx + N_window] = BAT_CAP_MWh - soc_0
    row_idx += N_window

    # SoC[t+1] >= 0
    A_ub[row_idx : row_idx + N_window, idx_charge] = tril_matrix * (-BAT_EFF_C)
    A_ub[row_idx : row_idx + N_window, idx_discharge] = tril_matrix * (1.0 / BAT_EFF_D)
    b_ub[row_idx : row_idx + N_window] = soc_0
    row_idx += N_window

    # 6.2 Esnek Yük Kuyruk Kısıtları (N adet)
    # Queue[t+1] >= 0  (Yani -> Sum(P_flex_exec) <= queue_0 + Sum(P_flex_in))
    A_ub[row_idx : row_idx + N_window, idx_flex_exec] = tril_matrix
    b_ub[row_idx : row_idx + N_window] = queue_0 + cum_flex_in
    row_idx += N_window
    
    # 6.3 GÜNCELLEME: Kapasite Kısıtı (N adet)
    # P_IT[t] <= MAX_IT_CAPACITY_MW
    # (P_crit_in[t] + P_flex_exec[t]) <= MAX_IT_CAPACITY_MW
    # P_flex_exec[t] <= MAX_IT_CAPACITY_MW - P_crit_in[t]
    
    # Bu kısıt kümülatif değil, anlıktır (identity matrix, 'np.eye'
    # kullanılır, ancak 1'lerden oluştuğu için 'tril_matrix'in
    # köşegenini kullanmakla aynıdır, biz yine de anlık yapalım)
    identity_matrix = np.eye(N_window)
    A_ub[row_idx : row_idx + N_window, idx_flex_exec] = identity_matrix
    b_ub[row_idx : row_idx + N_window] = MAX_IT_CAPACITY_MW - P_crit_in
    row_idx += N_window


    # --- 7. Modeli Çöz ---
    sol = linprog(c=c, 
                  A_ub=A_ub, b_ub=b_ub, 
                  A_eq=A_eq, b_eq=b_eq, 
                  bounds=list(zip(bounds[:, 0], bounds[:, 1])),
                  method='highs')
    
    return sol


def run_mpc_simulation(data, hvac_params, battery_params, queue_params, out_file, window_hours=24, capacity_mw=10.0): # GÜNCELLEME
    """
    Kayan Ufuk (MPC) simülasyonunu çalıştırır.
    """
    
    print(f"Kayan Ufuk (MPC) simülasyonu başlatılıyor. Ufuk: {window_hours} saat.")
    print(f"VERİ MERKEZİ BT KAPASİTE KISITI: {capacity_mw} MW") # GÜNCELLEME
    
    # --- 1. Sabitleri ve Başlangıç Durumlarını Ayarla ---
    total_steps = len(data)
    
    COP = hvac_params.CHILLER_COP_BASE 
    DC_LOAD_FACTOR = 1.0 + 1.0 / COP

    BAT_CAP_MWh = battery_params.capacity_mwh
    BAT_EFF_C = battery_params.eff_c
    BAT_EFF_D = battery_params.eff_d
    
    DELAY_PENALTY = queue_params.delay_penalty_eur_per_mwh

    current_soc_mwh = 0.0
    current_queue_mwh = 0.0
    
    results_list = []

    # --- 2. Ana Simülasyon Döngüsü (t = 0 ... N-1) ---
    for t in range(total_steps):
        if (t % 100 == 0) or (t == total_steps - 1):
             print(f"  Simülasyon adımı: {t+1} / {total_steps} (SoC: {current_soc_mwh:.1f} MWh, Kuyruk: {current_queue_mwh:.1f} MWh)")

        # 2.1. Optimizasyon Penceresini Al
        start_idx = t
        end_idx = min(t + window_hours, total_steps)
        window_data = data.iloc[start_idx:end_idx].reset_index(drop=True)
        
        if len(window_data) == 0:
            print("Veri sonu, simülasyon durduruluyor.")
            break

        # 2.2. O anki (t) durumlar için pencere optimizasyonunu çöz
        sol = solve_optimization_window(
            window_data, hvac_params, battery_params, queue_params,
            soc_0=current_soc_mwh,
            queue_0=current_queue_mwh,
            capacity_mw=capacity_mw  # GÜNCELLEME
        )
        
        N_window = len(window_data)
        data_t = data.iloc[t]
        
        if sol and sol.success:
            P_grid_decision = sol.x[0]
            P_charge_decision = sol.x[N_window * 1]
            P_discharge_decision = sol.x[N_window * 2]
            P_flex_exec_decision = sol.x[N_window * 3]
        
        else:
            if t % 100 == 0:
                print(f"    Uyarı: Saat {t} için optimizasyon başarısız. Güvenli mod uygulanıyor.")
            P_charge_decision = 0.0
            P_discharge_decision = 0.0
            P_flex_exec_decision = 0.0
            P_IT_fallback = data_t["critical_load_mw"] + P_flex_exec_decision
            P_total_dc_fallback = P_IT_fallback * DC_LOAD_FACTOR
            P_grid_decision = P_total_dc_fallback
            
        P_grid_decision = max(0, P_grid_decision)
        P_charge_decision = max(0, P_charge_decision)
        P_discharge_decision = max(0, P_discharge_decision)
        P_flex_exec_decision = max(0, P_flex_exec_decision)

        # 2.3. GÜNCELLEME: Kapasite kısıtını simülasyon adımında TEKRAR zorla
        # (Optimizasyon çözümü küçük sapmalar yapabilir)
        available_capacity = capacity_mw - data_t["critical_load_mw"]
        P_flex_exec_decision = max(0, min(P_flex_exec_decision, available_capacity))

        # 2.4. Durum Değişkenlerini (SoC ve Kuyruk) Güncelle
        P_flex_in_t = data_t["flexible_load_mw"]
        
        soc_change = (P_charge_decision * BAT_EFF_C) - (P_discharge_decision / BAT_EFF_D)
        current_soc_mwh += soc_change
        current_soc_mwh = max(0.0, min(BAT_CAP_MWh, current_soc_mwh))
        
        # GÜNCELLEME: Kuyruk hesabı artık P_flex_exec_decision'a tam bağlı
        # P_flex_exec_decision, kapasite nedeniyle kısıtlanmış olabilir.
        
        # O anki toplam iş yükü (bekleyen + yeni gelen)
        total_available_flex = current_queue_mwh + P_flex_in_t
        
        # Çalıştırılan iş, mevcut olandan fazla olamaz
        P_flex_exec_decision = max(0, min(P_flex_exec_decision, total_available_flex))
        
        # Kuyruğu güncelle
        current_queue_mwh = total_available_flex - P_flex_exec_decision
        current_queue_mwh = max(0.0, current_queue_mwh)
        
        # 2.5. Raporlama Metriklerini Hesapla
        P_IT_opt = data_t["critical_load_mw"] + P_flex_exec_decision
        P_HVAC_opt = P_IT_opt / COP
        P_total_dc_opt = P_IT_opt + P_HVAC_opt
        
        Grid_Cost = P_grid_decision * data_t["price_eur_per_mwh"]
        Delay_Cost = current_queue_mwh * DELAY_PENALTY
        Total_Cost = Grid_Cost + Delay_Cost

        # 2.6. Sonuçları Kaydet
        results_list.append({
            "timestamp": data_t["timestamp"],
            "critical_load_mw": data_t["critical_load_mw"],
            "flexible_load_mw_in": P_flex_in_t,
            "price_eur_per_mwh": data_t["price_eur_per_mwh"],
            "P_grid_mw_decision": P_grid_decision,
            "P_charge_mw_decision": P_charge_decision,
            "P_discharge_mw_decision": P_discharge_decision,
            "P_flex_exec_mw_decision": P_flex_exec_decision,
            "P_IT_load_mw": P_IT_opt, # GÜNCELLEME: Bu değer artık 10 MW'ı aşmamalı
            "P_HVAC_load_mw": P_HVAC_opt,
            "P_total_dc_load_mw": P_total_dc_opt,
            "SoC_mwh_end": current_soc_mwh,
            "Queue_mwh_end": current_queue_mwh,
            "grid_cost_eur": Grid_Cost,
            "delay_penalty_eur": Delay_Cost,
            "total_cost_eur": Total_Cost
        })

    # --- 3. Simülasyon Sonu: Raporlama ---
    print("Simülasyon tamamlandı. Sonuçlar işleniyor...")
    
    if not results_list:
        print("Uyarı: Hiçbir sonuç üretilmedi.")
        return

    results_df = pd.DataFrame(results_list)
    out_file_with_capacity = out_file.replace(".csv", f"_cap_{capacity_mw}MW.csv")
    results_df.to_csv(out_file_with_capacity, index=False)
    print(f"\n✅ MPC optimizasyon sonuçları şuraya kaydedildi: {out_file_with_capacity}")

    print("\n--- ÖZET (MPC) ---")
    print(f"KAPASİTE KISITI: {capacity_mw} MW")
    print(f"Toplam Şebeke Maliyeti: {results_df['grid_cost_eur'].sum():.2f} EUR")
    print(f"Toplam Gecikme Cezası:  {results_df['delay_penalty_eur'].sum():.2f} EUR")
    print(f"TOPLAM MALİYET:          {results_df['total_cost_eur'].sum():.2f} EUR")
    print(f"Toplam Esnek Yük (Girdi):{results_df['flexible_load_mw_in'].sum():.2f} MWh")
    print(f"Toplam Esnek Yük (Çalışan):{results_df['P_flex_exec_mw_decision'].sum():.2f} MWh")
    print(f"Son Kalan Kuyruk:        {current_queue_mwh:.2f} MWh")
    print(f"Son SoC:                 {current_soc_mwh:.2f} MWh")


def main():
    parser = argparse.ArgumentParser(description="Veri Merkezi MPC Optimizasyon Betiği (Kapasite Kısıtlı)")
    parser.add_argument("--workload", required=True, help="İş yükü (kritik+esnek) CSV dosyası (workload_flex.csv)")
    parser.add_argument("--price", required=True, help="Fiyat CSV dosyası (United Kingdom.csv)")
    parser.add_argument("--dc-config", required=True, help="Veri merkezi konfigürasyon JSON dosyası (dc_config.json)")
    parser.add_argument("--out-file", default="results_optimizer_mpc.csv", help="Çıktı CSV dosyasının adı")
    parser.add_argument("--horizon", type=int, default=24, help="Optimizasyon ufku (pencere) (saat)")
    
    # GÜNCELLEME: Yeni kapasite argümanı
    parser.add_argument("--capacity", type=float, default=10.0, help="Maksimum Veri Merkezi BT Yükü Kapasitesi (MW)")
    
    args = parser.parse_args()

    # Parametreleri ve verileri yükle
    data = load_data(args.workload, args.price)
    hvac_params, battery_params, queue_params = load_parameters(args.dc_config)
    
    # Optimizasyonu çalıştır
    run_mpc_simulation(data, hvac_params, battery_params, queue_params, 
                       args.out_file, args.horizon, args.capacity) # GÜNCELLEME

if __name__ == "__main__":
    main()