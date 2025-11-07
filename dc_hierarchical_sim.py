#!/usr/bin/env python
# coding: utf-8

"""
dc_hierarchical_sim.py

Bu betik, 'yeni batarya.pdf'  dosyasında belirtilen 3 katmanlı (Mikro/Mezo/Makro)
batarya hiyerarşisini ve 'datacenter1.py' içindeki fiziksel sunucu modelini
birleştiren KURAL TABANLI BİR SİMÜLATÖR'dür.

'dc_optimizer_mpc.py' betiğinin yerini alır.

Ana Mantık:
1.  Fiziksel DC'yi (20 Rack, 4000 CPU) başlatır.
2.  Hiyerarşik bataryaları (1 ESD, 4 Zone, 20 Rack)  başlatır.
3.  Toplam yükü ('workload_flex.csv') okur.
4.  Kritik yükü 20 rack'e eşit dağıtır.
5.  Esnek yükü, verimlilik ve batarya durumuna göre "en uygun" rack'e (kural tabanlı) dağıtır.
6.  Enerji akışını PDF'teki  hiyerarşiye (Rack -> Zone -> ESD -> Grid) göre hesaplar.
"""

import argparse
import pandas as pd
import numpy as np
import sys
from dataclasses import dataclass
from collections import deque
from typing import List, Deque, Dict

# --- Proje Dosyalarının Yüklenmesi ---
# Bu betiğin çalışması için diğer tüm .py dosyalarınızın
# (datacenter1.py, datacenter_core.py, config_loader.py, dc_config_reader.py)
# aynı dizinde veya Python path'inde olması gerekir.
try:
    # Fiziksel DC Modeli (Sunucular, Rackler)
    from datacenter1 import DataCenter_ITModel, Rack, CPU 
    # Fiziksel DC Konfigürasyonu
    from config_loader import load_dc, HVACParams, DCParams
    # HVAC (Soğutma) Modeli
    from datacenter_core import cop_from_ambient
    # Orijinal (basit) batarya modelini artık KULLANMIYORUZ.
    # from battery_core import Battery, BatteryParams 
except ImportError as e:
    print(f"Hata: Gerekli proje modülleri yüklenemedi: {e}")
    print("Lütfen 'datacenter1.py', 'config_loader.py' vb. dosyaların")
    print("bu betikle aynı dizinde olduğundan emin olun.")
    sys.exit(1)


# --- 'yeni batarya.pdf'  için Yeni Fizik Sınıfı ---

@dataclass
class BatteryConfig:
    """ PDF'teki  batarya parametrelerini tanımlar """
    capacity_mwh: float # C_ESD, C_Z, C_R [cite: 7, 56, 115]
    c_lim: float      # C_lim (şarj oranı) [cite: 19, 68, 115]
    d_lim: float      # D_lim (deşarj oranı) [cite: 23, 68, 115]
    eff_c: float = 0.95 # Şarj verimi [cite: 12, 61, 115]
    eff_d: float = 0.95 # Deşarj verimi [cite: 15, 63, 115]
    min_soc_pct: float = 0.20 # Deep discharge koruması (%20)
    max_soc_pct: float = 0.80 # Aşırı şarj koruması (%80)

class HierarchicalBattery:
    """
    'yeni batarya.pdf'  içindeki denklemleri [cite: 44, 105, 118] ve kısıtları [cite: 46-48, 107-109, 120-122]
    uygulayan batarya sınıfı.
    """
    def __init__(self, config: BatteryConfig):
        self.config = config
        self.min_energy_mwh = config.capacity_mwh * config.min_soc_pct
        self.max_energy_mwh = config.capacity_mwh * config.max_soc_pct
        
        # Başlangıç durumu (örn. minimumda başlar)
        self.energy_mwh = self.min_energy_mwh # E(t) [cite: 38, 84, 115]
        
    @property
    def soc(self) -> float:
        """ 0.0 - 1.0 arası SoC [cite: 35, 81, 115] """
        return self.energy_mwh / self.config.capacity_mwh

    @property
    def max_charge_mw(self) -> float:
        """ PDF Kısıt 47, 108, 121 [cite: 47, 108, 121] """
        return self.config.c_lim * self.config.capacity_mwh

    @property
    def max_discharge_mw(self) -> float:
        """ PDF Kısıt 48, 109, 122 [cite: 48, 109, 122] """
        return self.config.d_lim * self.config.capacity_mwh

    def charge(self, power_mw: float, dt_h: float) -> float:
        """ 
        Bataryayı şarj eder. PDF Denklem 44[cite: 44].
        Kısıtları (güç ve enerji) uygular.
        Dönen değer: Gerçekte şarj edilen GÜÇ (MW).
        """
        # Güç kısıtı (C-rate) [cite: 47, 108, 121]
        power_to_charge_mw = min(power_mw, self.max_charge_mw)
        
        # Enerji kısıtı (Maks SoC)
        available_room_mwh = self.max_energy_mwh - self.energy_mwh
        if available_room_mwh <= 0:
            return 0.0
        
        # Enerjiye çevir
        energy_to_add_mwh = power_to_charge_mw * dt_h * self.config.eff_c
        
        # Kısıtlı enerjiyi uygula
        actual_energy_added_mwh = min(available_room_mwh, energy_to_add_mwh)
        
        # SoC'u güncelle
        self.energy_mwh += actual_energy_added_mwh
        
        # Gerçekte kullanılan gücü döndür
        if dt_h > 0 and self.config.eff_c > 0:
            return (actual_energy_added_mwh / dt_h) / self.config.eff_c
        return 0.0

    def discharge(self, power_mw: float, dt_h: float) -> float:
        """ 
        Bataryayı deşarj eder. PDF Denklem 44 (tersi)[cite: 44].
        Kısıtları (güç ve enerji) uygular.
        Dönen değer: Gerçekte deşarj edilen GÜÇ (MW).
        """
        # Güç kısıtı (C-rate) [cite: 48, 109, 122]
        power_to_discharge_mw = min(power_mw, self.max_discharge_mw)
        
        # Enerji kısıtı (Min SoC)
        available_energy_mwh = self.energy_mwh - self.min_energy_mwh
        if available_energy_mwh <= 0:
            return 0.0
            
        # Enerjiye çevir (Verimlilik deşarjda ters uygulanır [cite: 44])
        energy_to_remove_mwh = (power_to_discharge_mw * dt_h) / self.config.eff_d
        
        # Kısıtlı enerjiyi uygula
        actual_energy_removed_mwh = min(available_energy_mwh, energy_to_remove_mwh)
        
        # SoC'u güncelle
        self.energy_mwh -= actual_energy_removed_mwh
        
        # Gerçekte sağlanan gücü döndür
        if dt_h > 0:
            return (actual_energy_removed_mwh * self.config.eff_d) / dt_h
        return 0.0


# --- ANA SİMÜLASYON YÖNETİCİSİ ---

class HierarchicalSimulation:
    def __init__(self, config_file: str):
        print("Hiyerarşik Simülasyon Başlatılıyor...")
        self.dt_h = 1.0 # Simülasyon zaman adımı (saat) [cite: 26-29]

        # 1. Fiziksel DC Altyapısını Yükle
        # (4 Row, 20 Rack, 4000 CPU)
        self.dc_conf_reader, self.hvac_params, self.dc_params = load_dc(config_file)
        self.n_racks = self.dc_params.num_racks # 20
        self.n_zones = self.dc_conf_reader.NUM_ROWS # 4
        self.racks_per_zone = self.dc_conf_reader.NUM_RACKS_PER_ROW # 5
        
        # dc_config.json'dan en verimli rack'leri bul (kontrol kuralı için)
        self.rack_efficiency_map = sorted(
            range(self.n_racks),
            key=lambda i: self.dc_conf_reader.RACK_SUPPLY_APPROACH_TEMP_LIST[i]
        )

        # 2. Fiziksel IT Modelini (datacenter1.py) Başlat
        # Not: datacenter1.py'nin düzgün çalışması için sunucu yapılandırması gerekir.
        # dc_runner1.py'den alınan örnek yapılandırma kullanılıyor.
        self.dc_it_model = DataCenter_ITModel(
            num_racks=self.n_racks,
            rack_supply_approach_temp_list=self.dc_conf_reader.RACK_SUPPLY_APPROACH_TEMP_LIST,
            rack_CPU_config=[[{"full_load_pwr": 200, "idle_pwr": 50}] * 200] * self.n_racks, # 200 CPU/Rack
            max_W_per_rack=self.dc_params.max_w_per_rack
        )
        # Her bir rack'in maksimum MW gücünü hesapla (örn. 200 CPU * 200W = 0.04 MW)
        # TODO: Bu hesaplama 'datacenter1.py' içinde daha karmaşık olabilir, şimdilik basitleştirildi.
        self.rack_max_mw = 0.04 # 40kW

        # 3. Batarya Hiyerarşisini (PDF ) Başlat
        # GEREKLİ VARSAYIMLAR: Kapasiteleri biz belirliyoruz.
        self.esd_battery = HierarchicalBattery(BatteryConfig(
            capacity_mwh=20.0, c_lim=0.5, d_lim=0.5 # ESD: 20MWh, 10MW limit [cite: 1-48]
        ))
        
        self.zone_batteries: List[HierarchicalBattery] = []
        for i in range(self.n_zones):
            self.zone_batteries.append(HierarchicalBattery(BatteryConfig(
                capacity_mwh=5.0, c_lim=0.4, d_lim=0.4 # Zone: 5MWh, 2MW limit [cite: 49-109]
            )))

        self.rack_batteries: List[HierarchicalBattery] = []
        for i in range(self.n_racks):
            self.rack_batteries.append(HierarchicalBattery(BatteryConfig(
                capacity_mwh=0.5, c_lim=1.0, d_lim=1.0 # Rack: 0.5MWh (UPS), 0.5MW limit 
            )))
            
        # 4. Esnek Yük Kuyruklarını Başlat (Rack başına 1 adet)
        self.rack_flex_queues: List[Deque[float]] = [deque() for _ in range(self.n_racks)]
        # TODO: Kuyruk yönetimi için gecikme cezası parametreleri eklenmeli.

    def step(self, t: pd.Timestamp, total_critical_mw: float, total_flexible_mw: float, price: float, ambient_temp_c: float) -> dict:
        """ Bir saatlik simülasyon adımını çalıştırır """
        
        # --- 1. YÜK DAĞITIMI (Kural Tabanlı Kontrol) ---
        
        # A. Kritik Yük Dağıtımı (Varsayım: Eşit)
        rack_critical_load_mw = np.full(self.n_racks, total_critical_mw / self.n_racks)
        
        # B. Esnek Yük Havuzu (Toplam talebi kuyruğa ekle)
        # Kural: Yükü en verimli rack'in kuyruğuna ata
        efficient_rack_idx = self.rack_efficiency_map[0]
        self.rack_flex_queues[efficient_rack_idx].append(total_flexible_mw)
        
        # C. Esnek Yükü Çalıştırma (Çok basit kural: Fiyat düşükse çalıştır)
        # TODO: Burası "optimizasyon"un (kural tabanlı kontrol) kalbidir.
        # Çok daha karmaşık kurallar (batarya SoC'u, kuyruk uzunluğu vb.) eklenmelidir.
        
        rack_flex_exec_mw = np.zeros(self.n_racks)
        if price < 50.0: # Basit kural
            for i in range(self.n_racks):
                if self.rack_flex_queues[i]:
                    exec_load = self.rack_flex_queues[i].popleft()
                    rack_flex_exec_mw[i] = exec_load
        
        # --- 2. FİZİKSEL IT HESAPLAMASI (datacenter1.py) ---
        
        # Rack yüklerini %'e çevir
        rack_total_load_mw = rack_critical_load_mw + rack_flex_exec_mw
        rack_util_pct = (rack_total_load_mw / self.rack_max_mw) * 100.0
        # Kısıt: %100'ü geçme
        rack_util_pct = np.clip(rack_util_pct, 0, 100)
        
        # datacenter1.py'yi çağır
        # Not: CRAC setpoint (örn 22 C) gereklidir.
        crac_setpoint = 22.0 
        cpu_w_list, fan_w_list, outlet_temp_list = self.dc_it_model.compute_datacenter_IT_load_outlet_temp(
            ITE_load_pct_list=rack_util_pct,
            CRAC_setpoint=crac_setpoint
        )
        
        P_IT_total_mw = (np.sum(cpu_w_list) + np.sum(fan_w_list)) / 1e6
        
        # --- 3. SOĞUTMA HESAPLAMASI (datacenter_core.py) ---
        cop = cop_from_ambient(self.hvac_params, ambient_temp_c)
        P_cooling_mw = P_IT_total_mw / cop
        
        # Veri merkezinin o anki toplam ihtiyacı
        P_DC_total_need_mw = P_IT_total_mw + P_cooling_mw

        # --- 4. HİYERARŞİK ENERJİ AKIŞI (yeni batarya.pdf ) ---
        # Bu, kural tabanlı bir "şelale" (waterfall) mantığıdır.
        # Not: Bu bölüm, PDF'teki  denklemleri tam olarak uygulamak için
        # çok daha detaylı bir mantık (Zone/Rack güç akışı) gerektirir.
        # Bu, basitleştirilmiş bir Makro (ESD) seviyesi uygulamasıdır:

        P_grid_mw = 0.0
        
        # Basit Kural: Fiyat arbitrajı için ESD'yi kullan
        if price < 30.0: # Ucuz
            # DC ihtiyacını karşıla VE bataryayı şarj et
            charge_power = self.esd_battery.charge(self.esd_battery.max_charge_mw, self.dt_h)
            P_grid_mw = P_DC_total_need_mw + charge_power
        elif price > 100.0: # Pahalı
            # DC ihtiyacını bataryadan karşıla
            discharge_power = self.esd_battery.discharge(P_DC_total_need_mw, self.dt_h)
            P_grid_mw = P_DC_total_need_mw - discharge_power
        else: # Normal
            # Sadece DC ihtiyacını karşıla
            P_grid_mw = P_DC_total_need_mw

        cost = P_grid_mw * price

        # --- 5. Sonuçları Kaydet ---
        return {
            "timestamp": t,
            "total_critical_mw": total_critical_mw,
            "total_flexible_mw": total_flexible_mw,
            "P_IT_total_mw": P_IT_total_mw,
            "P_cooling_mw": P_cooling_mw,
            "P_DC_total_need_mw": P_DC_total_need_mw,
            "P_grid_mw": P_grid_mw,
            "cost_eur": cost,
            "esd_soc_pct": self.esd_battery.soc * 100,
            "ambient_temp_c": ambient_temp_c,
            "price_eur_per_mwh": price
        }

# --- ANA ÇALIŞTIRICI ---

def main():
    parser = argparse.ArgumentParser(description="Hiyerarşik Veri Merkezi Simülasyonu")
    parser.add_argument("--workload", required=True, help="İş yükü (kritik+esnek) CSV dosyası (workload_flex.csv)")
    parser.add_argument("--price", required=True, help="Fiyat CSV dosyası (United Kingdom.csv)")
    parser.add_argument("--dc-config", required=True, help="Veri merkezi konfigürasyon JSON dosyası (dc_config.json)")
    parser.add_argument("--out-file", default="results_hierarchical_sim.csv", help="Çıktı CSV dosyasının adı")
    
    args = parser.parse_args()

    # Verileri Yükle
    try:
        workload_df = pd.read_csv(args.workload)
        price_df = pd.read_csv(args.price)
    except FileNotFoundError as e:
        print(f"Hata: Girdi dosyası bulunamadı: {e}")
        return

    # TODO: 'workload_df' ve 'price_df' için 'ensure_timestamp' ve birleştirme işlemleri
    # (dc_optimizer_mpc.py'den kopyalanabilir)
    # Şimdilik basit bir birleştirme varsayılıyor:
    data = workload_df.merge(price_df, on="timestamp") # 'timestamp' sütun adlarının eşleştiğini varsayalım
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    
    # Simülasyonu Başlat
    sim = HierarchicalSimulation(config_file=args.dc_config)
    
    results = []
    
    print("Simülasyon döngüsü başlıyor...")
    for _, row in data.iterrows():
        # Dış ortam sıcaklığı (Hava Durumu) verisi gerekli.
        # Şimdilik sabit bir sıcaklık varsayalım:
        ambient_temp = 20.0 
        
        step_result = sim.step(
            t=row["timestamp"],
            total_critical_mw=row["critical_load_mw"],
            total_flexible_mw=row["flexible_load_mw"],
            price=row["Price (EUR/MWhe)"], # Sütun adının bu olduğunu varsayalım
            ambient_temp_c=ambient_temp
        )
        results.append(step_result)
        
    print("Simülasyon tamamlandı.")
    
    # Sonuçları Kaydet
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.out_file, index=False)
    print(f"✅ Sonuçlar {args.out_file} dosyasına kaydedildi.")

if __name__ == "__main__":
    main()