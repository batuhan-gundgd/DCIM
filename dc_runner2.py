#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dc_runner1.py  —  Data center yük + fiyat + batarya arbitraj simülasyonu

- Girdiler:
    * --workload : workload_flex.csv (timestamp, critical_load_mw, flexible_load_mw)
    * --price    : fiyat CSV'si (timestamp, "Price (EUR/MWhe)" ya da price_eur_mwh vs.)
    * --dc-config: dc_config.json
- Çıktı:
    * --out      : results_extended.csv
- Modlar:
    * arbitrage  : Sadece batarya arbitrajı
    * loadshift  : Basit esnek yük kaydırma + batarya arbitrajı
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd

# Yerel modüller (yüklü dosyalar)
from config_loader import load_dc                      # dc_config.json yükleyici
from datacenter1 import DataCenter_ITModel            # IT modeli (CPU, fan, outlet sıcaklık)
from battery_core import Battery, BatteryParams       # Batarya wrapper (Battery2 üzerinde)

# ---------------------------
# Yardımcı fonksiyonlar
# ---------------------------

def ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Zaman damgası sütununu bulup datetime'a çevirir (ISO ve gün.ay.yıl destekli)."""
    ts_col = None
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("timestamp", "time", "date", "datetime"):
            ts_col = c
            break
    if ts_col is None:
        for c in df.columns:
            if "time" in c.lower():
                ts_col = c
                break
    if ts_col is None:
        raise ValueError("Zaman sütunu bulunamadı (timestamp/time).")

    df = df.copy()

    # Önce ISO formatını dene
    try:
        df["timestamp"] = pd.to_datetime(df[ts_col], format="%Y-%m-%d %H:%M:%S", errors="raise")
    except Exception:
        # ISO değilse dayfirst=True ile serbest parse et
        df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce", dayfirst=True)

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df



def pick_price_column(df: pd.DataFrame) -> str:
    """Fiyat sütun adını akıllıca seç (esnek isim destekler)."""
    candidates = [
        "Price (EUR/MWhe)", "price_eur_mwh", "price_eur_per_mwh",
        "price", "fiyat", "EUR/MWh", "spot_eur_mwh"
    ]
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    # Bulunamazsa, sayısal tipte ve adı price geçen ilk sütunu dene
    for c in df.columns:
        if "price" in c.lower():
            return c
    raise ValueError("Fiyat sütunu bulunamadı. Lütfen sütun adını kontrol edin.")


@dataclass
class Args:
    workload: str
    price: str
    dc_config: str
    out: str
    mode: str = "arbitrage"            # arbitrage | loadshift
    hvac_setpoint_c: float = 22.0      # CRAC setpoint
    dt_minutes: int = 60               # zaman adımı (dakika)
    buy_q: float = 0.30                # düşük fiyat kantili
    sell_q: float = 0.70               # yüksek fiyat kantili
    capacity_mwh: float = 20.0         # batarya kapasitesi (MWh)
    eff_c: float = 0.95                # şarj verimi
    eff_d: float = 0.95                # deşarj verimi
    c_lim: float = 0.5                 # C-rate (şarj)
    d_lim: float = 0.5                 # C-rate (deşarj)
    # Not: BatteryParams içindeki upper/lower limitler Battery2'ye aktarılıyor


# ---------------------------
# Ana simülasyon
# ---------------------------

def run_simulation(a: Args) -> pd.DataFrame:
    # 1) Verileri oku ve zamanla hizala
    w = ensure_timestamp(pd.read_csv(a.workload))
    p = ensure_timestamp(pd.read_csv(a.price))
    price_col = pick_price_column(p)

    # asof-merge (en yakın zamana eşle)
    df = pd.merge_asof(w, p[["timestamp", price_col]], on="timestamp", direction="nearest")
    df.rename(columns={price_col: "price_eur_mwh"}, inplace=True)

    # Zorunlu sütunları doğrula
    for req in ("critical_load_mw", "flexible_load_mw"):
        if req not in df.columns:
            raise ValueError(f"Workload dosyasında '{req}' sütunu yok.")

    # 2) DC konfigürasyonunu yükle ve IT modelini hazırla
    dc_conf, hvac_params, dc_params = load_dc(a.dc_config)
    # RACK_SUPPLY_APPROACH_TEMP_LIST uzunluğu num_racks ile uyumlu olmalı:
    # config dosyasındaki listeyi varsa kullan, yoksa sabit değerle doldur.
    try:
        rack_supply_list = list(dc_conf.RACK_SUPPLY_APPROACH_TEMP_LIST)
        if len(rack_supply_list) != int(dc_params.num_racks):
            raise Exception("Length mismatch")
    except Exception:
        rack_supply_list = [0.5] * int(dc_params.num_racks)

    dc_model = DataCenter_ITModel(
        num_racks=int(dc_params.num_racks),
        rack_supply_approach_temp_list=rack_supply_list,
        rack_CPU_config=dc_conf.RACK_CPU_CONFIG,     # CPU listeleri
        max_W_per_rack=int(dc_params.max_w_per_rack),
        DC_ITModel_config=dc_conf
    )

    # 3) Bataryayı hazırla
    bat = Battery(BatteryParams(
        capacity_mwh=a.capacity_mwh, eff_c=a.eff_c, eff_d=a.eff_d,
        c_lim=a.c_lim, d_lim=a.d_lim
    ))

    # 4) Fiyat eşikleri
    q_low, q_high = np.quantile(df["price_eur_mwh"].values, [a.buy_q, a.sell_q])

    # 5) Zaman adımı (saat)
    dt_h = max(1.0, float(a.dt_minutes)) / 60.0

    # 6) Simülasyon döngüsü
    results: List[dict] = []
    for _, row in df.iterrows():
        crit = float(row["critical_load_mw"])
        flex = float(row["flexible_load_mw"])
        price = float(row["price_eur_mwh"])

        # Basit load shifting (istenirse)
        if a.mode.lower() == "loadshift":
            # çok basit örnek: 0.1 MW'ı ertele (negatif olmasın)
            shifted_flex = max(0.0, flex - 0.1)
        else:
            shifted_flex = flex

        # Veri merkezi toplam yük hedefi (MW)
        dc_total_mw = crit + shifted_flex

        # IT modeli: rack'lere tek tip % yük dağıtımı
        # total_DC_full_load: CPU full-load toplamı (W). % hesap için kullanıyoruz.
        try:
            dc_full_cpu_mw = float(dc_model.total_DC_full_load) / 1e6  # W -> MW
            load_pct = 100.0 * np.clip(dc_total_mw / max(1e-6, dc_full_cpu_mw), 0.0, 1.0)
        except Exception:
            # Yedek: muhafazakâr %50
            load_pct = 50.0

        rack_load_pct = [load_pct] * int(dc_params.num_racks)

        rack_cpu_w, rack_fan_w, rack_outlet_c = dc_model.compute_datacenter_IT_load_outlet_temp(
            ITE_load_pct_list=rack_load_pct,
            CRAC_setpoint=float(a.hvac_setpoint_c)
        )

        # Raporlama için ortam/oda sıcaklığı: raf çıkışlarının ortalaması
        dc_room_temp_c = float(np.mean(rack_outlet_c)) if len(rack_outlet_c) else float(a.hvac_setpoint_c)

        # -------------------------
        # Batarya arbitraj mantığı
        # -------------------------
        charged_mwh = 0.0
        discharged_mwh = 0.0
        action = "idle"

        if price <= q_low:
            # ucuz saat → şarj
            charged_mwh = float(bat.charge(bat.params.c_lim * bat.params.capacity_mwh, dt_h))
            action = "charge"
        elif price >= q_high:
            # pahalı saat → deşarj
            discharged_mwh = float(bat.discharge(bat.params.d_lim * bat.params.capacity_mwh, dt_h, dc_total_mw))
            action = "discharge"

        # Şebekeden çekilen enerji (MWh): yük + şarj - deşarj
        grid_import_mwh = max(0.0, dc_total_mw * dt_h + charged_mwh - discharged_mwh)
        total_power_mw = grid_import_mwh / dt_h  # net güç

        cost_eur = grid_import_mwh * price

        results.append(dict(
            timestamp=row["timestamp"],
            price_eur_mwh=price,
            critical_load_mw=crit,
            flexible_load_mw=flex,
            shifted_flexible_load_mw=shifted_flex,
            dc_total_load_mw=dc_total_mw,
            # IT modeli çıktılarından raporlanabilecekler:
            dc_room_temp_c=dc_room_temp_c,
            # Batarya & enerji metrikleri:
            action=action,
            charged_mwh=charged_mwh,
            discharged_mwh=discharged_mwh,
            battery_soc=bat.soc * 100.0,     # %
            grid_import_mwh=grid_import_mwh,
            total_power_mw=total_power_mw,
            cost_eur=cost_eur
        ))

    out = pd.DataFrame(results).sort_values("timestamp").reset_index(drop=True)
    out["cum_cost_eur"] = out["cost_eur"].cumsum()
    return out


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> Args:
    ap = argparse.ArgumentParser(description="Data center + battery arbitrage simulator")
    ap.add_argument("--workload", required=True)
    ap.add_argument("--price", required=True)
    ap.add_argument("--dc-config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["arbitrage", "loadshift"], default="arbitrage")

    ap.add_argument("--hvac-setpoint", type=float, default=22.0, dest="hvac_setpoint_c")
    ap.add_argument("--dt-minutes", type=int, default=60)

    ap.add_argument("--buy-q", type=float, default=0.30)
    ap.add_argument("--sell-q", type=float, default=0.70)

    ap.add_argument("--capacity-mwh", type=float, default=20.0)
    ap.add_argument("--eff-c", type=float, default=0.95)
    ap.add_argument("--eff-d", type=float, default=0.95)
    ap.add_argument("--c-lim", type=float, default=0.5)
    ap.add_argument("--d-lim", type=float, default=0.5)

    p = ap.parse_args()
    return Args(
        workload=p.workload,
        price=p.price,
        dc_config=p.dc_config,
        out=p.out,
        mode=p.mode,
        hvac_setpoint_c=p.hvac_setpoint_c,
        dt_minutes=p.dt_minutes,
        buy_q=p.buy_q,
        sell_q=p.sell_q,
        capacity_mwh=p.capacity_mwh,
        eff_c=p.eff_c,
        eff_d=p.eff_d,
        c_lim=p.c_lim,
        d_lim=p.d_lim
    )


def main():
    args = parse_args()
    df = run_simulation(args)
    df.to_csv(args.out, index=False)
    print(f"✅ Results saved to {args.out}")
    print(f"   Rows: {len(df)} | Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()