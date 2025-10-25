import argparse
import pandas as pd
import numpy as np

from battery_core import Battery, BatteryParams
from load_shift_price import greedy_arbitrage, greedy_load_shift, LoadShiftParams
from datacenter import DataCenter_ITModel
from config_loader import load_dc


def run_simulation(workload_csv, price_csv, dc_config=None, dt_minutes=60,
                   mode="arbitrage", out_csv="results.csv", crac_setpoint=22.0):
    """
    DataCenter + Battery + Load Shift Simülasyonu
    """
    dt_h = dt_minutes / 60.0

    # --- 1) Veri oku
    workload = pd.read_csv(workload_csv)
    price = pd.read_csv(price_csv)

    # --- 2) Timestamp temizliği
    workload["timestamp"] = workload["timestamp"].astype(str).str.strip()
    price["timestamp"] = price["timestamp"].astype(str).str.strip()

    # Workload (TR formatı olabiliyor) ve Price (ISO sık görülüyor)
    workload["timestamp"] = pd.to_datetime(workload["timestamp"], dayfirst=True, errors="coerce")
    price["timestamp"] = pd.to_datetime(price["timestamp"], errors="coerce")

    # Geçersiz timestamp'leri at
    workload = workload.dropna(subset=["timestamp"]).copy()
    price = price.dropna(subset=["timestamp"]).copy()

    # Merge öncesi sırala
    workload = workload.sort_values("timestamp")
    price = price.sort_values("timestamp")

    # --- 3) Merge (inner)
    df = pd.merge(workload, price, on="timestamp", how="inner")

    # --- 4) Kritik ve esnek yük ayrımı
    if "critical_load_mw" not in df.columns and "flexible_load_mw" not in df.columns:
        if "dc_load_mw" in df.columns:
            df["critical_load_mw"] = df["dc_load_mw"] * 0.7
            df["flexible_load_mw"] = df["dc_load_mw"] * 0.3
        else:
            raise ValueError("Workload CSV içinde ne 'dc_load_mw' ne de 'critical_load_mw/flexible_load_mw' bulunuyor!")

    # --- 5) DataCenter IT + HVAC
    dc_model = None
    dcparams = None
    if dc_config:
        # load_dc -> (cfg: DC_Config, hvac: HVACParams, dcparams: DCParams)
        cfg, hvac, dcparams = load_dc(dc_config)
        dc_model = DataCenter_ITModel(
            dcparams.num_racks,
            cfg.RACK_SUPPLY_APPROACH_TEMP_LIST,
            cfg.RACK_CPU_CONFIG,
            DC_ITModel_config=cfg
        )

    if dc_model is not None:
        # Toplam DC tam yükü (W) -> MW
        full_load_mw = dc_model.total_DC_full_load / 1000.0
        num_racks = int(dcparams.num_racks)

        it_power, hvac_power, room_temp = [], [], []

        total_load_series = (df["critical_load_mw"].astype(float) + df["flexible_load_mw"].astype(float)).to_numpy()

        for load_mw in total_load_series:
            # MW -> %util; delta<2 hatasını önlemek için alt sınır uygula
            if full_load_mw <= 0 or not np.isfinite(full_load_mw):
                util_pct = 50.0
            else:
                util_pct = (load_mw / full_load_mw) * 100.0
            util_pct = float(np.clip(util_pct, 5.0, 100.0))

            # Her rack aynı yüzde yükte çalışsın (ileride dağılım eklenebilir)
            rackwise_cpu_w, rackwise_fan_w, rackwise_outlet_c = dc_model.compute_datacenter_IT_load_outlet_temp(
                [util_pct] * num_racks,
                CRAC_setpoint=float(crac_setpoint)
            )

            # IT gücü (W) -> MW
            p_it_mw = (sum(rackwise_cpu_w) + sum(rackwise_fan_w)) / 1000.0
            # HVAC basit model: %30 (istersen COP tabanlıya çevirebiliriz)
            p_hvac_mw = 0.3 * p_it_mw

            it_power.append(p_it_mw)
            hvac_power.append(p_hvac_mw)
            room_temp.append(float(np.mean(rackwise_outlet_c)))

        df["it_power_mw"] = np.array(it_power)
        df["hvac_power_mw"] = np.array(hvac_power)
        df["room_temp_c"] = room_temp
        df["total_dc_load_mw"] = df["it_power_mw"] + df["hvac_power_mw"]
    else:
        # Basit model: HVAC = IT’in %30’u
        df["it_power_mw"] = df["critical_load_mw"] + df["flexible_load_mw"]
        df["hvac_power_mw"] = 0.3 * df["it_power_mw"]
        df["room_temp_c"] = np.nan
        df["total_dc_load_mw"] = df["it_power_mw"] + df["hvac_power_mw"]

    # --- 6) Batarya
    bat = Battery(BatteryParams(
        capacity_mwh=20, eff_c=0.95, eff_d=0.95, c_lim=0.5, d_lim=0.5
    ))

    # --- 7) Strateji
    # Fiyat sütunu ismi güvenliği (senin dosyanda price_eur_per_mwh var)
    price_col = None
    for cand in ["price_eur_per_mwh", "price_usd_per_mwh", "price", "elec_price"]:
        if cand in df.columns:
            price_col = cand
            break
    if price_col is None:
        raise ValueError("Fiyat sütunu bulunamadı. Lütfen CSV'de 'price_eur_per_mwh' (veya 'price') olduğundan emin ol.")

    if mode == "arbitrage":
        res = greedy_arbitrage(
            load_mw=df["total_dc_load_mw"].values,
            price_eur_per_mwh=df[price_col].values,
            dt_h=dt_h,
            bat=bat,
            price_quantiles=(0.3, 0.7),
        )
    elif mode == "loadshift":
        ls_params = LoadShiftParams(
            max_delay_steps=3,
            max_exec_mw=10,
            queue_energy_cap_mwh=30,
            drop_penalty_eur_per_mwh=200,
            delay_penalty_eur_per_mwh=1,
        )
        res = greedy_load_shift(
            critical_load_mw=df["critical_load_mw"].values,
            flexible_load_mw=df["flexible_load_mw"].values,
            price_eur_per_mwh=df[price_col].values,
            dt_h=dt_h,
            bat=bat,
            params=ls_params,
            price_quantiles=(0.3, 0.7),
        )
    else:
        raise ValueError("mode must be 'arbitrage' or 'loadshift'")

    # --- 8) Sonuçları kaydet
    df_out = pd.concat([df.reset_index(drop=True), res.reset_index(drop=True)], axis=1)
    df_out.to_csv(out_csv, index=False)
    print(f"✅ Results saved to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=str, required=True, help="Workload CSV")
    parser.add_argument("--price", type=str, required=True, help="Price CSV")
    parser.add_argument("--mode", type=str, default="arbitrage",
                        choices=["arbitrage", "loadshift"], help="Simulation mode")
    parser.add_argument("--out", type=str, default="results.csv", help="Output CSV")
    parser.add_argument("--dt", type=int, default=60, help="Timestep in minutes")
    parser.add_argument("--dc-config", type=str, default=None, help="Datacenter config JSON (dc_config.json)")
    parser.add_argument("--crac-setpoint", type=float, default=22.0, help="CRAC setpoint (°C)")
    args = parser.parse_args()

    run_simulation(
        workload_csv=args.workload,
        price_csv=args.price,
        dc_config=args.dc_config,
        dt_minutes=args.dt,
        mode=args.mode,
        out_csv=args.out,
        crac_setpoint=args.crac_setpoint,
    )







