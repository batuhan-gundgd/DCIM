import argparse
import pandas as pd
import numpy as np
import json
from config_loader import load_dc
from datacenter1 import DataCenter_ITModel

def ensure_timestamp(df):
    if "timestamp" not in df.columns:
        for col in df.columns:
            if "time" in col.lower():
                df["timestamp"] = df[col]
                break
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df

def run_simulation(workload_file, price_file, dc_config_file, out_file, mode):
    # Load workload & price
    workload = pd.read_csv(workload_file)
    price = pd.read_csv(price_file)

    workload = ensure_timestamp(workload)
    price = ensure_timestamp(price)

    # Config
    dc_conf, hvac_params, dc_params = load_dc(dc_config_file)
    dc_model = DataCenter_ITModel(
        num_racks=dc_params.num_racks,
        rack_supply_approach_temp_list=[0.5] * dc_params.num_racks,
        rack_CPU_config=[[{"full_load_pwr": 200, "idle_pwr": 50}] * 10] * dc_params.num_racks,
        max_W_per_rack=dc_params.max_w_per_rack,
        DC_ITModel_config=dc_conf,
    )

    # Merge workload with price
    df = pd.merge_asof(workload, price, on="timestamp", direction="nearest")

    # Init SOC
    soc = 50.0
    soc_list = []
    hvac_setpoint = 22.0  # sabit, ileride config’ten çekilebilir

    results = []
    timestep_h = 0.25  # 15 dakikalık zaman adımı

    for _, row in df.iterrows():
        crit = row.get("critical_load_mw", 0.0)
        flex = row.get("flexible_load_mw", 0.0)

        if mode == "loadshift":
            shifted_flex = max(0.0, flex - 0.1)  # basit load shifting örneği
        else:
            shifted_flex = flex

        dc_total = crit + shifted_flex
        load = [dc_total / dc_params.num_racks] * dc_params.num_racks

        # IT model
        rackwise_cpu, rackwise_fan, rackwise_outlet = dc_model.compute_datacenter_IT_load_outlet_temp(
            load, hvac_setpoint
        )
        cpu_power = sum(rackwise_cpu) / 1000.0  # kW → MW
        fan_power = sum(rackwise_fan) / 1000.0
        total_power = cpu_power + fan_power

        # Enerji MWh
        cpu_energy = cpu_power * timestep_h
        fan_energy = fan_power * timestep_h
        total_energy = total_power * timestep_h

        # Oda sıcaklığı
        dc_room_temp = np.mean(rackwise_outlet)

        # Battery SoC (örnek model)
        soc += (flex - shifted_flex) * timestep_h * 10  # ölçekleme basit
        soc = max(0, min(100, soc))

        # Maliyet
        price_eur = row.get("Price (EUR/MWhe)", 0.0)
        cost = total_energy * price_eur

        results.append({
            "timestamp": row["timestamp"],
            "critical_load_mw": crit,
            "flexible_load_mw": flex,
            "shifted_flexible_load_mw": shifted_flex,
            "dc_total_load_mw": dc_total,
            "total_power_mw": total_power,
            "cpu_energy_mwh": cpu_energy,
            "fan_energy_mwh": fan_energy,
            "energy_mwh": total_energy,
            "dc_room_temp_c": dc_room_temp,
            "battery_soc": soc,
            "hvac_setpoint_c": hvac_setpoint,
            "cost_eur_corrected": cost,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_file, index=False)
    print(f"✅ Results saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", required=True)
    parser.add_argument("--price", required=True)
    parser.add_argument("--dc-config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["arbitrage", "loadshift"], default="arbitrage")
    args = parser.parse_args()

    run_simulation(args.workload, args.price, args.dc_config, args.out, args.mode)


















