
import argparse, pandas as pd, numpy as np, sys, os
sys.path.append("/mnt/data")
import datacenter as dc
from battery_core import Battery, BatteryParams
from load_shift_price import greedy_arbitrage
from config_loader import load_dc
from datacenter_core import total_dc_power_mw

def parse_args():
    p = argparse.ArgumentParser("Detailed DC arbitrage (rack utils + CRAC + ambient + price)")
    p.add_argument("--rackutils", required=True, help="CSV with columns: timestamp, rack_1..rack_N, crac_setpoint_c, t_amb_c")
    p.add_argument("--price", required=True, help="CSV with columns: timestamp, <price_col>")
    p.add_argument("--price-col", default="price_eur_per_mwh")
    p.add_argument("--currency", choices=["EUR","USD","GBP"], default="EUR")
    p.add_argument("--fx-to-eur", type=float, default=1.0, help="Multiply price by this to convert to EUR")
    p.add_argument("--dc-config", default="/mnt/data/dc_config.json")
    p.add_argument("--dt", type=int, default=15)
    p.add_argument("--buy-q", type=float, default=0.3)
    p.add_argument("--sell-q", type=float, default=0.7)
    p.add_argument("--bat-cap", type=float, default=20.0)
    p.add_argument("--eff-c", type=float, default=0.95)
    p.add_argument("--eff-d", type=float, default=0.95)
    p.add_argument("--c-lim", type=float, default=0.5)
    p.add_argument("--d-lim", type=float, default=0.5)
    p.add_argument("--out", default="results_detailed.csv")
    return p.parse_args()

def main():
    a = parse_args()
    df = pd.read_csv(a.rackutils, parse_dates=["timestamp"]).sort_values("timestamp")
    pr = pd.read_csv(a.price, parse_dates=["timestamp"])
    if a.price_col not in pr.columns:
        raise ValueError(f"Price column '{a.price_col}' not found. Available: {list(pr.columns)}")
    pr = pr[["timestamp", a.price_col]].rename(columns={a.price_col: "price_raw"})
    pr["price_eur_per_mwh"] = pr["price_raw"] * a.fx_to_eur
    df = df.merge(pr, on="timestamp", how="inner")

    cfg, hvac, dcp = load_dc(a.dc_config)
    # Build DC IT model once
    dc_model = dc.DataCenter_ITModel(
        num_racks=int(cfg.NUM_RACKS),
        rack_supply_approach_temp_list=cfg.RACK_SUPPLY_APPROACH_TEMP_LIST,
        rack_CPU_config=cfg.RACK_CPU_CONFIG,
        max_W_per_rack=int(cfg.MAX_W_PER_RACK),
        DC_ITModel_config=cfg,
    )

    # rack_* kolonlarını sırala
    rack_cols = sorted([c for c in df.columns if c.startswith("rack_")], key=lambda x: int(x.split("_")[-1]))
    totals = []
    for _, row in df.iterrows():
        rack_utils = row[rack_cols].to_numpy(dtype=float)
        total_mw, it_mw, cool_mw, cop = total_dc_power_mw(
            rack_utils_pct=rack_utils,
            crac_setpoint_c=float(row["crac_setpoint_c"]),
            t_amb_c=float(row["t_amb_c"]),
            dc_model=dc_model,
            hvac=hvac,
        )
        totals.append((total_mw, it_mw, cool_mw, cop))
    totals = np.array(totals)
    df["dc_total_mw"] = totals[:,0]
    df["dc_it_mw"]    = totals[:,1]
    df["dc_cool_mw"]  = totals[:,2]
    df["cop"]         = totals[:,3]

    dt_h = a.dt / 60.0
    bat = Battery(BatteryParams(
        capacity_mwh=a.bat_cap, eff_c=a.eff_c, eff_d=a.eff_d, c_lim=a.c_lim, d_lim=a.d_lim
    ))
    res = greedy_arbitrage(
        load_mw=df["dc_total_mw"].to_numpy(),
        price_eur_per_mwh=df["price_eur_per_mwh"].to_numpy(),
        dt_h=dt_h, bat=bat, price_quantiles=(a.buy_q, a.sell_q)
    )
    out = pd.concat([df.reset_index(drop=True), res.drop(columns=["t"])], axis=1)
    out["cum_cost_eur"] = out["cost_eur"].cumsum()
    out.to_csv(a.out, index=False)
    print(f"Wrote {a.out}  rows={len(out)}  total_cost_eur={out['cost_eur'].sum():,.2f}")

if __name__ == "__main__":
    main()
