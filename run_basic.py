
import argparse, pandas as pd, numpy as np
from battery_core import Battery, BatteryParams
from load_shift_price import greedy_arbitrage

def parse_args():
    p = argparse.ArgumentParser("Basic DC arbitrage (aggregate MW + price)")
    p.add_argument("--workload", required=True, help="CSV with columns: timestamp, dc_load_mw")
    p.add_argument("--price", required=True, help="CSV with columns: timestamp, <price_col>")
    p.add_argument("--price-col", default="price_eur_per_mwh")
    p.add_argument("--currency", choices=["EUR","USD","GBP"], default="EUR")
    p.add_argument("--fx-to-eur", type=float, default=1.0, help="Multiply price by this to convert to EUR (e.g., USD->EUR)")
    p.add_argument("--dt", type=int, default=15, help="minutes")
    p.add_argument("--bat-cap", type=float, default=20.0)
    p.add_argument("--eff-c", type=float, default=0.95)
    p.add_argument("--eff-d", type=float, default=0.95)
    p.add_argument("--c-lim", type=float, default=0.5)
    p.add_argument("--d-lim", type=float, default=0.5)
    p.add_argument("--buy-q", type=float, default=0.3)
    p.add_argument("--sell-q", type=float, default=0.7)
    p.add_argument("--out", default="results_basic.csv")
    return p.parse_args()

def main():
    a = parse_args()
    wl = pd.read_csv(a.workload, parse_dates=["timestamp"])
    pr = pd.read_csv(a.price, parse_dates=["timestamp"])
    if a.price_col not in pr.columns:
        raise ValueError(f"Price column '{a.price_col}' not found. Available: {list(pr.columns)}")
    pr = pr[["timestamp", a.price_col]].rename(columns={a.price_col: "price_raw"})
    pr["price_eur_per_mwh"] = pr["price_raw"] * a.fx_to_eur  # user handles FX ex-ante
    df = wl.merge(pr, on="timestamp", how="inner").sort_values("timestamp")
    dt_h = a.dt / 60.0

    bat = Battery(BatteryParams(
        capacity_mwh=a.bat_cap, eff_c=a.eff_c, eff_d=a.eff_d, c_lim=a.c_lim, d_lim=a.d_lim
    ))
    res = greedy_arbitrage(
        load_mw=df["dc_load_mw"].to_numpy(),
        price_eur_per_mwh=df["price_eur_per_mwh"].to_numpy(),
        dt_h=dt_h, bat=bat, price_quantiles=(a.buy_q, a.sell_q)
    )
    out = pd.concat([df.reset_index(drop=True), res.drop(columns=["t"])], axis=1)
    out["cum_cost_eur"] = out["cost_eur"].cumsum()
    out.to_csv(a.out, index=False)
    print(f"Wrote {a.out}  rows={len(out)}  total_cost_eur={out['cost_eur'].sum():,.2f}")

if __name__ == "__main__":
    main()
