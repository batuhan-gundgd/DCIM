from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Deque, Dict
from collections import deque
import numpy as np
import pandas as pd

# Batarya modeli (Battery2 wrapper)
from battery_core import Battery, BatteryParams


# -----------------------------
# Basit batarya arbitrajı
# -----------------------------
def greedy_arbitrage(
    load_mw: np.ndarray,
    price_eur_per_mwh: np.ndarray,
    dt_h: float,
    bat: Battery,
    buy_threshold: Optional[float] = None,
    sell_threshold: Optional[float] = None,
    price_quantiles: Tuple[float, float] = (0.3, 0.7),
) -> pd.DataFrame:
    """
    Batarya ile fiyat arbitrajı (iki eşik kuralı).
    """
    n = len(load_mw)
    assert n == len(price_eur_per_mwh)

    # Eşikler otomatik seçilebilir (kantil)
    if buy_threshold is None or sell_threshold is None:
        q_low = np.quantile(price_eur_per_mwh, price_quantiles[0])
        q_high = np.quantile(price_eur_per_mwh, price_quantiles[1])
        buy_threshold = q_low if buy_threshold is None else buy_threshold
        sell_threshold = q_high if sell_threshold is None else sell_threshold

    recs = []
    for t in range(n):
        p = price_eur_per_mwh[t]
        base = load_mw[t]

        charged_mwh = 0.0
        discharged_mwh = 0.0

        if p <= buy_threshold:
            charged_mwh = bat.charge(bat.params.c_lim * bat.params.capacity_mwh, dt_h)
            grid_import_mwh = base * dt_h + charged_mwh
            action = "charge"
        elif p >= sell_threshold:
            discharged_mwh = bat.discharge(bat.params.d_lim * bat.params.capacity_mwh, dt_h, base)
            grid_import_mwh = max(0.0, base * dt_h - discharged_mwh)
            action = "discharge"
        else:
            grid_import_mwh = base * dt_h
            action = "idle"

        cost_eur = grid_import_mwh * p

        recs.append(dict(
            t=t, price=p, load_mw=base,
            grid_import_mwh=grid_import_mwh,
            soc=bat.soc, action=action,
            charged_mwh=charged_mwh, discharged_mwh=discharged_mwh,
            cost_eur=cost_eur,
        ))

    out = pd.DataFrame(recs)
    out["cum_cost_eur"] = out["cost_eur"].cumsum()
    return out


# -----------------------------
# Esnek yük kaydırma (queue)
# -----------------------------
@dataclass
class LoadShiftParams:
    max_delay_steps: int = 4
    max_exec_mw: Optional[float] = None
    queue_energy_cap_mwh: Optional[float] = None
    drop_penalty_eur_per_mwh: float = 0.0
    delay_penalty_eur_per_mwh: float = 0.0


@dataclass
class FlexJob:
    remaining_delay: int
    mw: float


def _queue_energy_mwh(q: Deque[FlexJob], dt_h: float) -> float:
    return sum(job.mw * dt_h for job in q)


def greedy_load_shift(
    critical_load_mw: np.ndarray,
    flexible_load_mw: np.ndarray,
    price_eur_per_mwh: np.ndarray,
    dt_h: float,
    bat: Battery,
    params: LoadShiftParams = LoadShiftParams(),
    buy_threshold: Optional[float] = None,
    sell_threshold: Optional[float] = None,
    price_quantiles: Tuple[float, float] = (0.3, 0.7),
) -> pd.DataFrame:
    """
    Fiyat tabanlı esnek yük kaydırma + batarya yönetimi.
    """
    n = len(price_eur_per_mwh)
    assert len(critical_load_mw) == n and len(flexible_load_mw) == n

    if buy_threshold is None or sell_threshold is None:
        q_low = np.quantile(price_eur_per_mwh, price_quantiles[0])
        q_high = np.quantile(price_eur_per_mwh, price_quantiles[1])
        buy_threshold = q_low if buy_threshold is None else buy_threshold
        sell_threshold = q_high if sell_threshold is None else sell_threshold

    q: Deque[FlexJob] = deque()
    recs: List[Dict] = []
    cum_delay_penalty = 0.0
    cum_drop_penalty = 0.0

    for t in range(n):
        p = price_eur_per_mwh[t]
        crit = critical_load_mw[t]
        new_flex_mw = flexible_load_mw[t]

        # Yeni iş ekle
        if new_flex_mw > 0:
            q.append(FlexJob(remaining_delay=params.max_delay_steps, mw=new_flex_mw))

        # Delay azalt
        for i in range(len(q)):
            q[i] = FlexJob(q[i].remaining_delay - 1, q[i].mw)

        # Deadline işleri
        must_run = [job for job in q if job.remaining_delay < 0]
        q = deque([job for job in q if job.remaining_delay >= 0])

        exec_flex_mw = sum(job.mw for job in must_run)

        # Ucuz → kuyruktan iş çek
        if p <= buy_threshold:
            extra_allow = float("inf") if params.max_exec_mw is None else max(0.0, params.max_exec_mw - exec_flex_mw)
            pulled = []
            while q and extra_allow > 1e-9:
                job = q.popleft()
                take = min(job.mw, extra_allow)
                exec_flex_mw += take
                extra_allow -= take
                if job.mw - take > 1e-9:
                    q.appendleft(FlexJob(job.remaining_delay, job.mw - take))
            action_note = "cheap-flex"
        else:
            action_note = "expensive-delay"

        # Kuyruk kapasitesi kontrolü
        queued_mwh_before = _queue_energy_mwh(q, dt_h)
        if params.queue_energy_cap_mwh and queued_mwh_before > params.queue_energy_cap_mwh:
            overflow = queued_mwh_before - params.queue_energy_cap_mwh
            dropped = overflow
            cum_drop_penalty += dropped * params.drop_penalty_eur_per_mwh
            # Basit yaklaşım: overflow kadar son işlerden at
            while dropped > 1e-9 and q:
                job = q.pop()
                job_energy = job.mw * dt_h
                if job_energy <= dropped:
                    dropped -= job_energy
                else:
                    kept = (job_energy - dropped) / dt_h
                    q.append(FlexJob(job.remaining_delay, kept))
                    dropped = 0

        # Delay cezası
        queued_mwh_after = _queue_energy_mwh(q, dt_h)
        cum_delay_penalty += queued_mwh_after * params.delay_penalty_eur_per_mwh

        # Toplam yük
        total_load_mw = crit + exec_flex_mw

        # Batarya kararı
        if p <= buy_threshold:
            charged_mwh = bat.charge(bat.params.c_lim * bat.params.capacity_mwh, dt_h)
            grid_import_mwh = total_load_mw * dt_h + charged_mwh
            action = "charge+" + action_note
        elif p >= sell_threshold:
            discharged_mwh = bat.discharge(bat.params.d_lim * bat.params.capacity_mwh, dt_h, total_load_mw)
            grid_import_mwh = max(0.0, total_load_mw * dt_h - discharged_mwh)
            action = "discharge+queue"
        else:
            grid_import_mwh = total_load_mw * dt_h
            action = "idle"

        energy_cost_eur = grid_import_mwh * p
        cost_eur = energy_cost_eur + cum_delay_penalty + cum_drop_penalty

        recs.append(dict(
            t=t, price=p,
            critical_load_mw=crit, exec_flex_mw=exec_flex_mw,
            queued_flex_mwh=queued_mwh_after,
            grid_import_mwh=grid_import_mwh,
            soc=bat.soc, action=action,
            energy_cost_eur=energy_cost_eur,
            cum_delay_penalty_eur=cum_delay_penalty,
            cum_drop_penalty_eur=cum_drop_penalty,
            cost_eur=cost_eur
        ))

    out = pd.DataFrame(recs)
    out["cum_cost_eur"] = out["cost_eur"].cumsum()
    return out

