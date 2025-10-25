
from typing import Tuple, List
import numpy as np
import sys
sys.path.append("/mnt/data")
import datacenter as dc
from config_loader import HVACParams

def compute_it_power_mw(
    rack_utils_pct: np.ndarray,
    crac_setpoint_c: float,
    dc_model: dc.DataCenter_ITModel,
) -> Tuple[float, float, float]:
    """Returns (IT_total_mw, cpu_mw, itfan_mw)."""
    cpu_w, itfan_w, _ = dc_model.compute_datacenter_IT_load_outlet_temp(
        ITE_load_pct_list=[float(u) for u in rack_utils_pct],
        CRAC_setpoint=float(crac_setpoint_c),
    )
    cpu_w = float(np.sum(cpu_w))
    itf_w = float(np.sum(itfan_w))
    it_mw = (cpu_w + itf_w) / 1e6
    return it_mw, cpu_w/1e6, itf_w/1e6

def cop_from_ambient(hvac: HVACParams, t_amb_c: float) -> float:
    cop = max(1.0, hvac.CHILLER_COP_BASE - hvac.CHILLER_COP_K * (t_amb_c - hvac.CHILLER_COP_T_NOMINAL))
    return cop

def total_dc_power_mw(
    rack_utils_pct: np.ndarray,
    crac_setpoint_c: float,
    t_amb_c: float,
    dc_model: dc.DataCenter_ITModel,
    hvac: HVACParams,
) -> Tuple[float, float, float, float]:
    """Returns (total_mw, it_mw, cool_mw, cop)."""
    it_mw, _, _ = compute_it_power_mw(rack_utils_pct, crac_setpoint_c, dc_model)
    cop = cop_from_ambient(hvac, t_amb_c)
    cool_mw = it_mw / cop
    return it_mw + cool_mw, it_mw, cool_mw, cop
