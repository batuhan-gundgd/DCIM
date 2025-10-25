
from dataclasses import dataclass
from typing import Optional
import sys
sys.path.append("/mnt/data")
import battery_model as bm  # uses user's Battery2

@dataclass
class BatteryParams:
    capacity_mwh: float = 20.0
    eff_c: float = 0.95
    eff_d: float = 0.95
    c_lim: float = 0.5
    d_lim: float = 0.5
    upper_u: float = -0.04
    upper_v: float = 1.0
    lower_u: float = 0.01
    lower_v: float = 0.0

class Battery:
    """Thin wrapper around Battery2 with MWh+MW semantics."""
    def __init__(self, params: BatteryParams):
        self.params = params
        self.model = bm.Battery2(
            capacity=params.capacity_mwh,
            current_load=0.0,
            eff_c=params.eff_c,
            eff_d=params.eff_d,
            c_lim=params.c_lim,
            d_lim=params.d_lim,
            upper_u=params.upper_u,
            upper_v=params.upper_v,
            lower_u=params.lower_u,
            lower_v=params.lower_v,
        )

    @property
    def soc(self) -> float:
        return float(self.model.get_battery_soc())

    def reset(self):
        self.model.reset()

    def charge(self, power_mw: float, dt_h: float) -> float:
        """Returns energy absorbed (MWh, AC side)."""
        pre = self.model.current_load
        self.model.charge(power_mw, dt_h)
        post = self.model.current_load
        return float(post - pre)

    def discharge(self, power_mw: float, dt_h: float, dc_load_mw: float) -> float:
        """Returns discharged energy delivered to DC (MWh)."""
        return float(self.model.discharge(power_mw, dt_h, dc_load_mw))
