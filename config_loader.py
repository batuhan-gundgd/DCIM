
from dataclasses import dataclass
from typing import Any, Dict
import json, os, sys
sys.path.append("/mnt/data")
from dc_config_reader import DC_Config

@dataclass
class HVACParams:
    CHILLER_COP_BASE: float = 4.5
    CHILLER_COP_K: float = 0.1
    CHILLER_COP_T_NOMINAL: float = 25.0

@dataclass
class DCParams:
    num_racks: int
    max_w_per_rack: int

def load_dc(dc_config_path: str = "/mnt/data/dc_config.json"):
    # DC_Config wraps json; it expects only basename
    cfg = DC_Config(dc_config_file=os.path.basename(dc_config_path), datacenter_capacity_mw=1.0)
    with open(dc_config_path, "r") as f:
        j = json.load(f)
    hvac_json = j.get("hvac_configuration", {})
    hvac = HVACParams(
        CHILLER_COP_BASE=float(hvac_json.get("CHILLER_COP_BASE", 4.5)),
        CHILLER_COP_K=float(hvac_json.get("CHILLER_COP_K", 0.1)),
        CHILLER_COP_T_NOMINAL=float(hvac_json.get("CHILLER_COP_T_NOMINAL", 25.0)),
    )
    dcparams = DCParams(num_racks=int(cfg.NUM_RACKS), max_w_per_rack=int(cfg.MAX_W_PER_RACK))
    return cfg, hvac, dcparams
