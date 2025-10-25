import numpy as np

class CPU:
    def __init__(self, full_load_pwr, idle_pwr, cpu_config=None):
        self.full_load_pwr = full_load_pwr
        self.idle_pwr = idle_pwr
        self.cpu_config = cpu_config
        self.m_cpu = None
        self.c_cpu = None
        self.m_itfan = None
        self.c_itfan = None
        self.cpu_curve1()

    def cpu_curve1(self):
        # CPU power curve calculation
        self.m_cpu = (self.cpu_config.CPU_POWER_RATIO_UB[0] - self.cpu_config.CPU_POWER_RATIO_LB[0]) / \
                     (self.cpu_config.INLET_TEMP_RANGE[1] - self.cpu_config.INLET_TEMP_RANGE[0])
        self.c_cpu = self.cpu_config.CPU_POWER_RATIO_LB[0] - self.m_cpu * self.cpu_config.INLET_TEMP_RANGE[0]

    def compute_instantaneous_pwr_vecd(self, inlet_temp, load_pct):
        cpu_power = self.idle_pwr + (self.full_load_pwr - self.idle_pwr) * load_pct / 100.0
        # ❌ Daha önce ITFAN_RATIO vardı → ✔ Şimdi ITFAN_REF_P kullanıyoruz
        itfan_power = self.cpu_config.ITFAN_REF_P  
        return cpu_power, itfan_power


class Rack:
    def __init__(self, CPU_config_list, max_W_per_rack=10000, rack_config=None):
        self.CPU_list = []
        self.rack_config = rack_config
        self.max_W_per_rack = max_W_per_rack
        for CPU_config in CPU_config_list:
            self.CPU_list.append(
                CPU(full_load_pwr=CPU_config["full_load_pwr"],
                    idle_pwr=CPU_config["idle_pwr"],
                    cpu_config=self.rack_config)
            )

    def compute_instantaneous_pwr_vecd(self, inlet_temp, load_pct):
        cpu_pwr, itfan_pwr = 0, 0
        for cpu in self.CPU_list:
            c_pwr, f_pwr = cpu.compute_instantaneous_pwr_vecd(inlet_temp, load_pct)
            cpu_pwr += c_pwr
            itfan_pwr += f_pwr
        return cpu_pwr, itfan_pwr

    def get_current_rack_load(self):
        cpu_pwr, itfan_pwr = self.compute_instantaneous_pwr_vecd(25, 100)
        return cpu_pwr + itfan_pwr


class DataCenter_ITModel:
    def __init__(self, num_racks, rack_supply_approach_temp_list, rack_CPU_config, 
                 max_W_per_rack=10000, DC_ITModel_config=None, chiller_sizing=False):
        self.DC_ITModel_config = DC_ITModel_config
        self.racks_list = []
        self.rack_supply_approach_temp_list = rack_supply_approach_temp_list
        self.rack_CPU_config = rack_CPU_config
        self.rackwise_inlet_temp = []

        for _, CPU_config_list in zip(range(num_racks), self.rack_CPU_config):
            self.racks_list.append(
                Rack(CPU_config_list, max_W_per_rack=max_W_per_rack, rack_config=self.DC_ITModel_config)
            )

        self.total_datacenter_full_load()

    def total_datacenter_full_load(self):
        """Calculate the total DC IT(IT CPU POWER + IT FAN POWER) power consumption"""
        x = [rack_item.compute_instantaneous_pwr_vecd(25, 100)[0] for rack_item in self.racks_list]
        self.total_DC_full_load = sum(x)

    def compute_datacenter_IT_load_outlet_temp(self, ITE_load_pct_list, CRAC_setpoint):
        rackwise_cpu_pwr, rackwise_itfan_pwr, rackwise_outlet_temp = [], [], []
        rackwise_inlet_temp = []

        for rack, rack_supply_approach_temp, ITE_load_pct in zip(
                self.racks_list, self.rack_supply_approach_temp_list, ITE_load_pct_list):
            rack_inlet_temp = rack_supply_approach_temp + CRAC_setpoint
            rackwise_inlet_temp.append(rack_inlet_temp)

            rack_cpu_power, rack_itfan_power = rack.compute_instantaneous_pwr_vecd(rack_inlet_temp, ITE_load_pct)
            rackwise_cpu_pwr.append(rack_cpu_power)
            rackwise_itfan_pwr.append(rack_itfan_power)

            outlet_temp = rack_inlet_temp + (rack_cpu_power + rack_itfan_power) / 10000.0
            rackwise_outlet_temp.append(outlet_temp)

        self.rackwise_inlet_temp = rackwise_inlet_temp
        return rackwise_cpu_pwr, rackwise_itfan_pwr, rackwise_outlet_temp



