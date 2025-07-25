import numpy as np
from attrs import field, define
from openmdao.utils.units import convert_units

from h2integrate.core.utilities import merge_shared_inputs
from h2integrate.converters.methanol.methanol_baseclass import (
    MethanolCostConfig,
    MethanolCostBaseClass,
    MethanolFinanceConfig,
    MethanolFinanceBaseClass,
    MethanolPerformanceConfig,
    MethanolPerformanceBaseClass,
)


@define
class CO2HPerformanceConfig(MethanolPerformanceConfig):
    meoh_syn_cat_consume_ratio: float = field()
    ng_consume_ratio: float = field()
    co2_consume_ratio: float = field()
    h2_consume_ratio: float = field()
    elec_consume_ratio: float = field()


class CO2HMethanolPlantPerformanceModel(MethanolPerformanceBaseClass):
    """
    An OpenMDAO component for modeling the performance of a CO2 Hydrogenation (CO2H) methanol
    plant. Computes annual methanol and co-product production, feedstock consumption, and emissions
    based on plant capacity and capacity factor.

    Inputs:
        - meoh_syn_cat_consume_ratio: ratio of ft^3 methanol synthesis catalyst consumed to
            kg methanol produced
        - ng_consume_ratio: ratio of kg natural gas (NG) consumed to kg methanol produced
        - co2_consume_ratio: ratio of kg co2 consumed to kg methanol produced
        - h2_consume_ratio: ratio of kg h2 consumed to kg methanol produced
        - elec_consume_ratio: ratio of kWh electricity consumed to kg methanol produced
        - hydrogen_in: kg/s of hydrogen supplied to methanol reactor - hourly profile
        - electricity_in: kW of electricity supplied to methanol reactor - hourly profile
        - meoh_demand: kg/h of methanol demanded of the plant - hourly profile
    Outputs:
        - meoh_syn_cat_consume: annual consumption of methanol synthesis catalyst (ft**3/yr)
        - ng_consume: hourly consumption of NG (kg/h)
        - co2_consume: co2 consumption in kg/h
        - hydrogen_consume: h2 consumption in kg/h
        - electricity_consume: electricity consumption in kWh/h
    """

    def setup(self):
        self.config = CO2HPerformanceConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance")
        )
        super().setup()

        # Add in tech-specific consumption ratios
        self.add_input(
            "meoh_syn_cat_consume_ratio",
            units="ft**3/kg",
            val=self.config.meoh_syn_cat_consume_ratio,
        )
        self.add_input("ng_consume_ratio", units="kg/kg", val=self.config.ng_consume_ratio)
        self.add_input("co2_consume_ratio", units="kg/kg", val=self.config.co2_consume_ratio)
        self.add_input("h2_consume_ratio", units="kg/kg", val=self.config.h2_consume_ratio)
        self.add_input("elec_consume_ratio", units="kg/kW/h", val=self.config.elec_consume_ratio)

        # Set up feedstock supply inputs
        self.add_input("meoh_syn_cat_in", units="ft**3/yr")
        self.add_input("ng_in", shape=8760, units="kg/h")
        self.add_input("co2_in", shape=8760, units="kg/h")
        self.add_input("hydrogen_in", shape=8760, units="kg/s")
        self.add_input("electricity_in", shape=8760, units="kW*h/h")

        # Set up production demand inputs
        self.add_input("methanol_demand", shape=8760, units="kg/h")

        # Set up feedstock consumption outputs
        self.add_output("meoh_syn_cat_consume", units="ft**3/yr")
        self.add_output("ng_consume", shape=8760, units="kg/h")
        self.add_output("co2_consume", shape=8760, units="kg/h")
        self.add_output("hydrogen_consume", shape=8760, units="kg/s")
        self.add_output("electricity_consume", shape=8760, units="kW*h/h")

    def calc_production_demand(self):
        # If a production demand is not dictated by an upstream component,
        # set production demand equal to the plant capacity * capacity factor
        # Must run prob.setup() before calling to be able to call config
        meoh_cap = self.config.plant_capacity_kgpy
        cap_fac = self.config.capacity_factor
        meoh_demand = np.ones(8760) * meoh_cap / 8760 * cap_fac
        self.set_val("methanol_demand", meoh_demand)

        return meoh_demand

    def size_from_production_demand(self, meoh_demand):
        # If a production demand is dictated by an upstream component,
        # set plant capacity to be able to achieve that demand
        # Must run prob.setup() before calling to be able to call config
        cap_fac = self.config.capacity_factor
        plant_capacity = np.sum(meoh_demand) / cap_fac
        self.set_val("methanol_demand", meoh_demand)

        return plant_capacity

    def calc_feedstock_demand(self, meoh_demand, unconnected_feedstocks):
        # Calculate feedstock demands needed to achieve the production demand
        demand_dict = {}
        meoh_syn_cat_consume_ratio = self.config.meoh_syn_cat_consume_ratio
        ng_consume_ratio = self.config.ng_consume_ratio
        co2_consume_ratio = self.config.co2_consume_ratio
        h2_consume_ratio = self.config.h2_consume_ratio
        elec_consume_ratio = self.config.elec_consume_ratio
        demand_dict["meoh_syn_cat"] = np.sum(meoh_demand) * meoh_syn_cat_consume_ratio
        demand_dict["ng"] = meoh_demand * ng_consume_ratio
        demand_dict["co2"] = meoh_demand * co2_consume_ratio
        demand_dict["hydrogen"] = meoh_demand * h2_consume_ratio
        demand_dict["electricity"] = meoh_demand * elec_consume_ratio
        for feedstock in unconnected_feedstocks:
            self.set_val(feedstock + "_in", demand_dict[feedstock])

        return demand_dict

    def compute(self, inputs, outputs):
        # Calculate max methanol production from each inputs
        syn_in = inputs["meoh_syn_cat_in"]
        ng_in = inputs["ng_in"]
        co2_in = inputs["co2_in"]
        h2_in = inputs["hydrogen_in"] * 3600
        elec_in = inputs["electricity_in"]
        syn_ratio = inputs["meoh_syn_cat_consume_ratio"]
        ng_ratio = inputs["ng_consume_ratio"]
        co2_ratio = inputs["co2_consume_ratio"]
        h2_ratio = inputs["h2_consume_ratio"]
        elec_ratio = inputs["elec_consume_ratio"]
        meoh_from_syn = np.ones(8760) * syn_in / syn_ratio
        meoh_from_ng = ng_in / ng_ratio
        meoh_from_co2 = co2_in / co2_ratio
        meoh_from_h2 = h2_in / h2_ratio
        meoh_from_elec = elec_in / elec_ratio

        # Limiting methanol production per hour
        meoh_prod = np.minimum.reduce(
            [meoh_from_syn, meoh_from_ng, meoh_from_co2, meoh_from_h2, meoh_from_elec]
        )
        meoh_cap = np.ones(8760) * inputs["plant_capacity_kgpy"] / 8760
        meoh_prod = np.minimum.reduce([meoh_prod, meoh_cap])

        # Parse outputs
        outputs["methanol_out"] = meoh_prod
        outputs["meoh_syn_cat_consume"] = np.sum(meoh_prod) * syn_ratio
        outputs["ng_consume"] = meoh_prod * ng_ratio
        outputs["co2_consume"] = meoh_prod * co2_ratio
        outputs["hydrogen_consume"] = meoh_prod * h2_ratio
        outputs["electricity_consume"] = meoh_prod * elec_ratio


@define
class CO2HCostConfig(MethanolCostConfig):
    ng_lhv: float = field()
    meoh_syn_cat_price: float = field()
    ng_price: float = field()
    co2_price: float = field()


class CO2HMethanolPlantCostModel(MethanolCostBaseClass):
    """
    An OpenMDAO component for modeling the cost of a CO2 hydrogenation (CO2H) methanol plant.

    Inputs:
        ng_lhv: natural gas lower heating value in MJ/kg
        meoh_syn_cat_consumption: annual consumption of methanol synthesis catalyst (ft**3/yr)
        ng_consumption: hourly consumption of NG (kg/h)
        carbon_dioxide: hourly consumption of CO2 (kg/h)
        electricity: hourly electricity consumption (kW*h/h)
        meoh_syn_cat_price: price of methanol synthesis catalyst (USD/ft**3)
        ng_price: price of NG (USD/MBtu)
        co2_price: price of CO2 (USD/kg)
    Outputs:
        meoh_syn_cat_cost: annual cost of methanol synthesis catalyst (USD/year)
        ng_cost: annual cost of NG (USD/year)
        co2_cost: annual cost of CO2 (USD/year)
    """

    def setup(self):
        self.config = CO2HCostConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "cost")
        )
        super().setup()

        self.add_input("ng_lhv", units="MJ/kg", val=self.config.ng_lhv)
        self.add_input("meoh_syn_cat_price", units="USD/ft**3", val=self.config.meoh_syn_cat_price)
        self.add_input(
            "ng_price", units="USD/MBtu", val=self.config.ng_price
        )  # TODO: get OpenMDAO to recognize 'MMBtu'
        self.add_input("co2_price", units="USD/kg", val=self.config.co2_price)

        self.add_input("meoh_syn_cat_consume", units="ft**3/yr")
        self.add_input("ng_consume", shape=8760, units="kg/h")
        self.add_input("co2_consume", shape=8760, units="kg/h")
        self.add_input("hydrogen_consume", shape=8760, units="kg/h")
        self.add_input("electricity_consume", shape=8760, units="kW*h/h")

        self.add_output("meoh_syn_cat_cost", units="USD/year")
        self.add_output("ng_cost", units="USD/year")
        self.add_output("co2_cost", units="USD/year")

    def compute(self, inputs, outputs):
        toc_usd = inputs["plant_capacity_kgpy"] * inputs["toc_kg_y"]
        foc_usd_y = inputs["plant_capacity_kgpy"] * inputs["foc_kg_y2"]
        voc_usd_y = np.sum(inputs["methanol_out"]) * inputs["voc_kg"]

        lhv_mj = inputs["ng_lhv"]
        lhv_mmbtu = convert_units(lhv_mj, "MJ", "MBtu")

        outputs["CapEx"] = toc_usd
        outputs["OpEx"] = foc_usd_y + voc_usd_y
        outputs["Fixed_OpEx"] = foc_usd_y
        outputs["Variable_OpEx"] = voc_usd_y
        outputs["meoh_syn_cat_cost"] = inputs["meoh_syn_cat_consume"] * inputs["meoh_syn_cat_price"]
        outputs["ng_cost"] = np.sum(inputs["ng_consume"]) * lhv_mmbtu * inputs["ng_price"]
        outputs["co2_cost"] = np.sum(inputs["co2_consume"]) * inputs["co2_price"]


class CO2HMethanolPlantFinanceModel(MethanolFinanceBaseClass):
    """
    An OpenMDAO component for modeling the financing of a CO2 Hydrogenation (CO2H) methanol plant.

    Inputs:
        meoh_syn_cat_cost: annual cost of synthesis catalyst in USD/year
        ng_cost: annual cost of natural gas in USD/year
        co2_cost: annual cost of CO2 in USD/year
        LCOE: levelized cost of electricity in USD/kWh
        LCOH: levelized cost of hydrogen in USD/kg
    Outputs:
        LCOM_meoh_atr_cat: levelized cost of methanol from ATR catalyst in USD/kg
        LCOM_meoh_syn_cat: levelized cost of methanol from synthesis catalyst in USD/kg
        LCOM_ng: levelized cost of methanol from natural gas in USD/kg
        LCOM_elec: levelized cost of methanol from electricity revenue in USD/kg
    """

    def setup(self):
        self.config = MethanolFinanceConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "finance")
        )
        super().setup()

        self.add_input(
            "meoh_syn_cat_cost",
            units="USD/year",
            desc="Annual cost of synthesis catalyst in USD/year",
        )
        self.add_input(
            "ng_cost",
            units="USD/year",
            desc="Annual cost of natural gas in USD/year",
        )
        self.add_input(
            "co2_cost",
            units="USD/year",
            desc="Annual cost of carbon dioxide in USD/year",
        )
        self.add_input(
            "electricity_in",
            units="kW*h/h",
            desc="Electricity consumption in kWh/h",
            shape_by_conn=True,
        )
        self.add_input(
            "hydrogen_in",
            units="kg/h",
            desc="Hydrogen consumption in kg/h",
            shape_by_conn=True,
        )
        self.add_input(
            "LCOE",
            units="USD/kW/h",
            desc="Levelized cost of electricity in USD/kWh",
        )
        self.add_input(
            "LCOH",
            units="USD/kg",
            desc="Levelized cost of hydrogen in USD/kg",
        )
        self.add_output(
            "LCOM_meoh_syn_cat",
            units="USD/kg",
            desc="Levelized cost of methanol from synthesis catalyst in USD/kg",
        )
        self.add_output(
            "LCOM_ng",
            units="USD/kg",
            desc="Levelized cost of methanol from natural gas in USD/kg",
        )
        self.add_output(
            "LCOM_co2",
            units="USD/kg",
            desc="Levelized cost of methanol from CO2 in USD/kg",
        )
        self.add_output(
            "LCOM_elec",
            units="USD/kg",
            desc="Levelized cost of methanol from electricity in USD/kWh",
        )
        self.add_output(
            "LCOM_h2",
            units="USD/kg",
            desc="Levelized cost of methanol from hydrogen in USD/kWh",
        )

    def compute(self, inputs, outputs):
        kgph = inputs["methanol_out"]

        lcoe = inputs["LCOE"]
        elec = inputs["electricity_in"]
        elec_cost = lcoe * np.sum(elec)
        lcom_elec = elec_cost / np.sum(kgph)
        outputs["LCOM_elec"] = lcom_elec

        lcoh = inputs["LCOH"]
        h2 = inputs["hydrogen_in"]
        h2_cost = lcoh * np.sum(h2)
        lcom_h2 = h2_cost / np.sum(kgph)
        outputs["LCOM_h2"] = lcom_h2

        lcom_capex = (
            inputs["CapEx"]
            * inputs["fixed_charge_rate"]
            * inputs["tasc_toc_multiplier"]
            / np.sum(kgph)
        )
        lcom_fopex = inputs["Fixed_OpEx"] / np.sum(kgph)
        lcom_vopex = inputs["Variable_OpEx"] / np.sum(kgph)
        outputs["LCOM_meoh_capex"] = lcom_capex
        outputs["LCOM_meoh_fopex"] = lcom_fopex

        meoh_syn_cat_cost = inputs["meoh_syn_cat_cost"]
        lcom_meoh_syn_cat = meoh_syn_cat_cost / np.sum(kgph)
        outputs["LCOM_meoh_syn_cat"] = lcom_meoh_syn_cat

        # Correct LCOM_meoh_vopex which initially included catalyst
        lcom_vopex -= lcom_meoh_syn_cat
        outputs["LCOM_meoh_vopex"] = lcom_vopex

        ng_cost = inputs["ng_cost"]
        lcom_ng = ng_cost / np.sum(kgph)
        outputs["LCOM_ng"] = lcom_ng

        co2_cost = inputs["co2_cost"]
        lcom_co2 = co2_cost / np.sum(kgph)
        outputs["LCOM_co2"] = lcom_co2

        lcom_meoh = lcom_capex + lcom_fopex + lcom_vopex + lcom_meoh_syn_cat
        outputs["LCOM_meoh"] = lcom_meoh

        lcom = lcom_meoh + lcom_ng + lcom_elec + lcom_h2 + lcom_co2
        outputs["LCOM"] = lcom
