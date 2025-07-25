import numpy as np
from attrs import field, define

from h2integrate.core.utilities import BaseConfig, merge_shared_inputs
from h2integrate.core.validators import gt_zero, contains
from h2integrate.tools.eco.utilities import ceildiv
from h2integrate.converters.hydrogen.electrolyzer_baseclass import ElectrolyzerPerformanceBaseClass
from h2integrate.simulation.technologies.hydrogen.electrolysis.run_h2_PEM import run_h2_PEM


@define
class ECOElectrolyzerPerformanceModelConfig(BaseConfig):
    """
    Configuration class for the ECOElectrolyzerPerformanceModel.

    Args:
        sizing (dict): A dictionary containing the following model sizing parameters:
            - resize_for_enduse (bool): Flag to adjust the electrolyzer based on the enduse.
            - size_for (str): Determines the sizing strategy, either "BOL" (generous), or
                "EOL" (conservative).
            - hydrogen_dmd (#TODO): #TODO
        rating (float): The rating of the electrolyzer in MW.
        location (str): The location of the electrolyzer; options include "onshore" or "offshore".
        cluster_rating_MW (float): The rating of the clusters that the electrolyzer is grouped
            into, in MW.
        pem_control_type (str): The control strategy to be used by the electrolyzer.
        eol_eff_percent_loss (float): End-of-life (EOL) defined as a percent change in efficiency
            from beginning-of-life (BOL).
        uptime_hours_until_eol (int): Number of "on" hours until the electrolyzer reaches EOL.
        include_degradation_penalty (bool): Flag to include degradation of the electrolyzer due to
            operational hours, ramping, and on/off power cycles.
        turndown_ratio (float): The ratio at which the electrolyzer will shut down.
        electrolyzer_capex (int): $/kW overnight installed capital costs for a 1 MW system in
            2022 USD/kW (DOE hydrogen program record 24005 Clean Hydrogen Production Cost Scenarios
            with PEM Electrolyzer Technology 05/20/24) #TODO: convert to refs
            (https://www.hydrogen.energy.gov/docs/hydrogenprogramlibraries/pdfs/24005-clean-hydrogen-production-cost-pem-electrolyzer.pdf?sfvrsn=8cb10889_1)
    """

    sizing: dict = field()
    rating: float = field(validator=gt_zero)
    location: str = field(validator=contains(["onshore", "offshore"]))
    cluster_rating_MW: float = field(validator=gt_zero)
    pem_control_type: str = field(validator=contains(["basic"]))
    eol_eff_percent_loss: float = field(validator=gt_zero)
    uptime_hours_until_eol: int = field(validator=gt_zero)
    include_degradation_penalty: bool = field()
    turndown_ratio: float = field(validator=gt_zero)
    electrolyzer_capex: int = field()
    capacity_factor: float = field()
    elec_consume_ratio: float = field()


class ECOElectrolyzerPerformanceModel(ElectrolyzerPerformanceBaseClass):
    """
    An OpenMDAO component that wraps the PEM electrolyzer model.
    Takes electricity input and outputs hydrogen and oxygen generation rates.
    """

    def setup(self):
        super().setup()
        self.config = ECOElectrolyzerPerformanceModelConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance"),
            strict=False,
        )
        self.add_output("efficiency", val=0.0, desc="Average efficiency of the electrolyzer")
        self.add_output(
            "rated_h2_production_kg_pr_hr",
            val=0.0,
            units="kg/h",
            desc="Rated hydrogen production of system in kg/hour",
        )

        self.add_input(
            "electrolyzer_size_mw",
            val=self.config.rating,
            units="MW",
            desc="Size of the electrolyzer in MW",
        )
        self.add_output(
            "electrolyzer_size_perf",
            units="MW",
            desc="Size of the electrolyzer in MW",
        )

        self.add_input("elec_consume_ratio", val=self.config.elec_consume_ratio, units="kW*h/kg")
        self.add_input("hydrogen_demand", units="kg/h")
        self.add_output("electricity_consume", shape=8760, units="kW*h/h")

    def calc_production_demand(self):
        # If a production demand is not dictated by an upstream component,
        # set production demand equal to the plant capacity * capacity factor
        # Must run prob.setup() before calling to be able to call config
        rating_mw = self.config.cluster_size_mw
        cap_fac = self.config.capacity_factor
        elec_ratio = self.config.elec_consume_ratio
        h2_demand = np.ones(8760) * rating_mw * 1000 / elec_ratio * cap_fac
        self.set_val("hydrogen_demand", h2_demand)

        return h2_demand

    def size_from_production_demand(self, hydrogen_demand):
        # If a production demand is dictated by an upstream component,
        # set plant capacity to be able to achieve that demand
        # Must run prob.setup() before calling to be able to call config
        cap_fac = self.config.capacity_factor
        elec_ratio = self.config.elec_consume_ratio
        rating = np.sum(hydrogen_demand) / 8760 * elec_ratio / cap_fac / 1000
        self.set_val("electrolyzer_size_mw", rating)

        return rating

    def calc_feedstock_demand(self, hydrogen_demand):
        # Calculate feedstock demands needed to achieve the production demand
        demand_dict = {}
        elec_consume_ratio = self.config.elec_consume_ratio
        demand_dict["electricity"] = hydrogen_demand * elec_consume_ratio

        return demand_dict

    def compute(self, inputs, outputs):
        plant_life = self.options["plant_config"]["plant"]["plant_life"]
        electrolyzer_size_mw = inputs["electrolyzer_size_mw"]
        outputs["electrolyzer_size_perf"] = electrolyzer_size_mw
        electrolyzer_capex_kw = self.config.electrolyzer_capex

        # # IF GRID CONNECTED
        # if plant_config["plant"]["grid_connection"]:
        #     # NOTE: if grid-connected, it assumes that hydrogen demand is input and there is not
        #     # multi-cluster control strategies.
        # This capability exists at the cluster level, not at the
        #     # system level.
        #     if config["sizing"]["hydrogen_dmd"] is not None:
        #         grid_connection_scenario = "grid-only"
        #         hydrogen_production_capacity_required_kgphr = config[
        #             "sizing"
        #         ]["hydrogen_dmd"]
        #         energy_to_electrolyzer_kw = []
        #     else:
        #         grid_connection_scenario = "off-grid"
        #         hydrogen_production_capacity_required_kgphr = []
        #         energy_to_electrolyzer_kw = np.ones(8760) * electrolyzer_size_mw * 1e3
        # # IF NOT GRID CONNECTED
        # else:
        hydrogen_production_capacity_required_kgphr = []
        grid_connection_scenario = "off-grid"
        energy_to_electrolyzer_kw = inputs["electricity_in"]

        n_pem_clusters = int(ceildiv(electrolyzer_size_mw, self.config.cluster_rating_MW))

        ## run using greensteel model
        pem_param_dict = {
            "eol_eff_percent_loss": self.config.eol_eff_percent_loss,
            "uptime_hours_until_eol": self.config.uptime_hours_until_eol,
            "include_degradation_penalty": self.config.include_degradation_penalty,
            "turndown_ratio": self.config.turndown_ratio,
        }

        H2_Results, h2_ts, h2_tot, power_to_electrolyzer_kw = run_h2_PEM(
            electrical_generation_timeseries=energy_to_electrolyzer_kw,
            electrolyzer_size=electrolyzer_size_mw,
            useful_life=plant_life,
            n_pem_clusters=n_pem_clusters,
            pem_control_type=self.config.pem_control_type,
            electrolyzer_direct_cost_kw=electrolyzer_capex_kw,
            user_defined_pem_param_dictionary=pem_param_dict,
            grid_connection_scenario=grid_connection_scenario,  # if not offgrid, assumes steady h2 demand in kgphr for full year  # noqa: E501
            hydrogen_production_capacity_required_kgphr=hydrogen_production_capacity_required_kgphr,
            debug_mode=False,
            verbose=False,
        )

        # Assuming `h2_results` includes hydrogen and oxygen rates per timestep
        outputs["hydrogen_out"] = H2_Results["Hydrogen Hourly Production [kg/hr]"]
        outputs["total_hydrogen_produced"] = H2_Results["Life: Annual H2 production [kg/year]"]
        outputs["efficiency"] = H2_Results["Sim: Average Efficiency [%-HHV]"]
        outputs["time_until_replacement"] = H2_Results["Time Until Replacement [hrs]"]
        outputs["rated_h2_production_kg_pr_hr"] = H2_Results["Rated BOL: H2 Production [kg/hr]"]
