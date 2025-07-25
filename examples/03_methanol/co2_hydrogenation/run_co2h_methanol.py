from h2integrate.core.h2integrate_model import H2IntegrateModel


# Create an H2I model
h2i = H2IntegrateModel("03_co2h_methanol.yaml")

h2i.prob.setup()
demand_dict = {}

# Get feedstock demands from methanol plant
meoh_perf = h2i.plant.methanol.co2h_methanol_plant_performance
meoh_demand = meoh_perf.calc_production_demand()
unconnected_feedstocks = ["meoh_syn_cat", "ng", "co2", "electricity"]
demand_dict["methanol"] = meoh_perf.calc_feedstock_demand(meoh_demand, unconnected_feedstocks)

# Get feedstock demands from electrolyzer (currently just electricity)
elyzer_perf = h2i.plant.electrolyzer.eco_pem_electrolyzer_performance
hydrogen_demand = demand_dict["methanol"]["hydrogen"]
demand_dict["electrolyzer"] = elyzer_perf.calc_feedstock_demand(hydrogen_demand)

# Get feedtsock demands from hopp (currently none)
hopp_perf = h2i.plant.hopp.hopp
elec_demand = demand_dict["electrolyzer"]["electricity"]
demand_dict["hopp"] = {}

# Set plant capacities based on demand
plant_capacity = meoh_perf.size_from_production_demand(meoh_demand)
rating = elyzer_perf.size_from_production_demand(hydrogen_demand)
num_turbines, pv_capacity_kw = hopp_perf.size_from_production_demand(elec_demand)


# Run the model
h2i.run()

h2i.post_process()
