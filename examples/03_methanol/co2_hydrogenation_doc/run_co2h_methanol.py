from pathlib import Path

import numpy as np
from plot_co2h_methanol import plot_methanol

from h2integrate.tools.run_cases import mod_tech_config, load_tech_config_cases
from h2integrate.core.h2integrate_model import H2IntegrateModel


# Create an H2I model
model = H2IntegrateModel("03_co2h_methanol.yaml")

# Load cases
case_file = Path("test_inputs.csv")
cases = load_tech_config_cases(case_file)

# Modify and run the model for different cases
casenames = ["Case " + str(i) for i in range(1, 2)]
lcoms = []

# Run the model
for casename in casenames:
    model = mod_tech_config(model, cases[casename])
    model.run()
    model.post_process()
    lcoms.append(float(model.model.get_val("finance_subgroup_default.LCOM")[0]))
print(np.argmin(lcoms))

# Plot major in/out flows
plot_methanol(model.prob.model)
