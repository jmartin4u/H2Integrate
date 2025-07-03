from h2integrate.core.h2integrate_model import H2IntegrateModel


# Create an H2I model for stimulated geologic hydrogen production
h2i_stim = H2IntegrateModel("04_geo_h2_opt.yaml")

# Run the model
h2i_stim.run()
h2i_stim.post_process()
