import openmdao.api as om


class N2PipePerformanceModel(om.ExplicitComponent):
    """
    Pass-through nitrogen pipe with no losses.
    """

    def setup(self):
        self.add_input(
            "nitrogen_in",
            val=0.0,
            shape_by_conn=True,
            copy_shape="nitrogen_out",
            units="kg/s",
        )
        self.add_output(
            "nitrogen_out",
            val=0.0,
            shape_by_conn=True,
            copy_shape="nitrogen_in",
            units="kg/s",
        )

    def compute(self, inputs, outputs):
        outputs["nitrogen_out"] = inputs["nitrogen_in"]
