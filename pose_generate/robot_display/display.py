from robot_display.utils.robot import Robot

class Display(Robot):
    def __init__(self, cfg: dict, vis_options=None):
        self.cfg = cfg
        if "control" not in self.cfg.keys():
            self.cfg["control"] = {"control_freq": 50}
        if "foot_names" not in self.cfg["robot"].keys():
            self.cfg["robot"]["foot_names"] = []
        if "links_to_keep" not in self.cfg["robot"].keys():
            self.cfg["robot"]["links_to_keep"] = []
        super().__init__(
            asset_file=self.cfg["robot"]["asset_path"],
            foot_names=self.cfg["robot"]["foot_names"],
            links_to_keep=self.cfg["robot"]["links_to_keep"],
            scale=self.cfg["robot"]["scale"],
            fps=self.cfg["control"]["control_freq"],
            vis_options=vis_options,
        )
        if "body_name" in self.cfg["robot"].keys():
            self.set_body_link(self.get_link_by_name(self.cfg["robot"]["body_name"]))
        if "dof_names" in self.cfg["control"].keys():
            assert len(self.cfg["control"]["dof_names"]) == self.num_dofs, "Number of dof names should match the number of dofs"
            self.set_dof_order(self.cfg["control"]["dof_names"])

    def update(self):
        self.step_vis()