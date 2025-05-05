import numpy as np
import torch

from robot_display.utils.gui import start_gui
from robot_display.utils.robot import Robot

class GUIDisplay:
    def __init__(self, cfg: dict, body_pos=True, body_pose=True, dofs_pos=True, foot_pos=False, pd_control=False, save_callable=None, vis_options=False, **kwargs):
        assert body_pos or body_pose or dofs_pos or foot_pos, "At least one of the interaction modes should be enabled"
        assert not (dofs_pos and foot_pos), "Dofs pos and foot position cannot be enabled at the same time"
        self.cfg = cfg
        self.pd_control = pd_control
        self.control_body_height = body_pos
        self.control_body_pose = body_pose
        self.control_dofs_pos = dofs_pos
        self.control_foot_pos = foot_pos
        self.save_callable = save_callable
        self.vis_options = vis_options
        self.setup_robot()
        self.setup_gui()

    def setup_robot(self):
        if "control" not in self.cfg.keys():
            self.cfg["control"] = {"control_freq": 50}
        if "links_to_keep" not in self.cfg["robot"].keys():
            self.cfg["robot"]["links_to_keep"] = []
            self.cfg["robot"]["body_init_quat"]
        self.robot = Robot(
            asset_file=self.cfg["robot"]["asset_path"],
            foot_names=None,
            links_to_keep=self.cfg["robot"]["links_to_keep"],
            scale=self.cfg["robot"]["scale"],
            fps=self.cfg["control"]["control_freq"],
            vis_options=self.vis_options,
            init_pos=self.cfg["robot"].get("body_init_pos", [0, 0, 0]),
            init_quat=self.cfg["robot"].get("body_init_quat", [1, 0, 0, 0]),
            init_dof_pos=self.cfg["robot"].get("default_dof_pos", None),
        )
        if "body_name" in self.cfg["robot"].keys():
            self.robot.set_body_link(self.robot.get_link_by_name(self.cfg["robot"]["body_name"]))
        # if "dof_names" in self.cfg["control"].keys():
        #     assert len(self.cfg["control"]["dof_names"]) == self.robot.num_dofs, "Number of dof names should match the number of dofs"
        #     self.robot.set_dof_order(self.cfg["control"]["dof_names"])
        if self.pd_control:
            self.robot.set_dofs_kp(self.cfg["control"]["kp"])
            self.robot.set_dofs_kd(self.cfg["control"]["kd"])
            if "armature" in self.cfg["control"].keys():
                self.robot.set_dofs_armature(self.cfg["control"]["armature"])
            if "damping" in self.cfg["control"].keys():
                self.robot.set_dofs_damping(self.cfg["control"]["damping"])

    def setup_gui(self):
        self.labels = []
        self.limits = {}
        self.values = []
        idx = 0
        if self.control_body_height:
            self.labels.append("Body Height")
            self.limits["Body Height"] = [0.0, self.robot.diameter * 2]
            self.values.append(self.robot.body_pos[2].item())
            self.value_body_height_idx = idx
            idx += 1
        if self.control_body_pose:
            self.labels.extend(["Body Roll", "Body Pitch" , "Body Yaw"])
            self.limits["Body Roll"] = [-180, 180]
            self.limits["Body Pitch"] = [-180, 180]
            self.limits["Body Yaw"] = [-180, 180]
            self.values.extend(self.robot.body_pose.numpy().tolist())
            self.value_body_pose_idx = idx
            idx += 3
        if self.control_dofs_pos:
            self.labels.extend(self.robot.dof_name)
            dof_limits = self.robot.dof_limit
            dof_limits = (torch.clip(dof_limits[0], -np.pi, np.pi), torch.clip(dof_limits[1], -np.pi, np.pi))
            for i in range(len(self.robot.dof_name)):
                self.limits[self.robot.dof_name[i]] = [dof_limits[0][i].item(), dof_limits[1][i].item()]
                soft_limits = [
                    dof_limits[1][i].item() - 0.9 * (dof_limits[1][i].item() - dof_limits[0][i].item()),
                    dof_limits[0][i].item() + 0.9 * (dof_limits[1][i].item() - dof_limits[0][i].item()),
                ]
                if "pose" in self.cfg.keys() and "default_joint_angles" in self.cfg["pose"].keys():
                    value = self.cfg["pose"]["default_joint_angles"][self.robot.dof_name[i]]
                else:
                    value = 0
                self.values.append(max(soft_limits[0], min(value, soft_limits[1])))
            self.value_dofs_pos_idx_start = idx
            self.value_dofs_pos_idx_end = idx + self.robot.num_dofs
            idx += self.robot.num_dofs
        if self.control_foot_pos:
            for foot in self.robot.foot_links:
                name = foot.name
                self.labels.extend([f"{name} x", f"{name} y", f"{name} z"])
                self.limits[f"{name} x"] = [-self.robot.diameter, self.robot.diameter]
                self.limits[f"{name} y"] = [-self.robot.diameter, self.robot.diameter]
                self.limits[f"{name} z"] = [-self.robot.diameter, self.robot.diameter]
                # self.limits[f"{name} x"] = [-self.robot.diameter * 2, self.robot.diameter * 2]
                # self.limits[f"{name} y"] = [-self.robot.diameter * 2, self.robot.diameter * 2]
                # self.limits[f"{name} z"] = [-self.robot.diameter * 2, self.robot.diameter * 2]
                if "foot_pos" not in self.cfg["robot"].keys():
                    self.values.extend((foot.get_pos() - self.robot.base_pos).numpy().tolist())
                else:
                    self.values.extend(self.cfg["robot"]["foot_pos"][name])
            self.value_foot_pos_idx_start = idx
            self.value_foot_pos_idx_end = idx + 3 * len(self.robot.foot_links)
            idx += self.robot.num_dofs
        cfg = {
            "label": self.labels,
            "range": self.limits,
        }
        self.gui = start_gui(
            cfg=cfg,
            values=self.values,
            save_callback=self.save_callback,
            reset_callback=self.reset_callback,
        )

    def save_callback(self):
        self.save_callable(self)

    def reset_callback(self):
        self.robot.reset()

    def update(self):
        self.robot.target_body_pos[:2] = 0
        if self.control_body_height:
            self.robot.set_body_height(self.values[self.value_body_height_idx])
        if self.control_body_pose:
            self.robot.set_body_pose(*self.values[self.value_body_pose_idx:self.value_body_pose_idx+3])
        if self.control_dofs_pos:
            self.robot.set_dofs_position(self.values[self.value_dofs_pos_idx_start:self.value_dofs_pos_idx_end])
        if self.control_foot_pos:
            poss = []
            quats = []
            body_pos = self.robot.target_body_pos
            for i in range(len(self.robot.foot_links)):
                pos = self.values[self.value_foot_pos_idx_start + 3 * i:self.value_foot_pos_idx_start + 3 * (i + 1)]
                poss.append(body_pos + torch.tensor(pos))
                # quats.append(link.get_quat())
                quats.append(None)
            self.robot.set_foot_links_pose(poss, quats)
        # print(f'pos {self.robot.base_pos}; quat {self.robot.base_quat}')
        # for link in self.robot.foot_links:
        #     print(link.name, link.get_pos() - self.robot.base_pos)
        if self.pd_control:
            self.robot.step()
        else:
            self.robot.step_vis()

    def run(self):
        self.reset_callback()
        while True:
            self.update()

    def render(self):
        return self.robot.render()