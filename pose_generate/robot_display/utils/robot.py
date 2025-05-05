import os
import time

import cv2
from scipy.ndimage import label, center_of_mass
import numpy as np
import torch
from taichi._lib import core as _ti_core
from taichi.lang import impl

import genesis as gs
from robot_display.utils.gs_math import *

def clean():
    gs.utils.misc.clean_cache_files()
    _ti_core.clean_offline_cache_files(os.path.abspath(impl.default_cfg().offline_cache_file_path))
    print("Cleaned up all genesis and taichi cache files.")

class Robot:
    def __init__(self, asset_file, foot_names, links_to_keep=[], scale=1.0, fps=60, substeps=1, vis_options=None, 
                 init_pos = [0., 0., 0.], init_quat = [1., 0., 0., 0.], init_dof_pos=None):

        gs.init(backend=gs.cpu)

        self.vis_options = vis_options
        self.visualize_interval = 0.2
        self.last_visualize_time = time.time()
        self.visualize_skeleton = getattr(vis_options, "visualize_skeleton", False)
        self.visualize_target_foot_pos = getattr(vis_options, "visualize_target_foot_pos", False)
        self.visualize_robot_frame = getattr(vis_options, "visualize_robot_frame", False)
        self.merge_fixed_links = getattr(vis_options, "merge_fixed_links", True)
        self.show_viewer = getattr(vis_options, "show_viewer", True)

        self.cfg_init_body_pos = init_pos
        self.cfg_init_body_quat = init_quat
        self.cfg_init_dof_pos = init_dof_pos

        # Create scene
        self.dt = 1 / fps
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.5, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                max_FPS=fps,
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=getattr(vis_options, "show_world_frame", False),
                shadow=getattr(vis_options, "shadow", True),
                background_color=getattr(vis_options, "background_color", (0.8, 0.8, 0.8)),
            ),
            sim_options=gs.options.SimOptions(
                gravity=(0, 0, 0),
                substeps=substeps,
            ),
            show_viewer=self.show_viewer,
        )

        # Load entity
        if asset_file.endswith(".urdf"):
            morph = gs.morphs.URDF(file=asset_file, collision=True, scale=scale, links_to_keep=links_to_keep, merge_fixed_links=self.merge_fixed_links)
        elif asset_file.endswith(".xml"):
            morph = gs.morphs.MJCF(file=asset_file, collision=True, scale=scale)
        else:
            raise ValueError(f"Unsupported file format: {asset_file}")
        self.entity = self.scene.add_entity(
            morph,
            surface=gs.surfaces.Default(
                vis_mode="visual",
            ),
        )
        self.body_link = self.links[0]
        self.body_name = self.body_link.name
        self.foot_names = []
        self.foot_links = []
        self.foot_joints = []
        is_availible = [True for link in self.links]
        for name in foot_names:
            for i, link in enumerate(self.links):
                if is_availible[i] and link.name == name:
                    is_availible[i] = False
                    self.foot_names.append(link.name)
                    self.foot_links.append(link)
                    self.foot_joints.append(link.joint)

        self.camera = self.scene.add_camera(
            pos=np.array([1, 0, 0]),
            lookat=np.array([0, 0, 0]),
            res=(1024, 1024),
            fov=getattr(vis_options, "fov", 40),
            GUI=False,
        )
        self.camera_azimuth = 45
        self.camera_elevation = 45
        self.camera_lookat = np.array([0, 0, 0])
        self.camera_distance = 1.5

        # Build scene
        self.scene.build(compile_kernels=False)
        self.last_step_time = time.time()
        self.last_debug_vis_time = time.time()

        self._init_buffers()

        center, diameter = self.get_center_diameter()
        self.set_camera_pose(lookat=center, distance=diameter * 2)

        self.step_target()

    def _init_buffers(self):

        self.link_name = [link.name for link in self.links]
        self.links_by_joint = {}
        self.joint_name = []
        self.dof_name = []
        self.dof_idx = []
        self.dof_idx_qpos = []
        self.joint_name_to_dof_order = {}
        order = 0
        base_dof_offset = 6 # Skip the base dofs
        for joint in self.joints:
            self.joint_name_to_dof_order[joint.name] = -1
            if joint.type == gs.JOINT_TYPE.FREE:
                continue
            else:
                self.joint_name.append(joint.name)
                if joint.type == gs.JOINT_TYPE.FIXED:
                    continue
                self.dof_name.append(joint.name)
                self.dof_idx.append(order + base_dof_offset)
                self.joint_name_to_dof_order[joint.name] = order
                self.dof_idx_qpos.append(order + base_dof_offset + 1)
                order += 1
        self.num_links = len(self.link_name)
        self.num_dofs = len(self.dof_name)
        self.num_joints = len(self.joint_name)

        self.link_colors = []
        for _ in range(self.num_links):
            self.link_colors.append(np.random.randint(0, 256, 3))

        self.link_adjacency_map = [[False for _ in range(self.num_links)] for _ in range(self.num_links)]
        for link in self.links:
            for idx in link.child_idxs_local:
                self.link_adjacency_map[link.idx_local][idx] = self.links[idx].joint.name
            if link.idx_local == 0:
                continue
            self.link_adjacency_map[link.idx_local][link.parent_idx_local] = link.joint.name

        if len(self.foot_links) > 1:
            self.update_skeleton()
        else:
            joint_pos = []
            for joint in self.joints:
                joint_pos.append(joint.get_pos())
            # calculate longest distance between joints and store it as self.diameter

        _, self.diameter = self.get_center_diameter()

        print("--------- Link Names ----------")
        print(self.link_name)
        print("--------- Joint Names ---------")
        print(self.dof_name)
        print("-------------------------------")

        self.init_body_pos = torch.tensor(self.cfg_init_body_pos, dtype=torch.float32)  
        self.init_body_quat = torch.tensor(self.cfg_init_body_quat, dtype=torch.float32)
        self.init_dof_pos = torch.zeros(self.num_dofs, dtype=torch.float32)
        if self.cfg_init_dof_pos is not None:
            for dof_name, pos in self.cfg_init_dof_pos.items():
                idx = self.dof_name.index(dof_name)
                self.init_dof_pos[idx] = pos

        self.target_body_pos = self.init_body_pos.clone()
        self.target_body_quat = self.init_body_quat.clone()
        self.target_dof_pos = self.init_dof_pos.clone()

        self.target_foot_pos = []
        self.target_foot_quat = []

    def update_skeleton(self):

        # for geom in self.body_link.geoms:
        #     vertices = geom.get_verts()
        #     print(vertices)
        #     print(vertices.shape)
        #     exit()

        self.leg = []
        self.body = []
        self.leg_joint_idx = []

        body_joint_name = []
        for idx in self.body_link.child_idxs_local:
            body_joint_name.append(self.links[idx].joint.name)
        if self.body_link.idx_local != 0: # If the base link is not the root link
            body_joint_name.append(self.body_link.joint.name)

        # vertices to visualize body
        for joint_name1 in body_joint_name:
            for joint_name2 in body_joint_name:
                if joint_name1 < joint_name2:
                    self.body.append((joint_name1, joint_name2))

        def dfs(curr, target, visited, path):
            visited[curr] = True
            path.append(curr)
            if curr == target:
                return True
            for i in range(self.num_links):
                if self.link_adjacency_map[curr][i] and not visited[i]:
                    if dfs(i, target, visited, path):
                        return True
            path.pop()
            return False

        paths = []
        for link in self.foot_links:
            foot_idx = link.idx_local
            body_idx = self.body_link.idx_local
            if foot_idx == body_idx: continue
            path = []
            visited = [False for _ in range(self.num_links)]
            dfs(foot_idx, body_idx, visited, path)
            paths.append(path)
        
        # distill the path
        # remove the links that are used by other paths
        links_used_counter = [0 for _ in range(self.num_links)]
        for path in paths:
            for idx in path:
                links_used_counter[idx] += 1
        distilled_paths = []
        for path in paths:
            joint_path = []
            leg_joint_idx = []
            for i in range(len(path) - 1):
                joint_name = self.link_adjacency_map[path[i]][path[i + 1]]
                if joint_name in self.dof_name:
                    leg_joint_idx.append(self.dof_name.index(joint_name))
            self.leg_joint_idx.append(leg_joint_idx)
            for i in range(len(path) - 2):
                if links_used_counter[path[i]] == 1:
                    joint_path.append(self.link_adjacency_map[path[i]][path[i + 1]])
            joint_path.append(self.link_adjacency_map[path[-2]][path[-1]])
            distilled_paths.append(joint_path)
        paths = distilled_paths.copy()
        distilled_paths = []

        joint_pos = {}
        for joint in self.joints:
            joint_pos[joint.name] = joint.get_pos()

        max_dist = 0
        for path in paths:
            dist_list = [0,]
            dist = 0
            for i in range(len(path) - 1):
                dist += torch.norm(joint_pos[path[i]] - joint_pos[path[i + 1]]).item()
                dist_list.append(dist)
            dist_list = [d / dist for d in dist_list]
            idx = np.argmin(np.abs(np.array(dist_list) - 0.5))
            distilled_paths.append([path[0], path[idx], path[-1]])
            if dist > max_dist:
                max_dist = dist
        paths = distilled_paths.copy()
        distilled_paths = []

        for path in paths:
            for i in range(len(path) - 1):
                if path[i] < path[i + 1]:
                    if (path[i], path[i + 1]) not in self.leg:
                        self.leg.append((path[i], path[i + 1]))
                else:
                    if (path[i + 1], path[i]) not in self.leg:
                        self.leg.append((path[i + 1], path[i]))

        _, self.diameter = self.get_center_diameter()

    def reset(self):
        self.target_body_pos = self.init_body_pos.clone()
        self.target_body_quat = self.init_body_quat.clone()
        self.target_dof_pos = self.init_dof_pos.clone()
        self.step_target()

    def step(self):
        if self.dt - (time.time() - self.last_step_time) > 0.01:
            time.sleep(self.dt - (time.time() - self.last_step_time))
        self.last_step_time = time.time()
        self.entity.control_dofs_position(self.target_dof_pos, self.dof_idx)
        self.scene.step()

    def step_vis(self):
        if self.dt - (time.time() - self.last_step_time) > 0.01:
            time.sleep(self.dt - (time.time() - self.last_step_time))
        self.last_step_time = time.time()
        self.step_target()
        self.scene.visualizer.update(force=True)

        if (time.time() - self.last_debug_vis_time) > 0.1:
            self.last_debug_vis_time = time.time()
            self.scene.clear_debug_objects()
            if self.visualize_skeleton:
                self._visualize_skeleton()
            if self.visualize_target_foot_pos:
                self._visualize_target_foot_pos()
            if self.visualize_robot_frame:
                self._visualize_robot_frame()

    def step_target(self):
        # Set the joint positions
        self.target_dof_pos = torch.max(torch.min(self.target_dof_pos, self.dof_limit[1]), self.dof_limit[0])
        self.entity.set_dofs_position(self.target_dof_pos, self.dof_idx, zero_velocity=True)

        # Set base rotation
        R = gs_quat_mul(self.target_body_quat, gs_quat_conjugate(self.body_quat))
        self.entity.set_quat(gs_quat_mul(R, self.entity.get_quat()))

        # Set base position
        delta_pos = self.target_body_pos - self.body_pos
        self.entity.set_pos(delta_pos + self.entity.get_pos())

    def clear_debug_objects(self):
        self.scene.clear_debug_objects()

    def _visualize_target_foot_pos(self):

        self.scene.draw_debug_spheres(poss=self.target_foot_pos, radius=self.diameter / 20, color=(1, 0, 0, 0.5))
        for link in self.foot_links:
            self.scene.draw_debug_sphere(pos=link.get_pos(), radius=self.diameter / 20, color=(0, 1, 0, 0.5))

    def visualize_frame(self, center):

        length_frac = 0.3
        width_frac = 100

        vector = torch.tensor([1.0, 0.0, 0.0]) * self.diameter * length_frac
        self.scene.draw_debug_arrow(center, vector, radius=self.diameter / width_frac, color=(1.0, 0.0, 0.0, 1.0))
        vector = torch.tensor([0.0, 1.0, 0.0]) * self.diameter * length_frac
        self.scene.draw_debug_arrow(center, vector, radius=self.diameter / width_frac, color=(0.0, 1.0, 0.0, 1.0))
        vector = torch.tensor([0.0, 0.0, 1.0]) * self.diameter * length_frac
        self.scene.draw_debug_arrow(center, vector, radius=self.diameter / width_frac, color=(0.0, 0.0, 1.0, 1.0))

        if self.visualize_skeleton:
            center = center - 2 * self.diameter
            vector = torch.tensor([1.0, 0.0, 0.0]) * self.diameter * length_frac
            self.scene.draw_debug_arrow(center, vector, radius=self.diameter / width_frac, color=(1.0, 0.0, 0.0, 1.0))
            vector = torch.tensor([0.0, 1.0, 0.0]) * self.diameter * length_frac
            self.scene.draw_debug_arrow(center, vector, radius=self.diameter / width_frac, color=(0.0, 1.0, 0.0, 1.0))
            vector = torch.tensor([0.0, 0.0, 1.0]) * self.diameter * length_frac
            self.scene.draw_debug_arrow(center, vector, radius=self.diameter / width_frac, color=(0.0, 0.0, 1.0, 1.0))

    def _visualize_robot_frame(self):

        length_frac = 0.6
        width_frac = 70

        center = self.body_pos
        vector = torch.tensor([1.0, 0.0, 0.0]) * self.diameter * length_frac
        self.scene.draw_debug_arrow(center, vector, radius=self.diameter / width_frac, color=(1.0, 0.0, 0.0, 1.0))
        vector = torch.tensor([0.0, 1.0, 0.0]) * self.diameter * length_frac
        self.scene.draw_debug_arrow(center, vector, radius=self.diameter / width_frac, color=(0.0, 1.0, 0.0, 1.0))
        vector = torch.tensor([0.0, 0.0, 1.0]) * self.diameter * length_frac
        self.scene.draw_debug_arrow(center, vector, radius=self.diameter / width_frac, color=(0.0, 0.0, 1.0, 1.0))

        if self.visualize_skeleton:
            center = self.body_pos - 2 * self.diameter
            vector = torch.tensor([1.0, 0.0, 0.0]) * self.diameter * length_frac
            self.scene.draw_debug_arrow(center, vector, radius=self.diameter / width_frac, color=(1.0, 0.0, 0.0, 1.0))
            vector = torch.tensor([0.0, 1.0, 0.0]) * self.diameter * length_frac
            self.scene.draw_debug_arrow(center, vector, radius=self.diameter / width_frac, color=(0.0, 1.0, 0.0, 1.0))
            vector = torch.tensor([0.0, 0.0, 1.0]) * self.diameter * length_frac
            self.scene.draw_debug_arrow(center, vector, radius=self.diameter / width_frac, color=(0.0, 0.0, 1.0, 1.0))

    def visualize_link_frame(self, center, x=True, y=True, z=True):

        length_frac = 0.8
        width_frac = 70

        if x:
            vector = torch.tensor([1.0, 0.0, 0.0]) * self.diameter * length_frac
            self.scene.draw_debug_arrow(center, vector, radius=self.diameter / width_frac, color=(1.0, 0.0, 0.0, 1.0))
        if y:
            vector = torch.tensor([0.0, 1.0, 0.0]) * self.diameter * length_frac
            self.scene.draw_debug_arrow(center, vector, radius=self.diameter / width_frac, color=(0.0, 1.0, 0.0, 1.0))
        if z:
            vector = torch.tensor([0.0, 0.0, 1.0]) * self.diameter * length_frac
            self.scene.draw_debug_arrow(center, vector, radius=self.diameter / width_frac, color=(0.0, 0.0, 1.0, 1.0))

    def _visualize_skeleton(self):

        if time.time() - self.last_visualize_time < self.visualize_interval:
            return
        else:
            self.last_visualize_time = time.time()

        joint_pos = {}
        joint_vis = {}
        for joint in self.joints:
            joint_vis[joint.name] = False
            joint_pos[joint.name] = joint.get_pos()
            joint_pos[joint.name] -= 2 * self.diameter

        body_color = (1.0, 0.5, 0., 1)
        leg_color = (0, 0.8, 0.8, 1)

        thickness = self.diameter / 50 

        lines = []
        for name1 in self.joint_name:
            pos1 = joint_pos[name1]
            for name2 in self.joint_name:
                if (name1, name2) in self.body:
                    color = body_color
                    joint_vis[name1] = "body"
                    joint_vis[name2] = "body"
                elif (name1, name2) in self.leg:
                    color = leg_color
                    joint_vis[name1] = "leg"
                    joint_vis[name2] = "leg"
                else:
                    continue
                pos2 = joint_pos[name2]
                lines.append((pos1, pos2, color))

        for line in lines:
            pos1, pos2, color = line
            self.scene.draw_debug_line(pos1, pos2, radius=thickness, color=color)

        for name in self.joint_name:
            if joint_vis[name] == "body":
                color = body_color
            elif joint_vis[name] == "leg":
                color = leg_color
            else:
                continue
            pos = joint_pos[name]
            self.scene.draw_debug_sphere(pos, radius=thickness, color=color)

    def get_joint(self, joint_name=None):
        return self.entity.get_joint(joint_name)

    def get_dofs_between_links(self, link_id1, link_id2):
        path = []
        visited = [False for _ in range(self.num_links)]
        def dfs(curr, target):
            visited[curr] = True
            if curr == target:
                return True
            for i in range(self.num_links):
                if self.link_adjacency_map[curr][i] and not visited[i]:
                    if dfs(i, target):
                        path.append(self.link_adjacency_map[curr][i])
                        return True
            return False
        dfs(link_id1, link_id2)
        dofs = []
        for joint_name in path:
            if self.joint_name_to_dof_order[joint_name] != -1:
                dofs.append(self.joint_name_to_dof_order[joint_name])
        return dofs

    def get_link_by_name(self, link_name=None):
        return self.entity.get_link(link_name)

    def get_qpos(self):
        return self.entity.get_qpos()

    def set_qpos(self, qpos):
        return self.entity.set_qpos(qpos)

    def get_link_by_id(self, link_id=0):
        return self.links[link_id]

    def set_body_link(self, link):
        self.body_link = link
        self.body_name = link.name
        if len(self.foot_links) > 1:
            self.update_skeleton()
        self.step_target()

    def set_body_link_by_name(self, body_name):
        self.body_name = body_name
        self.body_link = self.get_link_by_name(body_name)
        self.step_target()

    def set_body_link_by_id(self, body_id):
        self.body_link = self.links[body_id]
        self.body_name = self.body_link.name
        self.step_target()

    def set_init_state(self, body_pos, body_qaut, dof_pos):
        self.init_body_pos = torch.tensor(body_pos)
        self.init_body_pos[:2] = 0.0
        self.init_body_quat = torch.tensor(body_qaut)
        self.init_dof_pos = torch.tensor(dof_pos)
        self.target_body_pos = self.init_body_pos.copy()
        self.target_body_quat = self.init_body_quat.copy()
        self.target_dof_pos = self.init_dof_pos.copy()
        self.step_target()

    def set_dof_order(self, dof_names):
        dof_idx = []
        dof_idx_qpos = []
        for order, name in enumerate(dof_names):
            for idx, joint in enumerate(self.dof_name):
                if name == joint:
                    dof_idx.append(self.dof_idx[idx])
                    dof_idx_qpos.append(self.dof_idx_qpos[idx])
                    self.joint_name_to_dof_order[joint] = order
                    break
        assert len(dof_idx) == len(dof_names), "Some dof names are not found"
        self.dof_idx = dof_idx
        self.dof_idx_qpos = dof_idx_qpos
        self.dof_name = dof_names
        self.init_dof_pos = torch.zeros(len(dof_idx), dtype=torch.float32)
        self.target_dof_pos = self.init_dof_pos.clone()

    def set_body_pos(self, pos):
        self.target_body_pos = pos

    def set_body_height(self, height):
        self.target_body_pos[2] = height

    def set_body_quat(self, quat):
        quat = torch.tensor(quat)
        self.target_body_quat = normalize(quat)

    def set_body_pose(self, roll, pitch, yaw):
        roll, pitch, yaw = roll / 180 * np.pi, pitch / 180 * np.pi, yaw / 180 * np.pi
        xyz = torch.tensor([roll, pitch, yaw])
        R = gs_euler2quat(xyz)
        # Compute the rotation quaternion 
        self.target_body_quat = gs_quat_mul(R, self.init_body_quat)

    def set_dofs_position(self, positions, dof_idx_local=None):
        positions = torch.tensor(positions, dtype=torch.float32)
        if dof_idx_local is None:
            self.target_dof_pos = positions
        else:
            self.target_dof_pos[dof_idx_local] = positions

    def set_foot_links_pose(self, poss, quats):
        self.target_foot_pos = poss
        self.target_foot_quat = quats
        for i in range(len(self.foot_links)):
            links = [self.body_link, self.foot_links[i]]
            poss = [self.target_body_pos, self.target_foot_pos[i]]
            quats = [self.target_body_quat, self.target_foot_quat[i]]
            qpos, error = self.entity.inverse_kinematics_multilink(
                links=links,
                poss=poss,
                quats=quats,
                return_error=True,
            )
            self.set_dofs_position(qpos[self.dof_idx_qpos][self.leg_joint_idx[i]], self.leg_joint_idx[i])

    def set_link_pose(self, link_id, pos, quat=None):
        links = [self.body_link, self.links[link_id]]
        poss = [self.target_body_pos, pos]
        quats = [self.target_body_quat, quat]
        # quats = [None, quat]
        qpos, error = self.entity.inverse_kinematics_multilink(
            links=links,
            poss=poss,
            quats=quats,
            return_error=True,
            max_solver_iters=200,
        )
        if error.abs().max() > 0.01:
            return False
        else:
            dof_pos = qpos[self.dof_idx_qpos]
            dof_idx = self.get_dofs_between_links(self.body_link.idx_local, link_id)
            self.set_dofs_position(dof_pos[dof_idx], dof_idx)
            return True

    def set_link_pos(self, link_id, pos):
        links = [self.body_link, self.links[link_id]]
        poss = [self.target_body_pos, pos]
        quats = [self.target_body_quat, None]
        # quats = [None, quat]
        qpos, error = self.entity.inverse_kinematics_multilink(
            links=links,
            poss=poss,
            quats=quats,
            return_error=True,
            max_solver_iters=200,
        )
        dof_pos = qpos[self.dof_idx_qpos]
        dof_idx = self.get_dofs_between_links(self.body_link.idx_local, link_id)
        self.set_dofs_position(dof_pos[dof_idx], dof_idx)

    def try_set_link_pose(self, link_id, pos, quat=None):
        links = [self.body_link, self.links[link_id]]
        poss = [self.target_body_pos, pos]
        quats = [self.target_body_quat, quat]
        # quats = [None, quat]
        qpos, error = self.entity.inverse_kinematics_multilink(
            links=links,
            poss=poss,
            quats=quats,
            return_error=True,
            max_solver_iters=200,
        )
        if error.abs().max() > 0.01:
            return False
        else:
            return True

    def set_dofs_armature(self, armature):
        if not isinstance(armature, dict):
            dofs_armature = [armature] * self.num_dofs
        else:
            dofs_armature = []
            for name in self.dof_name:
                dofs_armature.append(armature[name])
        self.entity.set_dofs_armature(dofs_armature, self.dof_idx)

    def set_dofs_damping(self, damping):
        if not isinstance(damping, dict):
            dofs_damping = [damping] * self.num_dofs
        else:
            dofs_damping = []
            for name in self.dof_name:
                dofs_damping.append(damping[name])
        self.entity.set_dofs_damping(dofs_damping, self.dof_idx)

    def set_dofs_kp(self, kp):
        if not isinstance(kp, dict):
            dofs_kp = [kp] * self.num_dofs
        else:
            dofs_kp = []
            for name in self.dof_name:
                dofs_kp.append(kp[name])
        self.entity.set_dofs_kp(dofs_kp, self.dof_idx)

    def set_dofs_kd(self, kd):
        if not isinstance(kd, dict):
            dofs_kd = [kd] * self.num_dofs
        else:
            dofs_kd = []
            for name in self.dof_name:
                dofs_kd.append(kd[name])
        self.entity.set_dofs_kv(dofs_kd, self.dof_idx)

    def set_camera_pose(self, azimuth=None, elevation=None, distance=None, lookat=None):
        if azimuth is not None:
            self.camera_azimuth = azimuth
            if self.camera_azimuth < 0:
                self.camera_azimuth += 360
            if self.camera_azimuth > 360:
                self.camera_azimuth -= 360
        if elevation is not None:
            self.camera_elevation = elevation
            if self.camera_elevation < -90:
                self.camera_elevation = -90
            if self.camera_elevation > 90:
                self.camera_elevation = 90
        if distance is not None:
            self.camera_distance = distance
        if lookat is not None:
            self.camera_lookat = torch.tensor(lookat)
        pos = self.camera_lookat + self.camera_distance * torch.tensor([
            np.cos(self.camera_azimuth / 180 * np.pi) * np.cos(self.camera_elevation / 180 * np.pi),
            np.sin(self.camera_azimuth / 180 * np.pi) * np.cos(self.camera_elevation / 180 * np.pi),
            np.sin(self.camera_elevation / 180 * np.pi),
        ])
        self.camera.set_pose(
            pos=pos,
            lookat=self.camera_lookat,
        )
        # if self.show_viewer:
        #     self.scene.viewer.set_camera_pose(
        #         pos=pos,
        #         lookat=self.camera_lookat,
        #     )

    def render(self, link_ids=None):
        rgb_arr, depth_arr, seg_arr, normal_arr = self.camera.render(rgb=True, segmentation=True)
    
        alpha = 0.5

        # Create a color map for each segment ID
        unique_labels = np.unique(seg_arr)
        if link_ids is not None:
            unique_labels = unique_labels[np.isin(unique_labels, link_ids)]
        color_map = {}
        for label_id in unique_labels:
            if label_id == -1:
                color_map[label_id] = np.array((0, 0, 0), dtype=np.float32)  # Black for background
            else:
                color_map[label_id] = self.link_colors[label_id]

        # Create an output image with highlighted segments
        highlighted_image = np.zeros_like(rgb_arr, dtype=np.float32)
        for label_id in unique_labels:
            if label_id == -1:
                mask = (seg_arr == label_id)
                highlighted_image[mask] = rgb_arr[mask]
                continue
            mask = (seg_arr == label_id)
            highlight_color = color_map[label_id]
            # Blend the highlight color with the original image
            highlighted_image[mask] = alpha * highlight_color + (1 - alpha) * rgb_arr[mask]

        # Convert the highlighted image back to uint8
        highlighted_image = np.clip(highlighted_image, 0, 255).astype(np.uint8)

        # Draw borders between segments
        for label_id in unique_labels:
            if label_id == -1:
                continue  # Skip background
            mask = (seg_arr == label_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            border_color = 0.5 * (color_map[label_id])
            cv2.drawContours(highlighted_image, contours, -1, border_color, 2)  # Draw borders

        labelled_image = highlighted_image.copy()

        # Place segment IDs on the image
        for label_id in unique_labels:
            if label_id == -1:
                continue  # Skip background
            mask = (seg_arr == label_id)
            if np.any(mask):
                # Calculate the centroid of the segment
                centroid = center_of_mass(mask)
                y, x = map(int, centroid)

                # Ensure the centroid lies within the segment
                if not mask[y, x]:
                    # Find the closest pixel in the segment to the centroid
                    y, x = np.argwhere(mask)[np.linalg.norm(np.argwhere(mask) - centroid, axis=1).argmin()]

                # Add a black square under the number for better visibility
                text = str(label_id)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 2
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_w, text_h = text_size

                # Coordinates for the black square
                square_top_left = (x - text_w // 2 - 2, y - text_h // 2 - 2)
                square_bottom_right = (x + text_w // 2 + 2, y + text_h // 2 + 2)

                # Draw the black square
                cv2.rectangle(labelled_image, square_top_left, square_bottom_right, (0, 0, 0), -1)  # -1 fills the rectangle

                # Put the segment ID as text on the image
                cv2.putText(labelled_image, text, (x - text_w // 2, y + text_h // 2), font, font_scale, (255, 255, 255), thickness)

        return rgb_arr, seg_arr, labelled_image, unique_labels[unique_labels != -1]

    def get_center(self):
        AABB = self.entity.get_AABB()
        return (AABB[1] + AABB[0]) / 2

    def get_diameter(self):
        AABB = self.entity.get_AABB()
        return torch.norm(AABB[1] - AABB[0])

    def get_center_diameter(self):
        return self.get_center(), self.get_diameter()

    @property
    def links(self):
        return self.entity.links

    @property
    def links_pos(self):
        return self.entity.get_links_pos()

    @property
    def links_quat(self):
        return self.entity.get_links_quat()

    @property
    def body_pos(self):
        return self.links_pos[self.body_link.idx_local]

    @property
    def body_quat(self):
        return self.links_quat[self.body_link.idx_local]

    @property
    def body_pose(self):
        return gs_quat2euler(self.body_quat)

    @property
    def base_pos(self):
        return self.entity.get_pos()

    @property
    def base_quat(self):
        return self.entity.get_quat()

    @property
    def joints(self):
        return self.entity.joints

    @property
    def dof_pos(self):
        return self.entity.get_dofs_position(dofs_idx_local=self.dof_idx)

    @ property
    def dof_limit(self):
        return self.entity.get_dofs_limit(self.dof_idx)

    @property
    def foot_pos(self):
        return self.links_pos[[link.idx_local for link in self.foot_links],]

    @property
    def foot_quat(self):
        return self.links_quat[[link.idx_local for link in self.foot_links],]
    
    @property
    def mass(self):
        return self.entity.get_mass()
