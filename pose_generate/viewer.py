import os
import yaml
import argparse
import threading
import multiprocessing
import time

import numpy as np
import math
import torch
import cv2
from scipy.ndimage import label, center_of_mass
import matplotlib.pyplot as plt
import pickle

from robot_display.display import Display

class VisOptions:
    visualize_skeleton = False
    visualize_target_foot_pos = False
    merge_fixed_links = False
    show_world_frame = False
    shadow = False
    background_color = (0.8, 0.8, 0.8)
    show_viewer = False

def rotate_quat_from_rpy(quat, roll, pitch, yaw):
    """
    Rotate a quaternion by given roll, pitch, and yaw angles (in degrees).
    
    Args:
        quat: Input quaternion as a tensor in shape (4,) with order (w, x, y, z)
        roll: Rotation around x-axis in degrees
        pitch: Rotation around y-axis in degrees
        yaw: Rotation around z-axis in degrees

    Returns:
        Rotated quaternion as a tensor in shape (4,)
    """
    # Convert angles from degrees to radians
    roll_rad = math.radians(roll)
    pitch_rad = -math.radians(pitch)
    yaw_rad = math.radians(yaw)
    
    # Compute half angles
    half_roll = roll_rad / 2.0
    half_pitch = pitch_rad / 2.0
    half_yaw = yaw_rad / 2.0
    
    # Create rotation quaternions for each axis
    # Roll (x-axis rotation)
    cr, sr = math.cos(half_roll), math.sin(half_roll)
    q_roll = torch.tensor([cr, sr, 0.0, 0.0], dtype=quat.dtype, device=quat.device)
    
    # Pitch (y-axis rotation)
    cp, sp = math.cos(half_pitch), math.sin(half_pitch)
    q_pitch = torch.tensor([cp, 0.0, sp, 0.0], dtype=quat.dtype, device=quat.device)
    
    # Yaw (z-axis rotation)
    cy, sy = math.cos(half_yaw), math.sin(half_yaw)
    q_yaw = torch.tensor([cy, 0.0, 0.0, sy], dtype=quat.dtype, device=quat.device)
    
    # Combine the rotations (order: yaw -> pitch -> roll)
    q_rpy = quaternion_multiply(q_roll, quaternion_multiply(q_pitch, q_yaw))
    
    # Rotate the input quaternion by the combined rotation
    rotated_quat = quaternion_multiply(q_rpy, quat)
    
    return rotated_quat

def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.
    
    Args:
        q1: First quaternion (w, x, y, z)
        q2: Second quaternion (w, x, y, z)
        
    Returns:
        Product of q1 and q2
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.tensor([w, x, y, z], dtype=q1.dtype, device=q1.device)

class Viewer:
    def __init__(self, cfg_path, vis_options=VisOptions()):
        self.cfg = yaml.safe_load(open(cfg_path))
        self.vis_options = vis_options
        self.display = Display(
            cfg=self.cfg,
            vis_options=self.vis_options,
        )
        
        self.init_camera_pose = self.get_camera_pose()

    def get_body_pos(self):
        """
        Get the position of the body link.
        Return:
            body_pos (torch.tensor): The position of the body link
        """
        body_pos = self.display.body_pos
        return body_pos

    def get_camera_pose(self):
        """
        Get the current camera pose.
        Return:
            camera_pose (dict): The current camera pose
            camera_pose["azimuth"] (degree): The azimuth angle of the camera
            camera_pose["elevation"] (degree): The elevation angle of the camera
            camera_pose["lookat"] (torch.tensor): The lookat point of the camera
            camera_pose["distance"] (float): The distance of the camera from the lookat point
        The camera's position is defined by the azimuth, elevation, lookat point, and distance by
        camera_pos = lookat + distance * torch.tensor([
            np.cos(azimuth / 180 * np.pi) * np.cos(elevation / 180 * np.pi),
            np.sin(azimuth / 180 * np.pi) * np.cos(elevation / 180 * np.pi),
            np.sin(elevation / 180 * np.pi),
        ])
        """
        camera_pose = {
            "azimuth": self.display.camera_azimuth,
            "elevation": self.display.camera_elevation,
            "distance": self.display.camera_distance,
            "lookat": self.display.camera_lookat,
        }
        return camera_pose

    def set_camera_pose(self, camera_pose):
        """
        Set camera pose.
        Args:
            camera_pose (dict): The target camera pose
            camera_pose["azimuth"] (degree): The azimuth angle of the camera
            camera_pose["elevation"] (degree): The elevation angle of the camera
            camera_pose["lookat"] (torch.tensor): The lookat point of the camera
            camera_pose["distance"] (float): The distance of the camera from the lookat point
        The camera's position is defined by the azimuth, elevation, lookat point, and distance by
        camera_pos = lookat + distance * torch.tensor([
            np.cos(azimuth / 180 * np.pi) * np.cos(elevation / 180 * np.pi),
            np.sin(azimuth / 180 * np.pi) * np.cos(elevation / 180 * np.pi),
            np.sin(elevation / 180 * np.pi),
        ])
        """
        self.display.set_camera_pose(
            azimuth=camera_pose["azimuth"],
            elevation=camera_pose["elevation"],
            distance=camera_pose["distance"],
            lookat=camera_pose["lookat"],
        )
    
    def reset_camera_pose(self):
        """
        Reset camera pose to the initial pose.
        """
        self.set_camera_pose(self.init_camera_pose)
        self.display.update()

    def pack_camera_transform(self):
        camera_transform = {}
        camera_transform["extrinsics"] = self.display.camera.extrinsics
        camera_transform["intrinsics"] = self.display.camera.intrinsics
        return camera_transform

    def render(self, link_ids=None, log_dir="."):
        """
        Render current camera view
        Return (list):
            All visible links ids
        """
        self.update()
        image, _, labelled_image, visible_links = self.display.render(link_ids)
        cv2.imwrite(os.path.join(f"{log_dir}/rgb.png"), image[:, :, ::-1])
        cv2.imwrite(os.path.join(f"{log_dir}/label.png"), labelled_image[:, :, ::-1])
        return visible_links.tolist()

    def render_from_x(self, camera_lookat, link_ids=None, log_dir="."):
        """
        Get an image from x (front)
        Return (list):
            All visible links ids
        """
        self.update()
        initial_camera_pose = self.get_camera_pose()
        camera_pose = initial_camera_pose.copy()
        camera_pose["azimuth"] = 0
        camera_pose["elevation"] = 0
        camera_pose["lookat"] = camera_lookat
        self.set_camera_pose(camera_pose)
        self.display.visualize_link_frame(camera_lookat, x=False)
        image, _, labelled_image, visible_links = self.display.render(link_ids)
        camera_transform = self.pack_camera_transform()
        cv2.imwrite(os.path.join(f"{log_dir}/rgb_x.png"), image[:, :, ::-1])
        cv2.imwrite(os.path.join(f"{log_dir}/label_x.png"), labelled_image[:, :, ::-1])
        self.set_camera_pose(initial_camera_pose)
        self.display.clear_debug_objects()
        return visible_links.tolist(), camera_transform

    def render_from_y(self, camera_lookat, link_ids=None, log_dir="."):
        """
        Get an image from y (left)
        Return (list):
            All visible links ids
        """
        self.update()
        initial_camera_pose = self.get_camera_pose()
        camera_pose = initial_camera_pose.copy()
        camera_pose["azimuth"] = 90
        camera_pose["elevation"] = 0
        camera_pose["lookat"] = camera_lookat
        self.set_camera_pose(camera_pose)
        self.display.visualize_link_frame(camera_lookat, y=False)
        image, _, labelled_image, visible_links = self.display.render(link_ids)
        camera_transform = self.pack_camera_transform()
        cv2.imwrite(os.path.join(f"{log_dir}/rgb_y.png"), image[:, :, ::-1])
        cv2.imwrite(os.path.join(f"{log_dir}/label_y.png"), labelled_image[:, :, ::-1])
        self.set_camera_pose(initial_camera_pose)
        self.display.clear_debug_objects()
        return visible_links.tolist(), camera_transform

    def render_from_z(self, camera_lookat, link_ids=None, log_dir="."):
        """
        Get an image from z (up)
        Return (list):
            All visible links ids
        """
        self.update()
        initial_camera_pose = self.get_camera_pose()
        camera_pose = initial_camera_pose.copy()
        camera_pose["azimuth"] = 180
        camera_pose["elevation"] = 89
        camera_pose["lookat"] = camera_lookat
        self.set_camera_pose(camera_pose)
        self.display.visualize_link_frame(camera_lookat, z=False)
        image, _, labelled_image, visible_links = self.display.render(link_ids)
        camera_transform = self.pack_camera_transform()
        cv2.imwrite(os.path.join(f"{log_dir}/rgb_z.png"), image[:, :, ::-1])
        cv2.imwrite(os.path.join(f"{log_dir}/label_z.png"), labelled_image[:, :, ::-1])
        self.set_camera_pose(initial_camera_pose)
        self.display.clear_debug_objects()
        return visible_links.tolist(), camera_transform

    def render_from_nx(self, camera_lookat, link_ids=None, log_dir="."):
        """
        Get an image from -x (back)
        Return (list):
            All visible links ids
        """
        self.update()
        initial_camera_pose = self.get_camera_pose()
        camera_pose = initial_camera_pose.copy()
        camera_pose["azimuth"] = 180
        camera_pose["elevation"] = 0
        camera_pose["lookat"] = camera_lookat
        self.set_camera_pose(camera_pose)
        self.display.visualize_link_frame(camera_lookat, x=False)
        image, _, labelled_image, visible_links = self.display.render(link_ids)
        camera_transform = self.pack_camera_transform()
        cv2.imwrite(os.path.join(f"{log_dir}/rgb_-x.png"), image[:, :, ::-1])
        cv2.imwrite(os.path.join(f"{log_dir}/label_-x.png"), labelled_image[:, :, ::-1])
        self.set_camera_pose(initial_camera_pose)
        self.display.clear_debug_objects()
        return visible_links.tolist(), camera_transform

    def render_from_ny(self, camera_lookat, link_ids=None, log_dir="."):
        """
        Get an image from -y (right)
        Return (list):
            All visible links ids
        """
        self.update()
        initial_camera_pose = self.get_camera_pose()
        camera_pose = initial_camera_pose.copy()
        camera_pose["azimuth"] = 270
        camera_pose["elevation"] = 0
        camera_pose["lookat"] = camera_lookat
        self.set_camera_pose(camera_pose)
        self.display.visualize_link_frame(camera_lookat, y=False)
        image, _, labelled_image, visible_links = self.display.render(link_ids)
        camera_transform = self.pack_camera_transform()
        cv2.imwrite(os.path.join(f"{log_dir}/rgb_-y.png"), image[:, :, ::-1])
        cv2.imwrite(os.path.join(f"{log_dir}/label_-y.png"), labelled_image[:, :, ::-1])
        self.set_camera_pose(initial_camera_pose)
        self.display.clear_debug_objects()
        return visible_links.tolist(), camera_transform

    def render_from_nz(self, camera_lookat, link_ids=None, log_dir="."):
        """
        Get an image from -z (down)
        Return (list):
            All visible links ids
        """
        self.update()
        initial_camera_pose = self.get_camera_pose()
        camera_pose = initial_camera_pose.copy()
        camera_pose["azimuth"] = 0
        camera_pose["elevation"] = -89
        camera_pose["lookat"] = camera_lookat
        self.set_camera_pose(camera_pose)
        self.display.visualize_link_frame(camera_lookat, z=False)
        image, _, labelled_image, visible_links = self.display.render(link_ids)
        camera_transform = self.pack_camera_transform()
        cv2.imwrite(os.path.join(f"{log_dir}/rgb_-z.png"), image[:, :, ::-1])
        cv2.imwrite(os.path.join(f"{log_dir}/label_-z.png"), labelled_image[:, :, ::-1])
        self.set_camera_pose(initial_camera_pose)
        self.display.clear_debug_objects()
        return visible_links.tolist(), camera_transform

    def render_from_xyz(self, camera_lookat, log_dir="."):
        """
        Render three different views from x (front), y (left) and z (up)
        Return (list):
            All visible links ids
        """
        all_visible_links = np.array([-1])
        camera_transforms = []
        visible_links, camera_transform = self.render_from_x(camera_lookat, log_dir=log_dir)
        all_visible_links = np.concatenate([all_visible_links, visible_links])
        camera_transforms.append(camera_transform)
        visible_links, camera_transform = self.render_from_y(camera_lookat, log_dir=log_dir)
        all_visible_links = np.concatenate([all_visible_links, visible_links])
        camera_transforms.append(camera_transform)
        visible_links, camera_transform = self.render_from_z(camera_lookat, log_dir=log_dir)
        all_visible_links = np.concatenate([all_visible_links, visible_links])
        camera_transforms.append(camera_transform)
        all_visible_links = np.unique(all_visible_links)
        return all_visible_links[all_visible_links != -1].tolist(), camera_transforms

    def render_from_nxyz(self, camera_lookat, log_dir="."):
        """
        Render three different views from -x (back), -y (right) and -z (down)
        Return (list):
            All visible links ids
        """
        all_visible_links = np.array([-1])
        camera_transforms = []
        visible_links, camera_transform = self.render_from_nx(camera_lookat, log_dir=log_dir)
        all_visible_links = np.concatenate([all_visible_links, visible_links])
        camera_transforms.append(camera_transform)
        visible_links, camera_transform = self.render_from_ny(camera_lookat, log_dir=log_dir)
        all_visible_links = np.concatenate([all_visible_links, visible_links])
        camera_transforms.append(camera_transform)
        visible_links, camera_transform = self.render_from_nz(camera_lookat, log_dir=log_dir)
        all_visible_links = np.concatenate([all_visible_links, visible_links])
        camera_transforms.append(camera_transform)
        all_visible_links = np.unique(all_visible_links)
        return all_visible_links[all_visible_links != -1].tolist(), camera_transforms

    def render_link(self, link_id, log_dir="."):
        self.update()
        initial_camera_pose = self.get_camera_pose()
        camera_pose = initial_camera_pose.copy()
        camera_pose["lookat"] = self.get_link_pos(link_id)
        camera_transforms = {}
        axis = []

        camera_pose["azimuth"] = 0
        camera_pose["elevation"] = 0
        self.set_camera_pose(camera_pose)
        self.display.visualize_link_frame(self.get_link_pos(link_id), x=False)
        image, segmentation_x, labelled_image, _ = self.display.render()
        camera_transforms["x"] = self.pack_camera_transform()
        cv2.imwrite(os.path.join(f"{log_dir}/rgb_x.png"), image[:, :, ::-1])
        cv2.imwrite(os.path.join(f"{log_dir}/label_x.png"), labelled_image[:, :, ::-1])
        camera_pose["azimuth"] = 180
        camera_pose["elevation"] = 0
        self.set_camera_pose(camera_pose)
        self.display.visualize_link_frame(self.get_link_pos(link_id), x=False)
        image, segmentation_nx, labelled_image, _ = self.display.render()
        camera_transforms["-x"] = self.pack_camera_transform()
        cv2.imwrite(os.path.join(f"{log_dir}/rgb_-x.png"), image[:, :, ::-1])
        cv2.imwrite(os.path.join(f"{log_dir}/label_-x.png"), labelled_image[:, :, ::-1])
        if np.sum(segmentation_x == link_id) > np.sum(segmentation_nx == link_id):
            axis.append("x")
        else:
            axis.append("-x")
        self.display.clear_debug_objects()

        camera_pose["azimuth"] = 90
        camera_pose["elevation"] = 0
        self.set_camera_pose(camera_pose)
        self.display.visualize_link_frame(self.get_link_pos(link_id), y=False)
        image, segmentation_y, labelled_image, _ = self.display.render()
        camera_transforms["y"] = self.pack_camera_transform()
        cv2.imwrite(os.path.join(f"{log_dir}/rgb_y.png"), image[:, :, ::-1])
        cv2.imwrite(os.path.join(f"{log_dir}/label_y.png"), labelled_image[:, :, ::-1])
        camera_pose["azimuth"] = 270
        camera_pose["elevation"] = 0
        self.set_camera_pose(camera_pose)
        self.display.visualize_link_frame(self.get_link_pos(link_id), y=False)
        image, segmentation_ny, labelled_image, _ = self.display.render()
        camera_transforms["-y"] = self.pack_camera_transform()
        cv2.imwrite(os.path.join(f"{log_dir}/rgb_-y.png"), image[:, :, ::-1])
        cv2.imwrite(os.path.join(f"{log_dir}/label_-y.png"), labelled_image[:, :, ::-1])
        if np.sum(segmentation_y == link_id) > np.sum(segmentation_ny == link_id):
            axis.append("y")
        else:
            axis.append("-y")
        self.display.clear_debug_objects()

        camera_pose["azimuth"] = 180
        camera_pose["elevation"] = 89
        self.set_camera_pose(camera_pose)
        self.display.visualize_link_frame(self.get_link_pos(link_id), z=False)
        image, segmentation_z, labelled_image, _ = self.display.render()
        camera_transforms["z"] = self.pack_camera_transform()
        cv2.imwrite(os.path.join(f"{log_dir}/rgb_z.png"), image[:, :, ::-1])
        cv2.imwrite(os.path.join(f"{log_dir}/label_z.png"), labelled_image[:, :, ::-1])
        camera_pose["azimuth"] = 0
        camera_pose["elevation"] = -89
        self.set_camera_pose(camera_pose)
        self.display.visualize_link_frame(self.get_link_pos(link_id), z=False)
        image, segmentation_nz, labelled_image, _ = self.display.render()
        camera_transforms["-z"] = self.pack_camera_transform()
        cv2.imwrite(os.path.join(f"{log_dir}/rgb_-z.png"), image[:, :, ::-1])
        cv2.imwrite(os.path.join(f"{log_dir}/label_-z.png"), labelled_image[:, :, ::-1])
        if np.sum(segmentation_z == link_id) > np.sum(segmentation_nz == link_id):
            axis.append("z")
        else:
            axis.append("-z")
        self.display.clear_debug_objects()

        camera_transforms = [camera_transforms[ax] for ax in axis]

        return camera_transforms, axis

    def update(self):
        "Update the robot display"
        self.display.update()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--robot', type=str, default='go2')
    parser.add_argument('-n', '--name', type=str, default='default')
    args = parser.parse_args()

    cfg_path = f"./cfgs/{args.robot}/basic.yaml"
    agent = Agent(cfg_path)
    # print(agent.display.joint_name_to_dof_order)
    # print(agent.get_joints_between_links(20, 22))
    # time.sleep(0.5)
    # agent.set_body_link(20)
    # agent.set_link_pose(0, torch.tensor([0., 0., 0.5]))
    # print(agent.get_link_pos(20))
    # print(agent.get_link_pos(0))
    # agent.display.update()
    # agent.render()
    # agent.render_from_xyz()
    # exit()

    # agent.run()

    task = "generate a pose that the hand walks with like a dog"
    agent.run(task)