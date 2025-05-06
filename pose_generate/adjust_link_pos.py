import os
import yaml
import argparse
import threading

from robot_display.gui_display import GUIDisplay

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--robot', type=str, default='leap_hand')
parser.add_argument('-n', '--name', type=str, default='basic')
parser.add_argument('-c', '--cfg', type=str, default=None)
args = parser.parse_args()

cfg = yaml.safe_load(open(f"./cfgs/{args.robot}/{args.name}_body_rot.yaml"))
if args.cfg is not None:
    cfg = yaml.safe_load(open(f"./cfgs/{args.robot}/{args.cfg}.yaml"))

def save(display):
    cfg["robot"]["link_pos"] = {}
    for i in range(len(display.robot.links)):
        link = display.robot.links[i]
        cfg["robot"]["link_pos"][link.name] = display.robot.links_pos[i].numpy().tolist()
    cfg["robot"]["default_dof_pos"] = {}
    default_dof_pos = [round(val, 2) for val in display.robot.dof_pos.numpy().tolist()]
    for i in range(display.robot.num_dofs):
        cfg["robot"]["default_dof_pos"][display.robot.dof_name[i]] = default_dof_pos[i]
    yaml.safe_dump(cfg, open(f"./cfgs/{args.robot}/{args.name}_link_pos.yaml", "w"))
    print("Save to", f"./cfgs/{args.robot}/{args.name}_link_pos.yaml")

class VisOptions:
    def __init__(self):
        self.visualize_skeleton = False
        self.visualize_target_foot_pos = False
        self.merge_fixed_links = True
        self.show_world_frame = True
        self.shadow = False
        self.background_color = (0.8, 0.8, 0.8)

display = GUIDisplay(
    cfg=cfg,
    body_pos=True,
    body_pose=False,
    dofs_pos=False,
    foot_pos=True,
    save_callable=save,
    vis_options=VisOptions(),
)

def run():
    display.run()
display_thread = threading.Thread(target=run)
display_thread.start()

# import time
# time.sleep(1)
# display.render()