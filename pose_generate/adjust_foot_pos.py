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

cfg = yaml.safe_load(open(f"./cfgs/{args.robot}/{args.name}_body_name.yaml"))
if args.cfg is not None:
    cfg = yaml.safe_load(open(f"./cfgs/{args.robot}/{args.cfg}.yaml"))

def save(display):
    cfg["robot"]["foot_pos"] = {}
    for i in range(len(display.robot.foot_links)):
        link = display.robot.foot_links[i]
        pos = display.values[display.value_foot_pos_idx_start + 3 * i:display.value_foot_pos_idx_start + 3 * (i + 1)]
        pos = [round(val, 2) for val in pos]
        cfg["robot"]["foot_pos"][link.name] = pos
    cfg["control"]["default_dof_pos"] = {}
    default_dof_pos = [round(val, 2) for val in display.robot.dof_pos.numpy().tolist()]
    for i in range(display.robot.num_dofs):
        cfg["control"]["default_dof_pos"][display.robot.dof_name[i]] = default_dof_pos[i]
    yaml.safe_dump(cfg, open(f"./cfgs/{args.robot}/{args.name}_foot_pos.yaml", "w"))
    print("Save to", f"./cfgs/{args.robot}/{args.name}_foot_pos.yaml")

class VisOptions:
    def __init__(self):
        self.visualize_skeleton = False
        self.visualize_target_foot_pos = False
        self.merge_fixed_links = True
        self.show_world_frame = False
        self.shadow = False
        self.background_color = (0.8, 0.8, 0.8)

display = GUIDisplay(
    cfg=cfg,
    body_pos=False,
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