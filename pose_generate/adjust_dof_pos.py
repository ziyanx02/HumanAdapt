import os
import yaml
import argparse
import threading

from robot_display.gui_display import GUIDisplay

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--robot', type=str, default='leap_hand')
parser.add_argument('-n', '--name', type=str, default=None)
parser.add_argument('-s', '--save', type=str, default='dof_pos')
args = parser.parse_args()

cfg = yaml.safe_load(open(f"./cfgs/{args.robot}/basic.yaml"))
if args.name is not None:
    cfg = yaml.safe_load(open(f"./cfgs/{args.robot}/{args.name}.yaml"))

def save(display):
    cfg["robot"]["link_names"] = display.robot.link_name
    cfg["robot"]["base_init_quat"] = [round(val, 5) for val in display.robot.base_quat.tolist()]
    cfg["robot"]["body_init_quat"] = [round(val, 5) for val in display.robot.body_quat.tolist()]
    dof_pos = display.robot.target_dof_pos.tolist()
    cfg["robot"]["default_dof_pos"] = {display.robot.dof_name[i] : round(dof, 5) for i, dof in enumerate(dof_pos)}

    yaml.safe_dump(cfg, open(f"./cfgs/{args.robot}/{args.save}.yaml", "w"), sort_keys=False)
    print("Save to", f"./cfgs/{args.robot}/{args.save}.yaml")

class VisOptions:
    def __init__(self):
        self.visualize_skeleton = False
        self.visualize_robot_frame = True
        self.visualize_target_foot_pos = False
        self.merge_fixed_links = True
        self.show_world_frame = True
        self.shadow = False
        self.background_color = (0.8, 0.8, 0.8)

'''
For hand-designed pose
'''
display = GUIDisplay(
    cfg=cfg,
    body_pos=True,
    body_pose=True,
    dofs_pos=True,
    pd_control=False,
    save_callable=save,
    vis_options=VisOptions(),
)

def run():
    display.run()
display_thread = threading.Thread(target=run)
display_thread.start()