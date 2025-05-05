import os
import yaml
import argparse
import threading

import numpy as np
import cv2
import pickle

from robot_display.display import Display
from viewer import Viewer, VisOptions

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--robot', type=str, default='leap_hand')
parser.add_argument('-n', '--name', type=str, default='basic')
args = parser.parse_args()

cfg_path = f"./cfgs/{args.robot}/{args.name}.yaml"
cfg = yaml.safe_load(open(cfg_path))

log_dir = os.path.dirname(os.path.abspath(__file__))

vis_options = VisOptions()
vis_options.show_viewer = False
agent = Viewer(cfg_path, vis_options=vis_options)

all_visible_links, _ = agent.render_from_xyz(agent.get_body_pos(), log_dir=log_dir)

print("Links visible in the rendered images:")
for i in all_visible_links[:-1]:
    print(f"link {i}:", agent.display.links[i].name)