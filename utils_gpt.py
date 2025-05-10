import re 
import yaml
import time

def extract_code(filename):
    in_code = False
    code_string = ''
    with open(filename, 'r') as file:
        for line in file:
            # Skip leading comments (lines starting with '#')
            if not in_code and line.lstrip().startswith('#'):
                continue
            
            # Start adding lines to code_string once we've found a non-commented line
            if not in_code and not line.lstrip().startswith('#'):
                in_code = True

            # Add line to code_string
            if in_code:
                code_string += line
    return code_string

def complete(client, messages):
    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
            ).choices[0].message.content
        except:
            time.sleep(1)
            continue
        break
    return response

def generate_state_wrapper(client, messages):
    prefix = """
import numpy as np
import torch

import genesis as gs
from envs.vec_env import VecEnv
from utils import *

"""
    while True:
        print("Start Generating...")
        response = complete(client, messages)

        try:
            code_pattern = r"```python(.*?)```"
            match = re.findall(code_pattern, response, re.DOTALL)[0]
            if "class StateEnv" in match:
                lines = match.splitlines()
                for i in range(len(lines)):
                    if "class StateEnv" in lines[i]:
                        lines = lines[i:]
                        break
                state_wrapper = "\n".join(lines)
                state_wrapper = prefix + state_wrapper
                with open("envs/state_wrapper_gpt.py", "w") as f:
                    f.write(state_wrapper)
                break
        except:
            print(response)
            print("Generating failed when extracting code.")
            continue

    return response

def generate_reward_wrapper(client, messages):
    prefix = """
import numpy as np
import torch

import genesis as gs
from envs.state_wrapper_gpt import StateEnv
from utils import *

"""
    while True:
        print("Start Generating...")
        response = complete(client, messages)

        try:
            code_pattern = r"```python(.*?)```"
            match = re.findall(code_pattern, response, re.DOTALL)[0]
            if "class RewardEnv" in match:
                lines = match.splitlines()
                for i in range(len(lines)):
                    if "class RewardEnv" in lines[i]:
                        lines = lines[i:]
                        break
                reward_wrapper = "\n".join(lines)
                reward_wrapper = prefix + reward_wrapper
                with open("envs/reward_wrapper_gpt.py", "w") as f:
                    f.write(reward_wrapper)
                break
        except:
            print(response)
            print("Generating failed when extracting code.")
            continue

    return response

def generate_cfg(client, messages, task):
    while True:
        response = complete(client, messages)

        try:
            yaml_pattern = r"```yaml(.*?)```"
            text_pattern = r"```text(.*?)```"
            yaml_file = re.findall(yaml_pattern, response, re.DOTALL)[0]
            with open("cfgs/cfg_gpt.yaml", "w") as f:
                f.write(yaml_file)
            with open("cfgs/cfg_gpt.yaml", "r") as f:
                cfg_gpt = yaml.safe_load(f)
            with open(f"cfgs/{task}.yaml", "r") as f:
                cfg = yaml.safe_load(f)
            for key in cfg_gpt["environment"].keys():
                if key not in cfg["environment"].keys():
                    cfg["environment"][key] = cfg_gpt["environment"][key]
            with open("cfgs/leap_hand_gpt.yaml", "w") as f:
                yaml.safe_dump(cfg, f)
            text = re.findall(text_pattern, response, re.DOTALL)[0]
            with open("cfgs/description.txt", "w") as f:
                f.write(text)
        except:
            print(response)
            print("Generating failed when extracting cfg.")
            continue
        break

    return response