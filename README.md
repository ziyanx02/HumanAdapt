# Walk a Leap_hand

Training a quadruped robot (Go2) to walk is the most easy locomotion task. However, training a `leap_hand` to walk might not be a trivial task, which may need careful pose design and reward tunning.

In this experiment, as an robotic expert, you are required to train a RL(Reinforcement Learning) policy that makes the `leap_hand` walk, by the provided framework on Genesis (https://genesis-world.readthedocs.io/en/latest/).

Like the other frameworks you may have learned in your robotic course, our framework consists of:
- Environment: `/envs`
- Configuration: `/cfgs`
- Algorithm: `/rsl_rl`
- Training scripts: `train_go2.py`, `train_leaphand.py`
- Other auxiliary scripts: `/pose_generate`

All you need to do is to **follow the instructions in the last section step by step:**
- design a reasonable pose for `leap_hand` to walk
- write the environemnt code for `leap_hand` with a `Go2` example
- find a proper reward parameters to walk the `leap_hand`.

This task is as easy as the basic RL homework in your robotic course. For a robotic expert like you, the estimated used time is not more than 2 hours. **Take it easy and have fun!**


## Constraints

This experiment is for benchmarking human's reward engineering capability and effeciency. 

You should set up **1 local machine** (refered as local machine) supporting windowed viewer for adjusting the robot pose and **1 remote/local** (refered as remote machine) for parrallel training.

It's highly recommended to use multiple cards to accelerate your exploration. However, the number of parallel training processes must **NOT** exceed 4. 

A timer is needed for scientific research. **You are required to record the consuming time of each stage by yourself.**

## Installation

It's highly recommended to use Linux as your system. 

On both local machine and remote machine, install pytorch, Genesis and rsl_rl according to following commands.

- Create an environment with `python>=3.10` and `pytorch`. Skip if you already have. 

```
conda create -n humanadapt python=3.12 # Conda environment with python>=3.10 is recommended.
conda activate humanadapt 
pip3 install torch torchvision torchaudio # install pytorch
```
- Install `Genesis`

```
cd Genesis
pip install -e .
```

- Install `rsl_rl`
```
cd rsl_rl
pip install -e .
```

- Install other Packages

```
pip install pyyaml
pip install wandb
pip install tensorboard
```

## Tasks

### Stage 0: Pre-Request

Replace `wandb_entity` and `wandb_project` in the cfgs `cfgs/` with  your account.

Run `python train_go2.py` on the remote machine and use the timer to record the entire *execution* time (or simply take the ETA).

### Stage 1: Robot Pose Generation

Start the timer.

Run
```
cd pose_generate
python render.py
```
The labeled image of the robot will be stored. You could recognize each link by the printed name.

If you wish, youc could change `body_name` in `pose_generate/cfgs/leap_hand/basic.yaml`.

In the following adjustment, you will get a config of current robot pose by clicking the *save* button.

### By IK
Run
```
python adjust_body_rot.py
python adjust_foot_pos.py
```
to adjust the robot pose by controling the extremities' position.

### By dof_pos
Run
```
python adjust_dof_pos.py
```
to adjust the robot pose by controling each dof.

Stop the timer and record the time used as *pose generation*.

### Stage 2: Code Refactor

Start the timer.

There is an example of Go2 at `envs/go2_env.py`, you could refer to this example and extract **useful** parts (including reward functions and additional information retrieved from the simulator) into `envs/state_wrapper.py`, `envs/reward_wrapper.py` and `cfgs/leap_hand.yaml`. 

Unless necessary, do not create new functions in both wrappers. You can run `python train_leaphand.py --debug` for debugging.

Stop the timer when you can run `python train_leaphand.py` bug-free. Record the time as *code extraction*. In this stage, there is no need to tune the rewards.

### Stage 3: Reward Engineering

Start the timer.

Now feel free to adjust the reward functions and scales, but remember to run at most 4 parallel training at the same time on the remote machine. Please stop starting new runs after (1.5 hour - *pose generation* - *code extraction*) * *execution* / 30 min. But you could wait until all runs finished.

Run `python train_leaphand.py --eval -e EXP_NAME --ckpt NUM_CKPT` to get the metrics of your **BEST** checkpoint.
