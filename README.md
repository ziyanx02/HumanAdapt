# Human Experiment

This experiment is for benchmarking human's reward engineering capability and effeciency. You should set up 1 local machine (refered as local machine) supporting windowed viewer for adjusting the robot pose and 1 remote/local (the number of grphic cards used should not exceed 4, refered as remote machine) for parralel training. A timer is need.

## Installation

On both local machine and remote machine, install Genesis and rsl_rl according to following commands.

### Genesis

```
cd Genesis
pip install -e .
```

### rsl_rl
```
cd rsl_rl
pip install -e .
```

### Other Packages

```
pip install yaml
pip install wandb
pip install tensorboard
```

## Pre-request

Replace `wandb_entity` and `wandb_project` in the cfgs `cfgs/`.

Run `python train_go2.py` on the remote machine and use the timer to record the entire *execution* time (or simply take the ETA).

## Stage 1: Robot Pose Generation

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

## Stage 2: Code Refactor

Start the timer.

There is an example of Go2 at `envs/go2_env.py`, you could refer to this example and extract **useful** parts (including reward functions and additional information retrieved from the simulator) into `envs/state_wrapper.py`, `envs/reward_wrapper.py` and `cfgs/leap_hand.yaml`. 

Unless necessary, do not create new functions in both wrappers. You can run `python train_leaphand.py --debug` for debugging.

Stop the timer when you can run `python train_leaphand.py` bug-free. Record the time as *code extraction*. In this stage, there is no need to tune the rewards.

## Stage 3: Reward Engineering

Start the timer.

Now feel free to adjust the reward functions and scales, but remember to run at most 4 parallel training at the same time on the remote machine. Please stop starting new runs after (1.5 hour - *pose generation* - *code extraction*) * *execution* / 10 min. But you could wait until all runs finished.

Run `python train_leaphand.py --eval -e EXP_NAME --ckpt NUM_CKPT` to get the metrics of your **BEST** checkpoint.
