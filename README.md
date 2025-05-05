# Human Experiment

## Installation

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

## Pre-request

Run `python train_go2.py` and record the execution time.

## Stage 1: Robot Pose Generation

```
cd pose_generate
python render.py
```
Change `body_name` in `pose_generate/cfgs/leap_hand/basic.yaml`.

### By IK
```
python adjust_body_rot.py
python adjust_foot_pos.py
```

### By dof_pos
```
python adjust_dof_pos.py
```

## Stage 2: Code Refactor
Rewrite functions in `envs/state_wrapper.py` and `envs/reward_wrapper.py` and config in `cfgs/leap_hand.yaml` to run `train_leaphand.py` bug-free.

## Stage 3: Reward Engineering
Run at most 4 parallel training at the same time. Tune the reward as good as possible in 1 hour.
