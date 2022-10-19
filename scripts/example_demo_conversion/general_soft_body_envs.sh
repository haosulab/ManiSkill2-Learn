#!/bin/bash 

# For all soft-body environments
# i.e. Fill-v0, Write-v0, Pinch-v0, Hang-v0, Pour-v0, Excavate-v0
ENV="Excavate-v0" 

# Create a dummy environment to initialize cache if it has not been done in the history, otherwise 
# we can't do multi-processing below
CUDA_VISIBLE_DEVICES=1 python -c "import mani_skill2.envs, gym; env=gym.make('$ENV'); env.reset();"

# Assume current working directory is ManiSkill2-Learn/
# Assume conda environment contains all dependencies of ManiSkill2 and ManiSkill2-Learn 
# Inside the ManiSkill2's directory, run replay_trajectory.py. See wiki page
# of ManiSkill2 for more information.
cd ../ManiSkill2
CUDA_VISIBLE_DEVICES=1 python tools/replay_trajectory.py --num-procs 4 \
--traj-path demos/soft_body_envs/$ENV/trajectory.h5 \
--save-traj \
--target-control-mode pd_ee_delta_pose \
--obs-mode none 

# Inside ManiSkill2-Learn's directory, run convert_state.py to generate visual observations
# for the demonstrations.
cd ../ManiSkill2-Learn

# Generate pointcloud demo
CUDA_VISIBLE_DEVICES=1 python tools/convert_state.py \
--env-name=$ENV \
--num-procs=4 \
--traj-name=../ManiSkill2/demos/soft_body_envs/$ENV/trajectory.none.pd_ee_delta_pose.h5 \
--json-name=../ManiSkill2/demos/soft_body_envs/$ENV/trajectory.none.pd_ee_delta_pose.json \
--output-name=../ManiSkill2/demos/soft_body_envs/$ENV/trajectory.none.pd_ee_delta_pose_pointcloud.h5 \
--control-mode=pd_ee_delta_pose \
--max-num-traj=-1 \
--obs-mode=pointcloud \
--reward-mode=dense \
--obs-frame=ee \
--n-points=1200

# Generate rgbd demo 
CUDA_VISIBLE_DEVICES=1 python tools/convert_state.py \
--env-name=$ENV \
--num-procs=4 \
--traj-name=../ManiSkill2/demos/soft_body_envs/$ENV/trajectory.none.pd_ee_delta_pose.h5 \
--json-name=../ManiSkill2/demos/soft_body_envs/$ENV/trajectory.none.pd_ee_delta_pose.json \
--output-name=../ManiSkill2/demos/soft_body_envs/$ENV/trajectory.none.pd_ee_delta_pose_rgbd.h5 \
--control-mode=pd_ee_delta_pose \
--max-num-traj=-1 \
--obs-mode=rgbd \
--reward-mode=dense