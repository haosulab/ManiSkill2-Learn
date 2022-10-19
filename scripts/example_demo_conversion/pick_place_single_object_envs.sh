#!/bin/bash 

# For Pick & Place envs with single object. These envs provide goal position in observation;
# Currently only PickCube-v0 ibelongs to such environment
ENV="PickCube-v0" 

# Assume current working directory is ManiSkill2-Learn/
# Assume conda environment contains all dependencies of ManiSkill2 and ManiSkill2-Learn 
# Inside the ManiSkill2's directory, run replay_trajectory.py. See wiki page
# of ManiSkill2 for more information.
cd ../ManiSkill2
python tools/replay_trajectory.py --num-procs 32 \
--traj-path demos/rigid_body_envs/$ENV/trajectory.h5 \
--save-traj \
--target-control-mode pd_ee_delta_pose \
--obs-mode none 

# Inside ManiSkill2-Learn's directory, run convert_state.py to generate visual observations
# for the demonstrations.
cd ../ManiSkill2-Learn

# Generate pointcloud demo; num_procs=64 takes about 30GB memory and 32 cpus
python tools/convert_state.py \
--env-name=$ENV \
--num-procs=32 \
--traj-name=../ManiSkill2/demos/rigid_body_envs/$ENV/trajectory.none.pd_ee_delta_pose.h5  \
--json-name=../ManiSkill2/demos/rigid_body_envs/$ENV/trajectory.none.pd_ee_delta_pose.json \
--output-name=../ManiSkill2/demos/rigid_body_envs/$ENV/trajectory.none.pd_ee_delta_pose_pointcloud.h5 \
--control-mode=pd_ee_delta_pose \
--max-num-traj=-1 \
--obs-mode=pointcloud \
--reward-mode=dense \
--obs-frame=ee \
--n-points=1200 \
--n-goal-points=50

# Generate rgbd demo 
python tools/convert_state.py \
--env-name=$ENV \
--num-procs=32 \
--traj-name=../ManiSkill2/demos/rigid_body_envs/$ENV/trajectory.none.pd_ee_delta_pose.h5 \
--json-name=../ManiSkill2/demos/rigid_body_envs/$ENV/trajectory.none.pd_ee_delta_pose.json \
--output-name=../ManiSkill2/demos/rigid_body_envs/$ENV/trajectory.none.pd_ee_delta_pose_rgbd.h5 \
--control-mode=pd_ee_delta_pose \
--max-num-traj=-1 \
--obs-mode=rgbd \
--reward-mode=dense