#!/bin/bash 

# For environments migrated from ManiSkill1
# i.e. OpenCabinetDoor-v1, OpenCabinetDrawer-v1, PushChair-v1, MoveBucket-v1
ENV="OpenCabinetDrawer-v1" 

# Assume current working directory is ManiSkill2-Learn/
cd ../ManiSkill2

find demos/rigid_body/$ENV -name "*trajectory.h5" | while read line; do

CUDA_VISIBLE_DEVICES=1 python mani_skill2/trajectory/replay_trajectory.py --num-procs 16 \
--traj-path $line \
--use-env-states \
--save-traj \
--obs-mode none 

# Inside ManiSkill2-Learn's directory, run convert_state.py to generate visual observations
# for the demonstrations.
cd ../ManiSkill2-Learn

Generate pointcloud demo
python tools/convert_state.py \
--env-name=$ENV \
--num-procs=12 \
--traj-name=../ManiSkill2/demos/rigid_body/$ENV/trajectory.none.base_pd_joint_vel_arm_pd_ee_delta_pose.h5 \
--json-name=../ManiSkill2/demos/rigid_body/$ENV/trajectory.none.base_pd_joint_vel_arm_pd_ee_delta_pose.json \
--output-name=../ManiSkill2/demos/rigid_body/$ENV/trajectory.none.base_pd_joint_vel_arm_pd_ee_delta_pose_pointcloud.h5 \
--control-mode=base_pd_joint_vel_arm_pd_joint_vel \
--max-num-traj=-1 \
--obs-mode=pointcloud \
--reward-mode=dense \
--obs-frame=ee \
--n-points=1200

# Generate rgbd demo 
CUDA_VISIBLE_DEVICES=1 python tools/convert_state.py \
--env-name=$ENV \
--num-procs=12 \
--traj-name=../ManiSkill2/"$(dirname "$line")"/trajectory.none.base_pd_joint_vel_arm_pd_joint_vel.h5 \
--json-name=../ManiSkill2/"$(dirname "$line")"/trajectory.none.base_pd_joint_vel_arm_pd_joint_vel.json \
--output-name=../ManiSkill2/"$(dirname "$line")"/trajectory.none.base_pd_joint_vel_arm_pd_joint_vel_rgbd.h5 \
--control-mode=base_pd_joint_vel_arm_pd_joint_vel \
--max-num-traj=-1 \
--obs-mode=rgbd \
--reward-mode=dense 

cd ../ManiSkill2

done
