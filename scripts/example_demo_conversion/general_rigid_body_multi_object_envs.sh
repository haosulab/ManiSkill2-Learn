#!/bin/bash 

# For general envs with multiple objects, where each object has its own demos,
# and goal information is not provided in the observation
# i.e. TurnFaucet-v0, OpenCabinetDoor-v0, OpenCabinetDrawer-v0, PushChair-v0, MoveBucket-v0
ENV="TurnFaucet-v0" 

# unzip the demonstration for TurnFaucet, if not already
if [[ $ENV =~ "TurnFaucet-v0" ]]; then
    cd ../ManiSkill2/demos/rigid_body/$ENV/
    if [[ -f "20220815.zip" ]]; then
        unzip 20220815.zip
        rm 20220815.zip
    fi
    cd - # ManiSkill2-Learn
    python tools/merge_trajectory.py \
    -i ../ManiSkill2/demos/rigid_body/$ENV/ \
    -o ../ManiSkill2/demos/rigid_body/$ENV/trajectory_merged.h5 \
    -p 5*.h5 # this can be replaced with other patterns 
else
    # Assume current working directory is ManiSkill2-Learn/
    # Assume conda environment contains all dependencies of ManiSkill2 and ManiSkill2-Learn 
    # Inside ManiSkill2's directory, run merge_trajectory to output merged h5 and json files 
    python tools/merge_trajectory.py \
    -i ../ManiSkill2/demos/rigid_body/$ENV/ \
    -o ../ManiSkill2/demos/rigid_body/$ENV/trajectory_merged.h5 \
    -p trajectory.h5 # this can be replaced with other patterns 
fi

# Inside the ManiSkill2's directory, run replay_trajectory.py. See wiki page
# of ManiSkill2 for more information.
cd ../ManiSkill2
python mani_skill2/trajectory/replay_trajectory.py --num-procs 32 \
--traj-path demos/rigid_body/$ENV/trajectory_merged.h5 \
--save-traj \
--target-control-mode pd_ee_delta_pose \
--obs-mode none

# Inside ManiSkill2-Learn's directory, run convert_state.py to generate visual observations
# for the demonstrations.
cd ../ManiSkill2-Learn
# Generate pointcloud demo
python tools/convert_state.py \
--env-name=$ENV \
--num-procs=12 \
--traj-name=../ManiSkill2/demos/rigid_body/$ENV/trajectory_merged.none.pd_ee_delta_pose.h5  \
--json-name=../ManiSkill2/demos/rigid_body/$ENV/trajectory_merged.none.pd_ee_delta_pose.json \
--output-name=../ManiSkill2/demos/rigid_body/$ENV/trajectory_merged.none.pd_ee_delta_pose_pointcloud.h5 \
--control-mode=pd_ee_delta_pose \
--max-num-traj=-1 \
--obs-mode=pointcloud \
--reward-mode=dense \
--obs-frame=ee \
--n-points=1200

# Generate rgbd demo 
python tools/convert_state.py \
--env-name=$ENV \
--num-procs=12 \
--traj-name=../ManiSkill2/demos/rigid_body/$ENV/trajectory_merged.none.pd_ee_delta_pose.h5 \
--json-name=../ManiSkill2/demos/rigid_body/$ENV/trajectory_merged.none.pd_ee_delta_pose.json \
--output-name=../ManiSkill2/demos/rigid_body/$ENV/trajectory_merged.none.pd_ee_delta_pose_rgbd.h5 \
--control-mode=pd_ee_delta_pose \
--max-num-traj=-1 \
--obs-mode=rgbd \
--reward-mode=dense

# Shuffle pointcloud demos
python tools/shuffle_demo.py \
--source-file ../ManiSkill2/demos/rigid_body/$ENV/trajectory_merged.none.pd_ee_delta_pose_pointcloud.h5 \
--target-file ../ManiSkill2/demos/rigid_body/$ENV/trajectory_merged.none.pd_ee_delta_pose_pointcloud_shuffled.h5

# Shuffle rgbd demos 
python tools/shuffle_demo.py \
--source-file ../ManiSkill2/demos/rigid_body/$ENV/trajectory_merged.none.pd_ee_delta_pose_rgbd.h5 \
--target-file ../ManiSkill2/demos/rigid_body/$ENV/trajectory_merged.none.pd_ee_delta_pose_rgbd_shuffled.h5