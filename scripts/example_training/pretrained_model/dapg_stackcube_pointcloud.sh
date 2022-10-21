# Assuming 2 gpus each with 12GB memory; 
# if you have a GPU with more memory (e.g. 24GB), you can set --gpu-ids and --sim-gpu-ids to be the same;
# if you only have one GPU with small memory, then you can set a smaller rollout_cfg.num_procs (e.g. =5)

python maniskill2_learn/apis/run_rl.py configs/mfrl/dapg/maniskill2_pn.py \
            --work-dir ./logs/dapg_stackcube_pointcloud --gpu-ids 0 --sim-gpu-ids 1 \
            --cfg-options "env_cfg.env_name=StackCube-v0" "env_cfg.obs_mode=pointcloud" "env_cfg.n_points=1200" \
            "rollout_cfg.num_procs=16" "env_cfg.reward_mode=dense" \
            "env_cfg.control_mode=pd_ee_delta_pose" "env_cfg.obs_frame=ee" \
            "agent_cfg.demo_replay_cfg.buffer_filenames=../ManiSkill2/demos/rigid_body_envs/StackCube-v0/trajectory.none.pd_ee_delta_pose_pointcloud.h5" \
            "eval_cfg.num=100" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
            "train_cfg.total_steps=25000000" "train_cfg.n_checkpoint=5000000"