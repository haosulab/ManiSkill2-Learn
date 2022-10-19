python maniskill2_learn/apis/run_rl.py configs/mfrl/ppo/maniskill2_pn_dapg.py \
            --work-dir YOUR_LOGGING_DIRECTORY --gpu-ids 0 \
            --cfg-options "env_cfg.env_name=PegInsertionSide-v0" "env_cfg.obs_mode=pointcloud" "env_cfg.n_points=1200" \
            "env_cfg.reward_mode=dense" "env_cfg.control_mode=pd_joint_delta_pos" \
            "agent_cfg.demo_replay_cfg.buffer_filenames=PATH_TO_POINT_CLOUD_DEMO.h5"
            "eval_cfg.num=100" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" 
# To manually evaluate the model, add --evaluation and --resume-from YOUR_LOGGING_DIRECTORY/models/SOME_CHECKPOINT.ckpt 
# to the above commands.

# Using multiple GPUs will increase training speed; 
# Note that train_cfg.n_steps will also be multiplied by the number of gpus you use, so you may want to divide it by the number of gpus
