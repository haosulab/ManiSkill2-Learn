python maniskill2_learn/apis/run_rl.py configs/mfrl/ppo/maniskill2_pn.py \
            --work-dir YOUR_LOGGING_DIRECTORY --gpu-ids 0 \
            --cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=pointcloud" "env_cfg.n_points=1200" \
            "env_cfg.control_mode=pd_joint_delta_pos" \
            "eval_cfg.num=100" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" 

# The above command does automatic evaluation after training. Alternatively, you can manually evaluate a model checkpoint 
# by appending --evaluation and  --resume-from YOUR_LOGGING_DIRECTORY/models/SOME_CHECKPOINT.ckpt to the above commands.            


# Using multiple GPUs will increase training speed; 
# Note that train_cfg.n_steps will also be multiplied by the number of gpus you use, so you may want to divide it by the number of gpus