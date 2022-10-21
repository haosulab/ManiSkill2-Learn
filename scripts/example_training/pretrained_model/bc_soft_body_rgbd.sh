# Replace --work-dir, env_cfg.env_name, and replay_cfg.buffer_filenames when you run other environments

python maniskill2_learn/apis/run_rl.py configs/brl/bc/rgbd_soft_body.py \
--work-dir ./logs/dapg_excavate_rgbd --gpu-ids 0 \
--cfg-options "env_cfg.env_name=Excavate-v0" "env_cfg.obs_mode=rgbd" "env_cfg.n_points=1200" \
"env_cfg.control_mode=pd_joint_delta_pos" \
"replay_cfg.buffer_filenames=../ManiSkill2/demos/rigid_body_envs/Excavate-v0/trajectory.none.pd_ee_delta_pose_rgbd.h5" \
"eval_cfg.num=100" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"train_cfg.n_eval=50000" "train_cfg.total_steps=50000" "train_cfg.n_checkpoint=50000" "train_cfg.n_updates=500"