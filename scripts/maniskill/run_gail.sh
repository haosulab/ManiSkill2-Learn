# Point Cloud-based GAIL

# ** Before you run it, check that demonstrations are converted with "--with-next" and rewards behave as intended. **

python maniskill2_learn/apis/run_rl.py configs/mfrl/gail/maniskill2_pn.py \
--gpu-ids 0 --work-dir YOUR_LOGGING_DIRECTORY --print-steps 16 \
--cfg-options "env_cfg.env_name=PlugCharger-v0" "env_cfg.control_mode=pd_joint_delta_pos" \
"replay_cfg.buffer_filenames=[PATH_TO_DEMO.h5]" \
"expert_replay_cfg.buffer_filenames=[PATH_TO_DEMO.h5]"

# Using multiple GPUs will increase training speed; 
# Note that the effective batch size is multiplied by the number of gpus; large batch can be crucial for stabilizing GAIL training




