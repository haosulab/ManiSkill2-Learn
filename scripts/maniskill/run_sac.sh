# Point Cloud-based SAC
python maniskill2_learn/apis/run_rl.py configs/mfrl/sac/maniskill2_pn.py \
--gpu-ids 0 --work-dir YOUR_LOGGING_DIRECTORY --print-steps 16 \
--cfg-options "env_cfg.env_name=PegInsertionSide-v0" "env_cfg.control_mode=pd_joint_delta_pos"

# Using multiple GPUs will increase training speed; 
# Note that the effective batch size is multiplied by the number of gpus; large batch can be crucial for stabilizing SAC training


# State-based SAC for debugging purposes
"""
python maniskill2_learn/apis/run_rl.py configs/mfrl/sac/maniskill2_state.py \
--work-dir YOUR_LOGGING_DIRECTORY --gpu-ids 0 \
--cfg-options 'env_cfg.env_name=PegInsertionSide-v0' 'env_cfg.control_mode=pd_joint_delta_pos'
"""



