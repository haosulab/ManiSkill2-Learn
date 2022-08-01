
# Modified from Hao Shen, Weikang Wan, and He Wang's ManiSkill2021 challenge submission:
# Paper: https://arxiv.org/pdf/2203.02107.pdf
# Code: https://github.com/wkwan7/EPICLab-ManiSkill

# Note that in the ManiSkill 2021 Challenge, ground truth segmentation is given, and PointNet-Transformer was able to be applied as visual backbone.
# In the ManiSkill 2022 Challenge, the ground truth segmentation masks are removed, so we modified the backbone to be PointNet.

agent_cfg = dict(
    type="GAIL",
    batch_size=330, # Using multiple gpus leads to larger effective batch size, which can be crucial for GAIL training
    discriminator_batch_size=512,
    discriminator_update_freq=0.125,
    discriminator_update_n=5,
    episode_based_discriminator_update=True,
    env_reward_proportion=0.3,    
    gamma=0.95,
    update_coeff=0.005,
    alpha=0.2,
    target_update_interval=1,
    automatic_alpha_tuning=True,
    shared_backbone=True,
    detach_actor_feature=True,
    alpha_optim_cfg=dict(type="Adam", lr=3e-4),
    actor_cfg=dict(
        type="ContinuousActor",
        head_cfg=dict(
            type="TanhGaussianHead",
            log_std_bound=[-20, 2],
        ),
        nn_cfg=dict(
            type="Visuomotor",
            visual_nn_cfg=dict(type="PointNet", feat_dim="pcd_all_channel", mlp_spec=[64, 128, 512], feature_transform=[]),
            mlp_cfg=dict(
                type="LinearMLP",
                norm_cfg=None,
                mlp_spec=["512 + agent_shape", 256, 256, "action_shape * 2"],
                inactivated_output=True,
                zero_init_output=True,
            ),
        ),
        optim_cfg=dict(type="Adam", lr=3e-4, param_cfg={"(.*?)visual_nn(.*?)": None}),
        # *Above removes visual_nn from actor optimizer; should only do so if shared_backbone=True and detach_actor_feature=True
        # *If either of the config options is False, then param_cfg={} should be removed, i.e. actor should also update visual backbone.
        #   mlp_specs should be updated as well

        # *It is unknown if sharing backbone and detaching feature works well under the 3D setting. It is up to the users to figure this out.
    ),
    critic_cfg=dict(
        type="ContinuousCritic",
        num_heads=2,
        nn_cfg=dict(
            type="Visuomotor",
            visual_nn_cfg=None,
            mlp_cfg=dict(
                type="LinearMLP", norm_cfg=None, mlp_spec=["512 + agent_shape + action_shape", 256, 256, 1], inactivated_output=True, zero_init_output=True
            ),
        ),
        optim_cfg=dict(type="Adam", lr=3e-4),
    ),
    discriminator_cfg=dict(
        type="ContinuousCritic",
        num_heads=1,
        nn_cfg=dict(
            type="Visuomotor",
            visual_nn_cfg=dict(type="PointNet", feat_dim="pcd_all_channel", mlp_spec=[64, 128, 512], feature_transform=[]),
            mlp_cfg=dict(
                type="LinearMLP",
                norm_cfg=None,
                mlp_spec=["512 + agent_shape + action_shape", 256, 256, 1],
                inactivated_output=True,
                zero_init_output=True,
            ),
        ),
        optim_cfg=dict(type="Adam", lr=3e-4),
    ),    
)


train_cfg = dict(
    on_policy=False,
    total_steps=20000000,
    warm_steps=8000,
    n_eval=20000000,
    n_checkpoint=2000000,
    n_steps=64, # in multi-gpu training, n_steps will be the same number (here 64) for every gpu
    n_updates=4,
    ep_stats_cfg=dict(
        info_keys_mode=dict(
            success=[True, "max", "mean"],
        )
    ),
)

env_cfg = dict(
    type="gym",
    env_name="PickCube-v0",
    obs_mode='pointcloud',
    ignore_dones=True,
)


replay_cfg = dict(
    type="ReplayMemory",
    capacity=400000,
    buffer_filenames=[
        "PATH_TO_DEMO.h5", # initializing replay buffer with demonstrations
    ],
)

expert_replay_cfg = dict(
    type="ReplayMemory",
    capacity=-1, # auto-adjust capacity based on loaded demonstrations
    buffer_filenames=[
        "PATH_TO_DEMO.h5",
    ],
)

recent_traj_replay_cfg = dict(
    type="ReplayMemory",
    capacity=20000,
)

rollout_cfg = dict(
    type="Rollout",
    num_procs=4,
    with_info=True,
    multi_thread=False,
)

eval_cfg = dict(
    type="Evaluation",
    num_procs=1,
    num=10,
    use_hidden_state=False,
    save_traj=False,
    save_video=True,
    log_every_step=False,
    env_cfg=dict(ignore_dones=False),
)