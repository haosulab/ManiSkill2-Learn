
agent_cfg = dict(
    type="PPO",
    gamma=0.95,
    lmbda=0.95,
    critic_coeff=0.5,
    entropy_coeff=0,
    critic_clip=False,
    obs_norm=False,
    rew_norm=True,
    adv_norm=True,
    recompute_value=True,
    num_epoch=2,
    critic_warmup_epoch=4,
    batch_size=330,
    detach_actor_feature=False,
    max_grad_norm=0.5,
    eps_clip=0.2,
    max_kl=0.2,
    dual_clip=None,
    shared_backbone=True,
    ignore_dones=True,
    dapg_lambda=0.1,
    dapg_damping=0.995,
    actor_cfg=dict(
        type="ContinuousActor",
        head_cfg=dict(
            type="GaussianHead",
            init_log_std=-1,
            clip_return=True,
            predict_std=False,
        ),
        nn_cfg=dict(
            type="FrameMiners",
            visual_nn_cfg=dict(type="PointNet", feat_dim="pcd_all_channel", mlp_spec=[64, 128, 300], feature_transform=[]),
            num_frames=2, 
            vis_feat_dim=300,
            action_dim='action_shape', 
            robot_state_dim='agent_shape',
        ),
        optim_cfg=dict(type="Adam", lr=3e-4),
    ),
    critic_cfg=dict(
        type="ContinuousCritic",
        nn_cfg=dict(
            type="FrameMiners",
            visual_nn_cfg=None,
            num_frames=2, 
            vis_feat_dim=300,
            action_dim='action_shape', 
            robot_state_dim='agent_shape',
            is_critic=True,
        ),
        optim_cfg=dict(type="Adam", lr=3e-4),
    ),
    demo_replay_cfg=dict(
        type="ReplayMemory",
        capacity=-1,
        num_samples=-1,
        keys=["obs", "actions", "dones", "episode_dones"],
        buffer_filenames=[
            "PATH_TO_DEMO.h5",
        ],
    ),
)


train_cfg = dict(
    on_policy=True,
    total_steps=int(5e6),
    warm_steps=0,
    n_steps=int(2.5e4), # FrameMiners might need a bit larger n_steps (and replay_cfg.capacity) than regular PPO (not tested)
    n_updates=1,
    n_eval=int(5e6),
    n_checkpoint=int(1e6),
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


rollout_cfg = dict(
    type="Rollout",
    num_procs=5,
    with_info=True,
    multi_thread=False,
)


replay_cfg = dict(
    type="ReplayMemory",
    capacity=int(2.5e4),
    sampling_cfg=dict(type="OneStepTransition", with_replacement=False),
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
