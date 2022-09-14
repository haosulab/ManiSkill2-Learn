import gym

import mani_skill2.envs

from maniskill2_learn.utils.meta import Config
from maniskill2_learn.env.wrappers import (
    ManiSkill2_ObsWrapper,
)  # , ManiSkill2_VisualAugWrapper


def build_env(env_cfg: Config):
    env_name = env_cfg.env_name
    obs_mode = env_cfg.get("obs_mode", "rgbd")
    control_mode = env_cfg.get("control_mode", "pd_joint_delta_pos")
    img_size = env_cfg.get("img_size", None)
    n_points = env_cfg.get("n_points", 1200)
    n_goal_points = env_cfg.get("n_goal_points", -1)
    obs_frame = env_cfg.get("obs_frame", "world")
    ignore_dones = env_cfg.get("ignore_dones", False)

    rgbd_aug = env_cfg.get("rgbd_aug", False)
    color_bright = env_cfg.get("color_bright", 0.0)
    color_contrast = env_cfg.get("color_contrast", 0.0)
    color_saturation = env_cfg.get("color_saturation", 0.0)
    color_hue = env_cfg.get("color_hue", 0.0)
    color_drop_prob = env_cfg.get("color_drop_prob", 0.0)
    depth_noise_scale = env_cfg.get("depth_noise_scale", 0.0)
    depth_salt = env_cfg.get("depth_salt", 0.0)
    env = gym.make(
        env_name, obs_mode=obs_mode, control_mode=control_mode
    )  # real env to be tested
    env = ManiSkill2_ObsWrapper(
        env,
        img_size=img_size,
        n_points=n_points,
        n_goal_points=n_goal_points,
        obs_frame=obs_frame,
        ignore_dones=ignore_dones,
    )
    # env = ManiSkill2_VisualAugWrapper(env, rgbd_aug, color_bright, color_contrast, color_saturation, color_hue,
    #                                   color_drop_prob, depth_noise_scale, depth_salt)

    return env
