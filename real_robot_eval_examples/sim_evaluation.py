import os
import os.path as osp
import numpy as np
import time

import argparse

import gym
from copy import deepcopy
from collections import OrderedDict

import torch

from maniskill2_learn.utils.meta import Config, DictAction, get_logger
from maniskill2_learn.networks.builder import build_model
from maniskill2_learn.networks.utils import (
    get_kwargs_from_shape,
    replace_placeholder_with_args,
)
from maniskill2_learn.utils.data import GDict
from .env_utils import build_env


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained agent")
    # Parameters for log dir
    parser.add_argument("--output-dir", required=True, help="output directory")
    parser.add_argument("--config", required=True, help="config file path")
    parser.add_argument("--model-path", required=True, help="checkpoint path")
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument(
        "--cfg-options",
        "--opt",
        nargs="+",
        action=DictAction,
        help="Override some settings in the configuration file. The key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overridden is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )

    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        # Merge cfg with args.cfg_options
        for key, value in args.cfg_options.items():
            try:
                value = eval(value)
                args.cfg_options[key] = value
            except:
                pass
        cfg.merge_from_dict(args.cfg_options)

    return args, cfg


if __name__ == "__main__":
    args, cfg = parse_args()
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        torch.set_num_threads(1)
        device = "cuda"
    else:
        device = "cpu"

    args.timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.output_dir = f"{args.output_dir}/{args.timestamp}"

    os.makedirs(osp.abspath(args.output_dir), exist_ok=True)

    logger = get_logger(
        name="SIM EVAL", log_file=osp.join(args.output_dir, f"{args.timestamp}.log")
    )

    logger.info(f"Args: \n{args}")
    logger.info(f"Config:\n{cfg.pretty_text}")

    # since we do evaluation, we use eval_env_cfg to override env_cfg
    env_cfg = cfg["env_cfg"]
    eval_env_cfg = cfg["eval_cfg"]["env_cfg"]
    if eval_env_cfg:
        for k, v in eval_env_cfg.items():
            env_cfg[k] = v

    env = build_env(env_cfg)

    obs = env.reset()
    viewer = env.render()

    logger.info(f"Observation space {env.observation_space}")
    logger.info(f"Action space {env.action_space}")
    logger.info(f"Control mode {env.control_mode}")
    logger.info(f"Reward mode {env.reward_mode}")

    actor_cfg = cfg.agent_cfg.actor_cfg
    actor_cfg.pop("optim_cfg")
    actor_cfg["obs_shape"] = GDict(obs).shape
    actor_cfg["action_shape"] = len(env.action_space.sample())
    actor_cfg["action_space"] = deepcopy(env.action_space)
    replaceable_kwargs = get_kwargs_from_shape(
        GDict(obs).list_shape, env.action_space.sample().shape[0]
    )
    cfg = replace_placeholder_with_args(cfg, **replaceable_kwargs)
    logger.info(f"Final actor config:\n{cfg.agent_cfg.actor_cfg}")
    actor = build_model(cfg.agent_cfg.actor_cfg).to(device)

    # load weight
    state_dict = torch.load(args.model_path)["state_dict"]
    logger.info(f"State dict keys in model checkpoint: {state_dict.keys()}")
    actor_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:6] == "actor.":
            actor_state_dict[k[6:]] = v

    assert (
        actor.state_dict().keys() == actor_state_dict.keys()
    ), f"Actor keys: {actor.state_dict().keys()}\nweight keys: {actor_state_dict.keys()}"
    actor.load_state_dict(actor_state_dict)

    # begin evaluation
    actor.eval()

    # logger.info("Press [s] to start")
    # while True:
    #     if viewer.window.key_down("s"):
    #         break
    #     env.render()

    done = False
    step = 0
    with torch.no_grad():
        while (not done) and step < 200:
            obs = (
                GDict(obs)
                .unsqueeze(0)
                .to_torch(dtype="float32", wrapper=False, device=device)
            )
            action = actor(obs)[0]
            action = action.cpu().numpy()
            obs, rew, done, info = env.step(action)
            env.render()
            logger.info(f"STEP: {step:5d}, rew: {rew}, done: {done}, info: {info}")
            step += 1
