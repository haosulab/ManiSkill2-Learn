from ..utils.meta import Registry, build_from_cfg


MPC = Registry("mpc")  # Model predictive control
MFRL = Registry("mfrl")  # Model free RL
BRL = Registry("brl")  # Offline RL / Batch RL


def build_agent(cfg, default_args=None):
    for agent_type in [MPC, MFRL, BRL]:
        if cfg["type"] in agent_type:
            return build_from_cfg(cfg, agent_type, default_args)
    return None
