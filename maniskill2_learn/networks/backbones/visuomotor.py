"""
End-to-End Training of Deep Visuomotor Policies
    https://arxiv.org/pdf/1504.00702.pdf
Visuomotor as the base class of all visual polices.
"""
import torch, torch.nn as nn, torch.nn.functional as F
from copy import copy, deepcopy
from maniskill2_learn.utils.meta import get_logger
from maniskill2_learn.utils.torch import ExtendedModule, ExtendedModuleList, freeze_params, unfreeze_params
from maniskill2_learn.utils.data import GDict, DictArray, recover_with_mask, is_seq_of
from .mlp import LinearMLP
from ..builder import build_model, BACKBONES


@BACKBONES.register_module()
class Visuomotor(ExtendedModule):
    def __init__(self, visual_nn_cfg, mlp_cfg, 
                visual_nn=None, 
                freeze_visual_nn=False, freeze_mlp=False):
        super(Visuomotor, self).__init__()
        # Feature extractor [Can be shared with other network]
        self.visual_nn = build_model(visual_nn_cfg) if visual_nn is None else visual_nn
        self.final_mlp = build_model(mlp_cfg)

        if freeze_visual_nn:
            get_logger().warning("We freeze the visual backbone!")
            freeze_params(self.visual_nn)

        if freeze_mlp:
            get_logger().warning("We freeze the whole mlp part!")
            from .mlp import LinearMLP

            assert isinstance(self.final_mlp, LinearMLP), "The final mlp should have type LinearMLP."
            freeze_params(self.final_mlp)

        self.saved_feature = None
        self.saved_visual_feature = None

    def forward(
        self,
        obs,
        feature=None,
        visual_feature=None,
        save_feature=False,
        detach_visual=False,
        episode_dones=None,
        is_valid=None,
        with_robot_state=True,
        **kwargs,
    ):
        obs = copy(obs)
        assert isinstance(obs, dict), f"obs is not a dict! {type(obs)}"
        assert not (feature is not None and visual_feature is not None), f"You cannot provide visual_feature and feature at the same time!"
        self.saved_feature = None
        self.saved_visual_feature = None
        robot_state = None
        save_feature = save_feature or (feature is not None or visual_feature is not None)

        obs_keys = obs.keys()
        for key in ["state", "agent"]:
            if key in obs:
                assert robot_state is None, f"Please provide only one robot state! Obs Keys: {obs_keys}"
                robot_state = obs.pop(key)
        if not ("xyz" in obs or "rgb" in obs or "rgbd" in obs):
            assert len(obs) == 1, f"Observations need to contain only one visual element! Obs Keys: {obs.keys()}!"
            obs = obs[list(obs.keys())[0]]

        if feature is None:
            if visual_feature is None:
                feat = self.visual_nn(obs)
                if detach_visual:
                    feat = feat.detach()
            else:
                feat = visual_feature

            if save_feature:
                self.saved_visual_feature = feat.clone()

            if robot_state is not None and with_robot_state:
                assert feat.ndim == robot_state.ndim, "Visual feature and state vector should have the same dimension!"
                feat = torch.cat([feat, robot_state], dim=-1)

            if save_feature:
                self.saved_feature = feat.clone()
        else:
            feat = feature

        if self.final_mlp is not None:
            feat = self.final_mlp(feat)

        return feat