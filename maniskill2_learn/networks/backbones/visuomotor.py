"""
End-to-End Training of Deep Visuomotor Policies
    https://arxiv.org/pdf/1504.00702.pdf
Visuomotor as the base class of all visual polices.
"""
import torch, torch.nn as nn, torch.nn.functional as F
from copy import copy, deepcopy
from maniskill2_learn.utils.meta import get_logger
from maniskill2_learn.utils.torch import (
    ExtendedModule,
    ExtendedModuleList,
    freeze_params,
    unfreeze_params,
)
from maniskill2_learn.utils.data import GDict, DictArray, recover_with_mask, is_seq_of
from .mlp import LinearMLP
from ..builder import build_model, BACKBONES

from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion


@BACKBONES.register_module()
class Visuomotor(ExtendedModule):
    def __init__(
        self,
        visual_nn_cfg,
        mlp_cfg,
        visual_nn=None,
        freeze_visual_nn=False,
        freeze_mlp=False,
    ):
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

            assert isinstance(
                self.final_mlp, LinearMLP
            ), "The final mlp should have type LinearMLP."
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
        assert not (
            feature is not None and visual_feature is not None
        ), f"You cannot provide visual_feature and feature at the same time!"
        self.saved_feature = None
        self.saved_visual_feature = None
        robot_state = None
        save_feature = save_feature or (
            feature is not None or visual_feature is not None
        )

        obs_keys = obs.keys()
        for key in ["state", "agent"]:
            if key in obs:
                assert (
                    robot_state is None
                ), f"Please provide only one robot state! Obs Keys: {obs_keys}"
                robot_state = obs.pop(key)
        if not ("xyz" in obs or "rgb" in obs or "rgbd" in obs):
            assert (
                len(obs) == 1
            ), f"Observations need to contain only one visual element! Obs Keys: {obs.keys()}!"
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
                assert (
                    feat.ndim == robot_state.ndim
                ), "Visual feature and state vector should have the same dimension!"
                feat = torch.cat([feat, robot_state], dim=-1)

            if save_feature:
                self.saved_feature = feat.clone()
        else:
            feat = feature

        if self.final_mlp is not None:
            feat = self.final_mlp(feat)

        return feat


@BACKBONES.register_module()
class FrameMiners(ExtendedModule):
    def __init__(
        self,
        visual_nn_cfg,
        num_frames,
        vis_feat_dim,
        action_dim,
        robot_state_dim,
        is_critic=False,
        critic_mode="V",
        visual_nn=None,
        **kwargs,
    ):
        super(FrameMiners, self).__init__()
        self.num_frames = num_frames
        self.vis_feat_dim = vis_feat_dim
        self.is_critic = is_critic
        self.visual_nn = visual_nn or ExtendedModuleList(
            [build_model(visual_nn_cfg) for i in range(num_frames)]
        )
        if is_critic:
            assert critic_mode in ["Q", "V"]
            critic_branch_input_dim = (vis_feat_dim + robot_state_dim) * num_frames
            if critic_mode == "Q":
                critic_branch_input_dim += action_dim
            self.final_mlp = LinearMLP(
                mlp_spec=[critic_branch_input_dim, 192, 128, 1],
                norm_cfg=None,
                inactivated_output=True,
                zero_init_output=True,  # zero_init_output is typically crucial for Point Cloud-based RL
            )
        else:
            self.final_mlp = ExtendedModuleList(
                [
                    LinearMLP(
                        mlp_spec=[vis_feat_dim + robot_state_dim, 192, 128, action_dim],
                        norm_cfg=None,
                        inactivated_output=True,
                        zero_init_output=True,
                    )
                    for i in range(num_frames)
                ]
            )
            self.fused_weight = LinearMLP(
                mlp_spec=[
                    (vis_feat_dim + robot_state_dim) * num_frames,
                    192,
                    action_dim * num_frames,
                ],
                norm_cfg=None,
                inactivated_output=True,
                zero_init_output=True,
            )
        self.saved_feature = None
        self.saved_visual_feature = None

    def forward(
        self,
        obs,
        feature=None,
        visual_feature=None,
        save_feature=False,
        **kwargs,
    ):
        obs = copy(obs)
        assert isinstance(obs, dict), f"obs is not a dict! {type(obs)}"
        obs_keys = obs.keys()
        robot_state = None
        for key in obs_keys:
            if (
                "_box" in key
                or "_seg" in key
                or "_sem_label" in key
                or key == "visual_state"
            ):
                obs.pop(key)

        to_frames = obs.pop("to_frames")  # [B, Nframe, 4, 4]
        assert self.num_frames <= to_frames.shape[1]
        frame_related_states = obs.pop("frame_related_states")  # [B, K, 3]
        frame_goal_related_poses = obs.pop(
            "frame_goal_related_poses", None
        )  # [B, K', 7]
        num = frame_related_states.shape[-1] * frame_related_states.shape[-2]
        if frame_goal_related_poses is not None:
            num = (
                num
                + frame_goal_related_poses.shape[-1]
                * frame_goal_related_poses.shape[-2]
            )
        agent_state = obs.pop("state")[..., :-num]
        B = agent_state.shape[0]

        def batch_transform_pos(frame, points):
            # frame: [B, 4, 4]; points: [B, K, 3]
            return (
                torch.einsum("bij,bnj->bni", frame[..., :3, :3], points)
                + frame[..., None, :3, 3]
            )  # [B, K, 3]

        def batch_transform_pose(frame, poses):
            # frame: [B, 4, 4]; poses: [B, K', 7]
            ret_pos = (
                torch.einsum("bij,bnj->bni", frame[..., :3, :3], poses[..., :, :3])
                + frame[..., None, :3, 3]
            )  # [B, K', 3]
            ret_pose = torch.einsum(
                "bij,bnjk->bnik",
                frame[..., :3, :3],
                quaternion_to_matrix(poses[..., :, 3:]),
            )  # [B, K', 3, 3]
            ret_pose = matrix_to_quaternion(ret_pose)  # [B, K', 4]
            return torch.cat([ret_pos, ret_pose], dim=-1)  # [B, K', 7]

        if feature is None:
            feats = []
            for i in range(self.num_frames):
                to_frame = to_frames[:, i]  # [B, 4, 4]
                pcd = {
                    key: obs[key] for key in ["xyz", "rgb", "seg"] if key in obs
                }  # [B, N, 3]
                pcd["xyz"] = batch_transform_pos(to_frame, pcd["xyz"])
                frs_i_pos = batch_transform_pos(to_frame, frame_related_states).reshape(
                    B, -1
                )  # [B, K * 3]
                if frame_goal_related_poses is not None:
                    frs_i_pose = batch_transform_pose(
                        to_frame, frame_goal_related_poses
                    ).reshape(B, -1)
                    frs_i = torch.cat([frs_i_pos, frs_i_pose], dim=-1)
                else:
                    frs_i = frs_i_pos
                vis_feat = self.visual_nn[i](pcd)  # [B, vis_feat_dim]
                feat = torch.cat([vis_feat, agent_state, frs_i], dim=-1)
                feats.append(feat)
        else:
            feat_dim = feature.shape[-1] // self.num_frames
            feats = [
                feature[..., i * feat_dim : (i + 1) * feat_dim]
                for i in range(self.num_frames)
            ]

        if save_feature:
            self.saved_feature = torch.cat(feats, dim=-1)

        if self.is_critic:
            global_feat = torch.cat(feats, dim=-1)
            ret = self.final_mlp(global_feat)
        else:
            actions = [self.final_mlp[i](feats[i]) for i in range(self.num_frames)]
            weight = self.fused_weight(torch.cat(feats, dim=-1)).reshape(
                [B, self.num_frames, -1]
            )  # [B, NF, NA]
            weight = weight.softmax(dim=1)
            actions = torch.stack(actions, dim=-2)  # [B, NF, NA]
            ret = (weight * actions).sum(1)
        return ret
