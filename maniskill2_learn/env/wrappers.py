from collections import deque
import cv2
import numpy as np

from gym import spaces
from gym.core import ObservationWrapper, Wrapper
from gym.spaces import Discrete

from maniskill2_learn.utils.data import (
    DictArray,
    GDict,
    deepcopy,
    encode_np,
    is_num,
    to_array,
    SLICE_ALL,
    to_np,
)
from maniskill2_learn.utils.meta import Registry, build_from_cfg

from .observation_process import pcd_uniform_downsample

WRAPPERS = Registry("wrappers of gym environments")


class ExtendedWrapper(Wrapper):
    def __getattr__(self, name):
        # gym standard do not support name with '_'
        return getattr(self.env, name)


class BufferAugmentedEnv(ExtendedWrapper):
    """
    For multi-process environments.
    Use a buffer to transfer data from sub-process to main process!
    """

    def __init__(self, env, buffers):
        super(BufferAugmentedEnv, self).__init__(env)
        self.reset_buffer = GDict(buffers[0])
        self.step_buffer = GDict(buffers[:4])
        if len(buffers) == 5:
            self.vis_img_buffer = GDict(buffers[4])

    def reset(self, *args, **kwargs):
        self.reset_buffer.assign_all(self.env.reset(*args, **kwargs))

    def step(self, *args, **kwargs):
        alls = self.env.step(*args, **kwargs)
        self.step_buffer.assign_all(alls)

    def render(self, *args, **kwargs):
        ret = self.env.render(*args, **kwargs)
        if ret is not None:
            assert (
                self.vis_img_buffer is not None
            ), "You need to provide vis_img_buffer!"
            self.vis_img_buffer.assign_all(ret)


class ExtendedEnv(ExtendedWrapper):
    """
    Extended api for all environments, which should be also supported by VectorEnv.

    Supported extra attributes:
    1. is_discrete, is_cost, reward_scale

    Function changes:
    1. step: reward multiplied by a scale, convert all f64 to to_f32
    2. reset: convert all f64 to to_f32

    Supported extra functions:
    2. step_random_actions
    3. step states_actions
    """

    def __init__(self, env, reward_scale, use_cost):
        super(ExtendedEnv, self).__init__(env)
        assert reward_scale > 0, "Reward scale should be positive!"
        self.is_discrete = isinstance(env.action_space, Discrete)
        self.is_cost = -1 if use_cost else 1
        self.reward_scale = reward_scale * self.is_cost

    def _process_action(self, action):
        if self.is_discrete:
            if is_num(action):
                action = int(action)
            else:
                assert (
                    action.size == 1
                ), f"Dim of discrete action should be 1, but we get {len(action)}"
                action = int(action.reshape(-1)[0])
        return action

    def reset(self, *args, **kwargs):
        kwargs = dict(kwargs)
        obs = self.env.reset(*args, **kwargs)
        return GDict(obs).f64_to_f32(wrapper=False)

    def step(self, action, *args, **kwargs):
        action = self._process_action(action)
        obs, reward, done, info = self.env.step(action, *args, **kwargs)
        if isinstance(info, dict) and "TimeLimit.truncated" not in info:
            info["TimeLimit.truncated"] = False
        obs, info = GDict([obs, info]).f64_to_f32(wrapper=False)
        return obs, np.float32(reward * self.reward_scale), np.bool_(done), info

    # The following three functions are available for VectorEnv too!
    def step_random_actions(self, num):
        from .env_utils import true_done

        # print("-----------------------------------------------")
        ret = None
        # import os
        # print(os.getpid(), obs)
        obs = GDict(self.reset()).copy(wrapper=False)

        # print(os.getpid(), self.env.level, obs)
        # print("-----------------------------------------------")
        # exit(0)

        for i in range(num):
            action = self.action_space.sample()
            next_obs, rewards, dones, infos = self.step(action)
            next_obs = GDict(next_obs).copy(wrapper=False)

            info_i = dict(
                obs=obs,
                next_obs=next_obs,
                actions=action,
                rewards=rewards,
                dones=true_done(dones, infos),
                infos=GDict(infos).copy(wrapper=False),
                episode_dones=dones,
            )
            info_i = GDict(info_i).to_array(wrapper=False)
            obs = GDict(next_obs).copy(wrapper=False)

            if ret is None:
                ret = DictArray(info_i, capacity=num)
            ret.assign(i, info_i)
            if dones:
                obs = GDict(self.reset()).copy(wrapper=False)
        return ret.to_two_dims(wrapper=False)

    def step_states_actions(self, states=None, actions=None):
        """
        For CEM only
        states: [N, NS]
        actions: [N, L, NA]
        return [N, L, 1]
        """
        assert actions.ndim == 3
        rewards = np.zeros_like(actions[..., :1], dtype=np.float32)
        for i in range(len(actions)):
            if hasattr(self, "set_state") and states is not None:
                self.set_state(states[i])
            for j in range(len(actions[i])):
                rewards[i, j] = self.step(actions[i, j])[1]
        return rewards

    def get_env_state(self):
        ret = {}
        if hasattr(self.env, "get_state"):
            ret["env_states"] = self.env.get_state()
        # if hasattr(self.env.unwrapped, "_scene") and save_scene_state:
        #     ret["env_scene_states"] = self.env.unwrapped._scene.pack()
        if hasattr(self.env, "level"):
            ret["env_levels"] = self.env.level
        return ret


@WRAPPERS.register_module()
class FixedInitWrapper(ExtendedWrapper):
    def __init__(self, env, init_state, level=None, *args, **kwargs):
        super(FixedInitWrapper, self).__init__(env)
        self.init_state = np.array(init_state)
        self.level = level

    def reset(self, *args, **kwargs):
        if self.level is not None:
            # For ManiSkill
            self.env.reset(level=self.level)
        else:
            self.env.reset()
        self.set_state(self.init_state)
        return self.env.get_obs()


class ManiSkill2_ObsWrapper(ExtendedWrapper, ObservationWrapper):
    def __init__(
        self,
        env,
        img_size=None,
        n_points=1200,
        n_goal_points=-1,
        obs_frame="base",
        ignore_dones=False,
        fix_seed=None,
    ):
        super().__init__(env)
        
        self.ms2_env_name = self.env.unwrapped.spec.id
        self.obs_frame = obs_frame
        if self.obs_mode == "state":
            pass
        elif self.obs_mode == "rgbd":
            self.img_size = img_size
        elif self.obs_mode == "pointcloud":
            self.n_points = n_points
            self.n_goal_points = n_goal_points
        elif self.obs_mode == "particles":
            obs_space = env.observation_space

        self.ignore_dones = ignore_dones

        self.fix_seed = fix_seed

    def reset(self, **kwargs):
        if self.fix_seed is not None:
            obs = self.env.reset(seed=self.fix_seed, **kwargs)
        else:
            obs = self.env.reset(**kwargs)
        return self.observation(obs)

    def step(self, action):
        next_obs, reward, done, info = super(ManiSkill2_ObsWrapper, self).step(action)
        if self.ignore_dones:
            done = False
        return next_obs, reward, done, info

    def get_obs(self):
        return self.observation(self.env.observation(self.env.unwrapped.get_obs()))

    def observation(self, observation):
        from mani_skill2.utils.common import flatten_state_dict
        from maniskill2_learn.utils.lib3d.mani_skill2_contrib import (
            apply_pose_to_points,
            apply_pose_to_point,
        )
        from mani_skill2.utils.sapien_utils import vectorize_pose
        from sapien.core import Pose
        
        
        if self.obs_mode == "state":
            return observation

        # print(GDict(observation).shape)
        # exit(0)

        # Note that rgb information returned from the environment must have range [0, 255]
        
        if 'OpenCabinet' in self.ms2_env_name or 'PushChair' in self.ms2_env_name or 'MoveBucket' in self.ms2_env_name:
            # For envs migrated from ManiSkill1, we need to manually calculate the robot pose and the end-effector pose(s)
            robot_base_link = None
            hand_tcp_links = []
            for rob_link in self.env.unwrapped.agent.robot.get_links():
                if rob_link.name == 'mobile_base':
                    robot_base_link = rob_link
                if 'hand_tcp' in rob_link.name:
                    hand_tcp_links.append(rob_link)
            observation["agent"]["base_pose"] = vectorize_pose(robot_base_link.get_pose()) # [7,]
            if len(hand_tcp_links) == 1:
                observation["extra"]["tcp_pose"] = vectorize_pose(hand_tcp_links[0].get_pose()) # [7,]
            else:
                assert len(hand_tcp_links) > 1
                observation["extra"]["tcp_pose"] = np.stack([vectorize_pose(l.get_pose()) for l in hand_tcp_links], axis=0) # [nhands, 7], multi-arm envs

        if self.obs_mode == "rgbd":
            """
            Example *input* observation keys and their respective shapes ('extra' keys don't necessarily match):
            {'image':
                {'hand_camera':
                    {'rgb': (128, 128, 3), 'depth': (128, 128, 1)},
                 'base_camera':
                    {'rgb': (128, 128, 3), 'depth': (128, 128, 1)}
                },
             'agent':
                {'qpos': 9, 'qvel': 9, 'controller': {'arm': {}, 'gripper': {}}, 'base_pose': 7},
             'camera_param':
                {'base_camera': {'extrinsic_cv': (4, 4), 'cam2world_gl': (4, 4), 'intrinsic_cv': (3, 3)}, 
                'hand_camera': {'extrinsic_cv': (4, 4), 'cam2world_gl': (4, 4), 'intrinsic_cv': (3, 3)}}
             'extra':
                {'tcp_pose': 7, 'goal_pos': 3}}
            """

            obs = observation
            rgb, depth, segs = [], [], []
            imgs = obs["image"]
            
            # IMPORTANT: the order of cameras can be different across different maniskill2 versions; 
            # thus we have to explicitly list out camera names to ensure that camera orders are consistent
            if 'hand_camera' in imgs.keys():
                cam_names = ['hand_camera', 'base_camera']
            elif 'overhead_camera_0' in imgs.keys(): # ManiSkill1 environments
                cam_names = ['overhead_camera_0', 'overhead_camera_1', 'overhead_camera_2']
            else:
                raise NotImplementedError()
            
            # Process RGB and Depth images
            for cam_name in cam_names: 
                rgb.append(imgs[cam_name]["rgb"])  # each [H, W, 3]
                depth.append(imgs[cam_name]["depth"])  # each [H, W, 1]
                if "Segmentation" in imgs[cam_name].keys():
                    segs.append(imgs[cam_name]["Segmentation"]) # each [H, W, 4], last dim = [mesh_seg, actor_seg, 0, 0]
            rgb = np.concatenate(rgb, axis=2)
            assert rgb.dtype == np.uint8
            depth = np.concatenate(depth, axis=2)
            depth = depth.astype(np.float32, copy=False)
            if len(segs) > 0:
                segs = np.concatenate(segs, axis=2)
            obs.pop("image")

            # Reshape goal images, if any, for environments that use goal image, e.g. Writer-v0, Pinch-v0
            def process_4d_goal_img_to_3d(goal_img):
                if goal_img.ndim == 4:  # [K, H, W, C]
                    # for Pinch-v0, where there are multiple views of the goal
                    goal_img = np.transpose(goal_img, (1, 2, 0, 3))
                    H, W = goal_img.shape[:2]
                    goal_img = goal_img.reshape([H, W, -1])
                return goal_img
            goal_rgb = obs["extra"].pop("goal", None)
            if goal_rgb is None:
                goal_rgb = obs["extra"].pop("target_rgb", None)
            if goal_rgb is not None:
                assert goal_rgb.dtype == np.uint8
                goal_rgb = process_4d_goal_img_to_3d(goal_rgb)
                goal_rgb = cv2.resize(
                    goal_rgb.astype(np.float32),
                    rgb.shape[:2],
                    interpolation=cv2.INTER_LINEAR,
                )
                goal_rgb = goal_rgb.astype(np.uint8)
                if goal_rgb.ndim == 2:  # [H, W]
                    goal_rgb = goal_rgb[:, :, None]
                rgb = np.concatenate([rgb, goal_rgb], axis=2)
            goal_depth = obs["extra"].pop("target_depth", None)
            if goal_depth is not None:
                goal_depth = process_4d_goal_img_to_3d(goal_depth)
                goal_depth = cv2.resize(
                    goal_depth.astype(np.float32, copy=False),
                    depth.shape[:2],
                    interpolation=cv2.INTER_LINEAR,
                )
                if goal_depth.ndim == 2:
                    goal_depth = goal_depth[:, :, None]
                depth = np.concatenate([depth, goal_depth], axis=2)

            # If goal info is provided, calculate the relative position between the robot fingers' tool-center-point (tcp) and the goal
            if "tcp_pose" in obs["extra"].keys() and "goal_pos" in obs["extra"].keys():
                assert obs["extra"]["tcp_pose"].ndim <= 2
                if obs["extra"]["tcp_pose"].ndim == 2:
                    tcp_pose = obs["extra"]["tcp_pose"][0] # take the first hand's tcp pose
                else:
                    tcp_pose = obs["extra"]["tcp_pose"]
                obs["extra"]["tcp_to_goal_pos"] = (
                    obs["extra"]["goal_pos"] - tcp_pose[:3]
                )
            if "tcp_pose" in obs["extra"].keys():
                obs["extra"]["tcp_pose"] = obs["extra"]["tcp_pose"].reshape(-1)
            
            obs['extra'].pop('target_points', None)
            obs.pop('camera_param', None)
            
            s = flatten_state_dict(obs) # Other observation keys should be already ordered and such orders shouldn't change across different maniskill2 versions, so we just flatten them

            # Resize RGB and Depth images
            if self.img_size is not None and self.img_size != (
                rgb.shape[0],
                rgb.shape[1],
            ):
                rgb = cv2.resize(
                    rgb.astype(np.float32),
                    self.img_size,
                    interpolation=cv2.INTER_LINEAR,
                )
                depth = cv2.resize(depth, self.img_size, interpolation=cv2.INTER_LINEAR)

            # compress rgb & depth for e.g., trajectory saving purposes
            out_dict = {
                "rgb": rgb.astype(np.uint8, copy=False).transpose(2, 0, 1), # [C, H, W]
                "depth": depth.astype(np.float16, copy=False).transpose(2, 0, 1),
                "state": s,
            }
            if len(segs) > 0:
                out_dict["segs"] = segs.transpose(2, 0, 1)

            return out_dict

        elif self.obs_mode == "pointcloud":
            """
            Example observation keys and respective shapes ('extra' keys don't necessarily match):
            {'pointcloud':
                {'xyz': (32768, 3), 'rgb': (32768, 3)},
                # 'xyz' can also be 'xyzw' with shape (N, 4),
                # where the last dim indicates whether the point is inside the camera depth range
             'agent':
                {'qpos': 9, 'qvel': 9, 'controller': {'arm': {}, 'gripper': {}}, 'base_pose': 7},
             'extra':
                {'tcp_pose': 7, 'goal_pos': 3}
            }
            """
            # Calculate coordinate transformations that transforms poses in the world to self.obs_frame
            # These "to_origin" coordinate transformations are formally T_{self.obs_frame -> world}^{self.obs_frame}
            if self.obs_frame in ["base", "world"]:
                base_pose = observation["agent"]["base_pose"]
                p, q = base_pose[:3], base_pose[3:]
                to_origin = Pose(p=p, q=q).inv()
            elif self.obs_frame == "ee":
                tcp_poses = observation["extra"]["tcp_pose"]
                assert tcp_poses.ndim <= 2
                if tcp_poses.ndim == 2:
                    tcp_pose = tcp_poses[0] # use the first robot hand's tcp pose as the end-effector frame
                else:
                    tcp_pose = tcp_poses # only one robot hand
                p, q = tcp_pose[:3], tcp_pose[3:]
                to_origin = Pose(p=p, q=q).inv()
            else:
                print("Unknown Frame", self.obs_frame)
                exit(0)

            # Unify the xyz and the xyzw point cloud format
            pointcloud = observation["pointcloud"].copy()
            xyzw = pointcloud.pop("xyzw", None)
            if xyzw is not None:
                assert "xyz" not in pointcloud.keys()
                mask = xyzw[:, -1] > 0.5
                xyz = xyzw[:, :-1]
                for k in pointcloud.keys():
                    pointcloud[k] = pointcloud[k][mask]
                pointcloud["xyz"] = xyz[mask]
                
            # Initialize return dict
            ret = {
                mode: pointcloud[mode]
                for mode in ["xyz", "rgb"]
                if mode in pointcloud
            }
            
            # Process observation point cloud segmentations, if given
            if "visual_seg" in pointcloud and "actor_seg" in pointcloud:
                visual_seg = pointcloud["visual_seg"].squeeze()
                actor_seg = pointcloud["actor_seg"].squeeze()
                assert visual_seg.ndim == 1 and actor_seg.ndim == 1
                N = visual_seg.shape[0]
                ret_visual_seg = np.zeros([N, 50]) # hardcoded
                ret_visual_seg[np.arange(N), visual_seg] = 1.0
                ret_actor_seg = np.zeros([N, 50]) # hardcoded
                ret_actor_seg[np.arange(N), actor_seg] = 1.0
                ret["seg"] = np.concatenate([ret_visual_seg, ret_actor_seg], axis=-1)

            # Process observation point cloud rgb, downsample observation point cloud, and transform observation point cloud coordinates to self.obs_frame
            ret["rgb"] = ret["rgb"] / 255.0
            uniform_downsample_kwargs = {"env": self.env, "ground_eps": 1e-4, "num": self.n_points}
            if "PointCloudPreprocessObsWrapper" not in self.env.__str__():
                pcd_uniform_downsample(
                    ret, **uniform_downsample_kwargs
                )
            ret["xyz"] = apply_pose_to_points(ret["xyz"], to_origin)

            # Sample from and append the goal point cloud to the observation point cloud, if the goal point cloud is given
            goal_pcd_xyz = observation.pop("target_points", None)
            if goal_pcd_xyz is not None:
                ret_goal = {}
                ret_goal["xyz"] = goal_pcd_xyz
                for k in ret.keys():
                    if k != "xyz":
                        ret_goal[k] = np.ones_like(ret[k]) * (-1) # special value to distinguish goal point cloud and observation point cloud
                pcd_uniform_downsample(ret_goal, **uniform_downsample_kwargs)
                ret_goal["xyz"] = apply_pose_to_points(ret_goal["xyz"], to_origin)
                for k in ret.keys():
                    ret[k] = np.concatenate([ret[k], ret_goal[k]], axis=0)

            # Get all kinds of position (pos) and 6D poses (pose) from the observation information
            # These pos & poses are in world frame for now (transformed later)
            obs_extra_keys = observation["extra"].keys()
            tcp_poses = None
            if "tcp_pose" in obs_extra_keys:
                tcp_poses = observation["extra"]["tcp_pose"]
                assert tcp_poses.ndim <= 2
                if tcp_poses.ndim == 1: # single robot hand
                    tcp_poses = tcp_poses[None, :]
                tcp_poses = [Pose(p=pose[:3], q=pose[3:]) for pose in tcp_poses] # [nhand] tcp poses, where nhand is the number of robot hands
            goal_pos = None
            goal_pose = None
            if "goal_pos" in obs_extra_keys:
                goal_pos = observation["extra"]["goal_pos"]
            elif "goal_pose" in obs_extra_keys:
                goal_pos = observation["extra"]["goal_pose"][:3]
                goal_pose = observation["extra"]["goal_pose"]
                goal_pose = Pose(p=goal_pose[:3], q=goal_pose[3:])
            tcp_to_goal_pos = None
            if tcp_poses is not None and goal_pos is not None:
                tcp_to_goal_pos = goal_pos - tcp_poses[0].p # use the first robot hand's tcp pose to calculate the relative position from tcp to goal

            # Sample green points near the goal and append them to the observation point cloud, which serve as visual goal indicator,
            # if self.n_goal_points is specified and the goal information if given in an environment
            # Also, transform these points to self.obs_frame
            if self.n_goal_points > 0:
                assert (
                    goal_pos is not None
                ), "n_goal_points should only be used if goal_pos(e) is contained in the environment observation"
                goal_pts_xyz = (
                    np.random.uniform(low=-1.0, high=1.0, size=(self.n_goal_points, 3))
                    * 0.01
                )
                goal_pts_xyz = goal_pts_xyz + goal_pos[None, :]
                goal_pts_xyz = apply_pose_to_points(goal_pts_xyz, to_origin)
                goal_pts_rgb = np.zeros_like(goal_pts_xyz)
                goal_pts_rgb[:, 1] = 1
                ret["xyz"] = np.concatenate([ret["xyz"], goal_pts_xyz])
                ret["rgb"] = np.concatenate([ret["rgb"], goal_pts_rgb])

            # Transform all kinds of positions to self.obs_frame; these information are dependent on
            # the choice of self.obs_frame, so we name them "frame_related_states"
            frame_related_states = []
            base_info = apply_pose_to_point(
                observation["agent"]["base_pose"][:3], to_origin
            )
            frame_related_states.append(base_info)
            if tcp_poses is not None:
                for tcp_pose in tcp_poses:
                    tcp_info = apply_pose_to_point(tcp_pose.p, to_origin)
                    frame_related_states.append(tcp_info)
            if goal_pos is not None:
                goal_info = apply_pose_to_point(goal_pos, to_origin)
                frame_related_states.append(goal_info)
            if tcp_to_goal_pos is not None:
                tcp_to_goal_info = apply_pose_to_point(tcp_to_goal_pos, to_origin)
                frame_related_states.append(tcp_to_goal_info)
            if "gripper_pose" in obs_extra_keys:
                gripper_info = observation["extra"]["gripper_pose"][:3]
                gripper_info = apply_pose_to_point(gripper_info, to_origin)
                frame_related_states.append(gripper_info)
            if "joint_axis" in obs_extra_keys:  # for TurnFaucet
                joint_axis_info = (
                    to_origin.to_transformation_matrix()[:3, :3]
                    @ observation["extra"]["joint_axis"]
                )
                frame_related_states.append(joint_axis_info)
            if "link_pos" in obs_extra_keys:  # for TurnFaucet
                link_pos_info = apply_pose_to_point(
                    observation["extra"]["link_pos"], to_origin
                )
                frame_related_states.append(link_pos_info)
            frame_related_states = np.stack(frame_related_states, axis=0)
            ret["frame_related_states"] = frame_related_states

            # Transform the goal pose and the pose from the end-effector (tool-center point, tcp)
            # to the goal into self.obs_frame; these info are also dependent on the choice of self.obs_frame,
            # so we name them "frame_goal_related_poses"
            frame_goal_related_poses = []
            if goal_pose is not None:
                pose_wrt_origin = to_origin * goal_pose
                frame_goal_related_poses.append(
                    np.hstack([pose_wrt_origin.p, pose_wrt_origin.q])
                )
                if tcp_poses is not None:
                    for tcp_pose in tcp_poses:
                        pose_wrt_origin = (
                            goal_pose * tcp_pose.inv()
                        )  # T_{tcp->goal}^{world}
                        pose_wrt_origin = to_origin * pose_wrt_origin
                        frame_goal_related_poses.append(
                            np.hstack([pose_wrt_origin.p, pose_wrt_origin.q])
                        )
            if len(frame_goal_related_poses) > 0:
                frame_goal_related_poses = np.stack(frame_goal_related_poses, axis=0)
                ret["frame_goal_related_poses"] = frame_goal_related_poses

            # ret['to_frames'] returns frame transformation information, which is information that transforms
            # from self.obs_frame to other common frames (e.g. robot base frame, end-effector frame, goal frame)
            # Each transformation is formally T_{target_frame -> self.obs_frame}^{target_frame}
            ret["to_frames"] = []
            base_pose = observation["agent"]["base_pose"]
            base_pose_p, base_pose_q = base_pose[:3], base_pose[3:]
            base_frame = (
                (to_origin * Pose(p=base_pose_p, q=base_pose_q))
                .inv()
                .to_transformation_matrix()
            )
            ret["to_frames"].append(base_frame)
            if tcp_poses is not None:
                for tcp_pose in tcp_poses:
                    hand_frame = (to_origin * tcp_pose).inv().to_transformation_matrix()
                    ret["to_frames"].append(hand_frame)
            if goal_pose is not None:
                goal_frame = (to_origin * goal_pose).inv().to_transformation_matrix()
                ret["to_frames"].append(goal_frame)
            ret["to_frames"] = np.stack(ret["to_frames"], axis=0)  # [Nframe, 4, 4]

            # Obtain final agent state vector, which contains robot proprioceptive information, frame-related states,
            # and other miscellaneous states (probably important) from the environment
            agent_state = np.concatenate(
                [observation["agent"]["qpos"], observation["agent"]["qvel"]]
            )
            if len(frame_related_states) > 0:
                agent_state = np.concatenate(
                    [agent_state, frame_related_states.flatten()]
                )
            if len(frame_goal_related_poses) > 0:
                agent_state = np.concatenate(
                    [agent_state, frame_goal_related_poses.flatten()]
                )
            for k in obs_extra_keys:
                if k not in [
                    "tcp_pose",
                    "goal_pos",
                    "goal_pose",
                    "tcp_to_goal_pos",
                    "tcp_to_goal_pose",
                    "joint_axis",
                    "link_pos",
                ]:
                    val = observation["extra"][k]
                    agent_state = np.concatenate(
                        [
                            agent_state,
                            val.flatten()
                            if isinstance(val, np.ndarray)
                            else np.array([val]),
                        ]
                    )

            ret["state"] = agent_state
            return ret

        elif self.obs_mode == "particles" and "particles" in observation.keys():
            obs = observation
            xyz = obs["particles"]["x"]
            vel = obs["particles"]["v"]
            state = flatten_state_dict(obs["agent"])
            ret = {
                "xyz": xyz,
                "state": state,
            }
            return ret
        else:
            return observation

    @property
    def _max_episode_steps(self):
        return self.env.unwrapped._max_episode_steps

    def render(self, mode="human", *args, **kwargs):
        if mode == "human":
            self.env.render(mode, *args, **kwargs)
            return

        if mode in ["rgb_array", "color_image"]:
            img = self.env.render(mode="rgb_array", *args, **kwargs)
        else:
            img = self.env.render(mode=mode, *args, **kwargs)
        if isinstance(img, dict):
            if "world" in img:
                img = img["world"]
            elif "main" in img:
                img = img["main"]
            else:
                print(img.keys())
                exit(0)
        if isinstance(img, dict):
            img = img["rgb"]
        if img.ndim == 4:
            assert img.shape[0] == 1
            img = img[0]
        if img.dtype in [np.float32, np.float64]:
            img = np.clip(img, a_min=0, a_max=1) * 255
        img = img[..., :3]
        img = img.astype(np.uint8)
        return img


class RenderInfoWrapper(ExtendedWrapper):
    def step(self, action):
        obs, rew, done, info = super().step(action)
        info["reward"] = rew
        self._info_for_render = info
        return obs, rew, done, info

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        # self._info_for_render = self.env.get_info()
        self._info_for_render = {}
        return obs

    def render(self, mode, **kwargs):
        from maniskill2_learn.utils.image.misc import put_info_on_image

        if mode == "rgb_array" or mode == "cameras":
            img = super().render(mode=mode, **kwargs)
            return put_info_on_image(
                img, self._info_for_render, extras=None, overlay=True
            )
        else:
            return super().render(mode=mode, **kwargs)


def build_wrapper(cfg, default_args=None):
    return build_from_cfg(cfg, WRAPPERS, default_args)
