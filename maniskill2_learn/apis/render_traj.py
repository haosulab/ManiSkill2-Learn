import os.path as osp, os, cv2
import json
import numpy as np
from pathlib import Path
from maniskill2_learn.env.env_utils import build_env
from maniskill2_learn.utils.data import GDict
from maniskill2_learn.utils.meta import mkdir_or_exist, ConfigDict
from copy import deepcopy


def get_reset_kwargs_from_json(json_name):
    with open(json_name, "r") as f:
        json_file = json.load(f)
    reset_kwargs = {}
    for d in json_file["episodes"]:
        episode_id = d["episode_id"]
        r_kwargs = d["reset_kwargs"]
        reset_kwargs[episode_id] = r_kwargs
    return reset_kwargs

def requires_rollout_w_actions(trajectory):
    keys = trajectory.keys()
    assert "env_states" in keys or "env_init_state" in keys
    return (not 'env_states' in keys)

def render_trajectories(trajectory_file, json_name, env_name, control_mode, video_dir):
    reset_kwargs = get_reset_kwargs_from_json(json_name)
    trajectories = GDict.from_hdf5(trajectory_file, wrapper=False)
    env = build_env(ConfigDict(
        {"type": "gym", "env_name": env_name, "control_mode": control_mode})
    )
    if not osp.exists(video_dir):
        os.makedirs(video_dir)

    for traj_name in trajectories:
        trajectory = trajectories[traj_name]
        traj_idx = eval(traj_name.split("_")[1])
        env.reset(**reset_kwargs[traj_idx])

        rrwa = requires_rollout_w_actions(trajectory)
        if rrwa:
            state = trajectory["env_init_state"]
            length = trajectory["actions"].shape[0] + 1
        else:
            state = trajectory["env_states"]
            length = state.shape[0]
        img = env.render("rgb_array")

        video_writer = cv2.VideoWriter(osp.join(video_dir, f"{traj_idx}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (img.shape[1], img.shape[0]))
        for j in range(length):
            if not rrwa:
                env.set_state(state[j])
            else:
                if j == 0:
                    pass# env.set_state(state)
                else:
                    _ = env.step(trajectory["actions"][j - 1])
            img = env.render("rgb_array")
            img = img[..., ::-1]
            video_writer.write(img)
        video_writer.release()


def render_with_o3d(trajectory_file, json_name, env_configs, traj_id=0):
    data = GDict.from_hdf5(trajectory_file)
    reset_kwargs = get_reset_kwargs_from_json(json_name)
    env = build_env(ConfigDict(**env_configs))

    from pynput import keyboard
    import open3d as o3d
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    geometry = o3d.geometry.PointCloud()
    geometry.points = o3d.utility.Vector3dVector(np.random.random([3,3]))
    geometry.colors = o3d.utility.Vector3dVector(np.ones([3,3]))
    vis.add_geometry(geometry)    

    trajectory = data[f'traj_{traj_id}']
    rrwa = requires_rollout_w_actions(trajectory)
    if not rrwa:
        env_states = trajectory['env_states']
        length = env_states.shape[0]
    else:
        env_states = trajectory['env_init_state']
        length = trajectory['actions'].shape[0] + 1

    env.reset(**reset_kwargs[traj_id])

    idx = 0
    def on_press(key):
        nonlocal idx
        if hasattr(key, 'char'):
            if key.char in ['n']:
                idx = idx + 1    
            elif key.char in ['l']:
                if rrwa:
                    print("Cannot go back to the previous frame because env_states is not given for every step.")
                else:
                    idx = max(idx - 1, 0)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()     

    print("Press 'n' for next frame, 'l' for previous frame, 'h' for Open3d help")
    while idx < length:
        if not rrwa:
            env.set_state(env_states[idx])
        else:
            if idx == 0:
                env.set_state(env_states)
            else:
                env.step(trajectory['actions'][idx - 1])
        obs = env.get_obs()
        geometry.points = o3d.utility.Vector3dVector(obs['xyz'])
        geometry.colors = o3d.utility.Vector3dVector(obs['rgb'])
        vis.update_geometry(geometry)
        old_idx = idx
        while idx == old_idx:
            vis.poll_events()
            vis.update_renderer()


def render_with_o3d_random_trajectory(env_configs):
    env = build_env(ConfigDict(**env_configs))

    from pynput import keyboard
    import open3d as o3d
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    geometry = o3d.geometry.PointCloud()
    geometry.points = o3d.utility.Vector3dVector(np.random.random([3,3]))
    geometry.colors = o3d.utility.Vector3dVector(np.ones([3,3]))
    vis.add_geometry(geometry)
    env.reset()

    idx = 0
    def on_press(key):
        nonlocal idx
        if hasattr(key, 'char'):
            if key.char in ['n']:
                idx = idx + 1

    listener = keyboard.Listener(on_press=on_press)
    listener.start()     

    print("Press 'n' for next frame, 'h' for Open3d help")
    while True:
        env.step(env.action_space.sample())
        obs = env.get_obs()
        geometry.points = o3d.utility.Vector3dVector(obs['xyz'])
        geometry.colors = o3d.utility.Vector3dVector(obs['rgb'])
        vis.update_geometry(geometry)
        old_idx = idx
        while idx == old_idx:
            vis.poll_events()
            vis.update_renderer()


if __name__ == "__main__":
    render_trajectories(
        trajectory_file = '/home/xuanlin/exp-logs/maniskill_2022/demos/PickSingleYCB-v0/trajectory.none.pd_joint_delta_pos.h5',
        json_name = '/home/xuanlin/exp-logs/maniskill_2022/demos/PickSingleYCB-v0/trajectory.none.pd_joint_delta_pos.json',
        env_name = 'PickSingleYCB-v0',
        control_mode = 'pd_joint_delta_pos',
        video_dir = "/home/xuanlin/exp-logs/maniskill_2022/demos/PickSingleYCB-v0/videos",
    )

    # render_with_o3d(
    #     trajectory_file = '/home/xuanlin/exp-logs/maniskill_2022/demos/PickSingleYCB-v0/trajectory.none.pd_joint_delta_pos.h5',
    #     json_name = '/home/xuanlin/exp-logs/maniskill_2022/demos/PickSingleYCB-v0/trajectory.none.pd_joint_delta_pos.json',
    #     env_configs = dict(type='gym',
    #          env_name='PickSingleYCB-v0',
    #          control_mode='pd_joint_delta_pos',
    #          obs_mode='pointcloud',
    #          n_points=5000,
    #          n_goal_points=50,
    #          obs_frame='base',
    #     ),
    #     traj_id = 0,
    # )    

    # render_with_o3d_random_trajectory(dict(
    #         type='gym',
    #         env_name='StackCube-v0',
    #         control_mode='pd_joint_delta_pos',
    #         obs_mode='pointcloud',
    #         n_points=1200,
    #         obs_frame='base',
    #     ),
    # )