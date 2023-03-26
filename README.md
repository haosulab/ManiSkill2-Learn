# ManiSkill2-Learn

ManiSkill2-Learn is a framework for training agents on [SAPIEN Open-Source Manipulation Skill Challenge 2](https://sapien.ucsd.edu/challenges/maniskill/2022/), a physics-rich generalizable manipulation skill benchmark over diverse objects and diverse rigid & soft-body environments with large-scale demonstrations.

Updates will be posted here.

Mar. 25, 2023: **[Breaking change, Important]** We modified `maniskill2_learn/env/wrappers.py` such that camera orders in RGBD mode are fixed and consistent, otherwise there can be unexpected performance drops when evaluating a RGBD checkpoint (especially when evaluating our published checkpoints on the latest ManiSkill2 >=0.4.0, and these checkpoints were trained on ManiSkill2 0.3.0). This is due to the fact that camera keys in visual observations are in inconsistent orders across different ManiSkill2 versions. The order of visual observation keys in ManiSkill2 >= 0.4.0 is different from Maniskill2 0.3.0  (i.e., [base_camera, hand_camera] in 0.4.0 and [hand_camera, base_camera] in 0.3.0). Since older ManiSkill2-Learn (prior to Mar. 25, 2023) simply stacks the observations from different cameras without checking the orders of these cameras, this causes problems.

Thus, we have fixed the camera orders to [hand_camera, base_camera] and [overhead_camera_0, overhead_camera_1, overhead_camera_2] for different environments. Please pull and check the latest updates.

However, if you have been using ManiSkill2-Learn lately and trained models on the latest ManiSkill2 (>=0.4.0) under RGB-D observation mode, then the above change in ManiSkill2-Learn can cause your recently trained models to evaluate poorly, since your models are most likely trained with [base_camera, hand_camera] following the visual key orders ManiSkill2 0.4.0, while the latest ManiSkill2-Learn fixes the order to [hand_camera, base_camera] to ensure consistency with our published pretrained models and the visual key orders of ManiSkill2 0.3.0.

Mar. 2, 2023: **[Breaking change, Important]** Factor in the environment render gpu config breaking change in ManiSkill2 0.4. Please pull the latest updates.

Feb. 12, 2023: Codebase updated to factor in some breaking changes in ManiSkill2 0.4. Please pull the latest updates.

Nov. 29, 2022: Modify example scripts to account for ManiSkill2 0.3 breaking change where ManiSkill2 tools/replay_trajectory.py is moved to mani_skill2/trajectory/replay_trajectory.py

Oct. 18, 2022: Added more example scripts for convenience, along with pretrained models.

Sep. 25, 2022: Address different action sampling modes during evaluation, since using the mean of gaussian as action output during evaluation could sometimes lead to lower success rates than during training rollouts, where actions are sampled stochastically. See [here](#important-notes-for-evaluation)

Sep. 18, 2022: Fixed demonstration / replay loading such that when specifying `num_samples=n` (i.e. limit the number of demos loaded per file to be `n`), it will not open the entire `.h5` file, thus saving memory. Also, add instructions for dynamically loading demonstrations through `dynamic_loading` in the replay buffer configuration.

Sep. 14, 2022: Added gzip by default when saving `.hdf5` files.

Sep. 11, 2022: Added `FrameMiner-MixAction` for point cloud-based manipulation learning from [Frame Mining: a Free Lunch for Learning Robotic Manipulation from 3D Point Clouds](https://openreview.net/forum?id=d-JYso87y6s) (CoRL 2022).

Sep. 5, 2022: Change `GDict.shape.memory` to `GDict.shape`; remove `.memory` usage in several places of the codebase.

Aug. 14, 2022: If you encounter such error after training finishes, you can simply ignore it. It does not affect results.
```
ImportError: sys.meta_path is None, Python is likely shutting down
/home/anaconda3/envs/mani_skill2/lib/python3.8/multiprocessing/resource_tracker.py:216: UserWarning: resource_tracker: There appear to be 9 leaked shared_memory objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
```

- [ManiSkill2-Learn](#maniskill2-learn)
  - [Getting Started](#getting-started)
    - [Installation](#installation)
  - [Example Scripts and Models](#example-scripts-and-models)
    - [Demonstration Conversion Scripts](#demonstration-conversion-scripts)
    - [Training Scripts](#training-scripts)
    - [Pretrained Models and Example Videos](#pretrained-models-and-example-videos)
  - [Simple Customized Workflow](#simple-customized-workflow)
    - [Converting and Viewing Demonstrations](#converting-and-viewing-demonstrations)
    - [Training and Evaluation](#training-and-evaluation)
    - [Important Notes on Control Mode](#important-notes-on-control-mode)
    - [Important Notes for Point Cloud-based Learning (with Demonstrations)](#important-notes-for-point-cloud-based-learning-with-demonstrations)
    - [Important Notes for Evaluation](#important-notes-for-evaluation)
  - [More Detailed Workflow](#more-detailed-workflow)
    - [Converting and Viewing Demonstrations](#converting-and-viewing-demonstrations-1)
      - [Demonstration Format and Environment Wrapper](#demonstration-format-and-environment-wrapper)
      - [Further Functionalities on Rendering and Interactively Viewing Demonstrations](#further-functionalities-on-rendering-and-interactively-viewing-demonstrations)
      - [Merging Demonstration Trajectories](#merging-demonstration-trajectories)
      - [Rigid and Soft Body Demonstrations](#rigid-and-soft-body-demonstrations)
      - [Customizing Demonstrations and Environment Wrappers](#customizing-demonstrations-and-environment-wrappers)
    - [Training and Evaluation](#training-and-evaluation-1)
      - [Overview](#overview)
      - [Configuration Files, Networks, and Algorithms](#configuration-files-networks-and-algorithms)
      - [Simulation and Network Parallelism](#simulation-and-network-parallelism)
      - [Replay Buffer](#replay-buffer)
      - [Some things to keep in mind](#some-things-to-keep-in-mind)
  - [Submission Example](#submission-example)
  - [Other Useful Functionalities](#other-useful-functionalities)
    - [Loading h5 files and handling iterated dictionaries with GDict](#loading-h5-files-and-handling-iterated-dictionaries-with-gdict)
  - [Acknowledgements](#acknowledgements)
  - [Citation](#citation)
  - [License](#license)

## Getting Started

### Installation

ManiSkill2-Learn requires `python >= 3.8` and `cuda 11.3`. For pytorch, our framework is tested on `pytorch == 1.11.0+cu113` and we recommend installing it. We do not recommend installing `pytorch == 1.12` due to a known [issue](https://github.com/pytorch/pytorch/issues/80809). When `pytorch >= 1.12.1` comes out in the future, you may choose to install the newer version.

To get started, enter the parent directory of where you installed [ManiSkill2](https://github.com/haosulab/ManiSkill2) and clone this repo. Assuming the anaconda environment you installed ManiSkill2 is `mani_skill2`, execute the following commands (**note that the ordering is strict**):

```
cd {parent_directory_of_ManiSkill2}
conda activate mani_skill2 #(activate the anaconda env where ManiSkill2 is installed)
git clone https://github.com/haosulab/ManiSkill2-Learn
cd ManiSkill2-Learn
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
pip install pytorch3d
pip install ninja
pip install -e .
pip install protobuf==3.19.0
```

If you would like to use SparseConvNet to perform 3D manipulation learning, install `torchsparse` and its releated dependencies (the `torchsparse` below is forked from the original repo with bug fix and additional normalization functionalities):

```
sudo apt-get install libsparsehash-dev # brew install google-sparsehash if you use Mac OS
pip install torchsparse@git+https://github.com/lz1oceani/torchsparse.git
```

## Example Scripts and Models

In this section, we provide convenient (minimal) examples for training and evaluating a model. If you would like to perform custom training, see the later sections for more detailed information, caveats, etc.
### Demonstration Conversion Scripts

We have provided example scripts for converting raw demonstrations in ManiSkill2 into the ones ready for downstream training. These scripts are in `./scripts/example_demo_conversion`. Different tasks may use different scripts. Please see the scripts for more details.

**Some scripts require bash to execute. If you are using another shell, please modify the scripts before executing them.** 

### Training Scripts

We have provided example training scripts to reproduce our pretrained models in `./scripts/example_training/pretrained_model`. Please see the scripts for more details. You need to modify the logging directory and the demonstration file path before training. Evaluation is automatically done after training.
### Pretrained Models and Example Videos

We have provided pretrained models (for tasks that achieve >10% performance) in our google drive. We have also provided example videos to showcase success and failures of each task using our models.

You can download them through the following script:

```
gdown https://drive.google.com/drive/folders/1VIJRJvazlEmAUjVGW0QJ9-d91D0EzyXm --folder
cd maniskill2_learn_pretrained_models_videos/
unzip maniskill2_learn_pretrained_models_videos.zip
```

To evaluate a DAPG+PPO model on rigid-body tasks, you can use the following script:
```
# Although models were trained with DAPG+PPO, since we are now only evaluating models,
# the demonstrations are not useful here, so we don't need to load the demonstrations through 
# DAPG configurations, and we can just use PPO configuration to evaluate the model, 
# as long as the model architecture configurations are the same

# For tasks other than Pick-and-Place (i.e. those that do not start with "Pick"), 
# you need to remove the argument "env_cfg.n_goal_points=50"

# Evaluating point cloud-based model
python maniskill2_learn/apis/run_rl.py configs/mfrl/ppo/maniskill2_pn.py \
            --work-dir YOUR_LOGGING_DIRECTORY --gpu-ids 0 \
            --cfg-options "env_cfg.env_name=PickSingleYCB-v0" "env_cfg.obs_mode=pointcloud" "env_cfg.n_points=1200" \
            "env_cfg.control_mode=pd_ee_delta_pose" "env_cfg.obs_frame=ee" "env_cfg.n_goal_points=50" \
            "eval_cfg.num=100" "eval_cfg.num_procs=5" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
            --evaluation --resume-from maniskill2_learn_pretrained_models_videos/PickSingleYCB-v0/dapg_pointcloud/model_25000000.ckpt

# Evaluating RGBD-based model
python maniskill2_learn/apis/run_rl.py configs/mfrl/ppo/maniskill2_rgbd.py \
            --work-dir YOUR_LOGGING_DIRECTORY --gpu-ids 0 \
            --cfg-options "env_cfg.env_name=PickSingleYCB-v0" "env_cfg.obs_mode=rgbd" \
            "env_cfg.control_mode=pd_ee_delta_pose" \
            "eval_cfg.num=100" "eval_cfg.num_procs=5" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
            --evaluation --resume-from maniskill2_learn_pretrained_models_videos/PickSingleYCB-v0/dapg_rgbd/model_25000000.ckpt
```

To evaluate a point cloud-based BC model on soft-body tasks, you can use the following script:
```
CUDA_VISIBLE_DEVICES=1 python maniskill2_learn/apis/run_rl.py configs/brl/bc/pointnet_soft_body.py \
            --work-dir YOUR_LOGGING_DIRECTORY --gpu-ids 0 \
            --cfg-options "env_cfg.env_name=Excavate-v0" "env_cfg.obs_mode=pointcloud" \
            "env_cfg.control_mode=pd_joint_delta_pos" \
            "eval_cfg.num=100" "eval_cfg.num_procs=4" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
            --evaluation --resume-from maniskill2_learn_pretrained_models_videos/Excavate_pcd/model_final.ckpt
```

You can also submit a pretrained model to our evaluation server if you'd like to. Please see files under `submission_example/` for more details.



## Simple Customized Workflow

### Converting and Viewing Demonstrations

**Before you convert (render) any demonstrations using ManiSkill2-Learn**, make sure that the demonstration has been converted to your desired control mode using the script from the ManiSkill2 repo (by following instructions [here](https://github.com/haosulab/ManiSkill2/#demonstrations)). ManiSkill2-Learn only supports customizing observation processing for demonstration generation / agent learning, and does not support control mode conversion.

The original demonstration files from ManiSkill2 do not contain visual information as they would be too large to download. Therefore, we provided helpful functionalities in `tools/convert_state.py` to locally convert environment-state trajectories in the demonstration files into visual trajectories that can be used for training in ManiSkill2-Learn. Run `tools/convert_state.py -h` for detailed command options.

An example script to render point cloud demonstrations is shown below:

```
# Replace `PATH` with appropriate path and `ENV` with appropriate environment name

python tools/convert_state.py --env-name ENV_NAME --num-procs 1 \
--traj-name PATH/trajectory.none.pd_joint_delta_pos.h5 \
--json-name PATH/trajectory.none.pd_joint_delta_pos.json \
--output-name PATH/trajectory.none.pd_joint_delta_pos_pcd.h5 \
--control-mode pd_joint_delta_pos --max-num-traj -1 --obs-mode pointcloud \
--n-points 1200 --obs-frame base --reward-mode dense --render
```

Further details on advanced usage and customization are presented [here](#converting-and-viewing-demonstrations-1).

Note that the `--render` option allows you to view the demonstration trajectories in Vulkan. However, `--render` slows down the demonstration conversion process, and should only be invoked when `--num-procs=1`. Without `--render` option, you can use multi-processing to accelerate demonstration conversion speed by setting `--num-procs=X`.

To use the Vulkan viewer, drag the mouse while `right-click` to rotate; press `A/D` to translate the view; `scroll` or `W/S` to zoom in and out; `left-click` to select a particular object part; once an object part is selected, you can press `f` to focus on the object part, such that view rotation and zooming are centered on this object part; you can also view its collision shape through the side bar, etc. 

### Training and Evaluation

Maniskill2_learn implements common Reinforcement Learning algorithms, including Behavior Cloning (BC), Proximal Policy Gradient (PPO), Demonstration-Augmented Policy Gradient (DAPG), Soft Actor-Critic (SAC), and Generative-Adversarial Imitation Learning (GAIL). For example, to train a PPO Agent on `PickCube-v0` with point cloud observation using the default algorithm configs in `configs/mfrl/ppo/maniskill2_pn.py`, run the following command (note that PPO is a demonstration-free algorithm):

```
python maniskill2_learn/apis/run_rl.py configs/mfrl/ppo/maniskill2_pn.py \
--work-dir YOUR_LOGGING_DIRECTORY --gpu-ids 0 \
--cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=pointcloud" \
"env_cfg.n_points=1200" "env_cfg.control_mode=pd_joint_delta_pos" \
"env_cfg.reward_mode=dense" "rollout_cfg.num_procs=5" \
"eval_cfg.num=100" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"eval_cfg.num_procs=5"

# "num_procs" controls parallelism during training. Details are described in later sections.

# FPS reported denotes the number of *control steps* per second.
# Note that the default simulation frequency in ManiSkill2 environments is 500hz, control frequency is 20hz.
# Therefore, 1 control step = 25 simulation steps.

# The above command does automatic evaluation after training. 
# Alternatively, you can manually evaluate a model checkpoint 
# by appending --evaluation and --resume-from YOUR_LOGGING_DIRECTORY/models/SOME_CHECKPOINT.ckpt 
# to the above commands.
```

To train a DAPG agent on `PegInsertionSide-v0` with point cloud observation, an example command is shown below (note that DAPG requires loading demonstrations):

```
python maniskill2_learn/apis/run_rl.py configs/mfrl/dapg/maniskill2_pn.py \
--work-dir YOUR_LOGGING_DIRECTORY --gpu-ids 0 \
--cfg-options "env_cfg.env_name=PegInsertionSide-v0" "env_cfg.obs_mode=pointcloud" \
"env_cfg.n_points=1200" "env_cfg.control_mode=pd_joint_delta_pos" \
"env_cfg.reward_mode=dense" "rollout_cfg.num_procs=5" \
"agent_cfg.demo_replay_cfg.buffer_filenames=PATH_TO_POINT_CLOUD_DEMO.h5" \
"eval_cfg.num=100" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"eval_cfg.num_procs=5"

# To manually evaluate the model, 
# add --evaluation and --resume-from YOUR_LOGGING_DIRECTORY/models/SOME_CHECKPOINT.ckpt 
# to the above commands.
```


For example commands on more algorithms, see `scripts/example_training/`.

For further details about specific arguments and configs, along with algorithm / wrapper customization, please read [here](#training-and-evaluation-1)

### Important Notes on Control Mode

Control modes could play a significant role in agent performance. For example, `pd_ee_delta_pose` generally performs better than `pd_joint_delta_pos`.

To modify the control mode used for training, pass in `env_cfg.control_mode={CONTROL_MODE}` to the `--cfg-options` in the training scripts above. Note that if you run demonstration-based algorithms (e.g., DAPG, GAIL), the demonstration actions need to be first converted to your desired control mode for training. This is done through `mani_skill2/trajectory/replay_trajectory.py` in **ManiSkill2 (not ManiSkill2-Learn)** (see ManiSkill2 readme). After you convert the demonstration actions to your desired control mode, you can then use `tools/convert_state.py` (see scripts above) to generate visual demonstration observations, and then run the training code.

Another caveat is that some controller names are very similar (e.g. `pd_ee_delta_pos` and `pd_ee_delta_pose`), but they are completely different controllers. Please pay attention to this!

### Important Notes for Point Cloud-based Learning (with Demonstrations)

For point cloud-based learning, there are some useful configurations you can add to `--cfg-options`. A few examples:
- `"env_cfg.obs_frame=ee"` : transform point cloud to the end-effector frame (instead of the robot base frame by default, i.e. `"env_cfg.obs_frame=base"`) 
- `"env_cfg.n_goal_points=50"` : if the environment observation contains goal position ("goal_pos"), randomly sample 50 points near the goal position and append to the point cloud; this allows goal info to be visual; if an environment does not have goal position (e.g. in PegInsertionSide-v0), error will be raised.

**Important notes:**
- Configuration options like these could significantly affect performance. For `PickCube-v0`, simultaneously adding both config options above allows PPO to achieve a very high success rate within a few million steps.
- If you are using demonstration-based algorithms (e.g. DAPG, GAIL), you need to ensure that the demonstration are rendered using the same setting as during training. For example, if you use `obs_frame=ee` and `n_goal_points=50` during training, then you should ensure that the demonstrations are rendered this way.
- You can visualize point clouds using the utilities [here](https://github.com/haosulab/ManiSkill2-Learn/blob/main/maniskill2_learn/utils/visualization/o3d_utils.py). You can import such functionalities through `from maniskill2_learn.utils.visualization import visualize_pcd`.


### Important Notes for Evaluation

When you run non-Behavior-Cloning RL algorithms (e.g. DAPG+PPO), during training, actions are sampled stochastically, i.e. randomly sampled based on the output gaussian distribution. During evaluation, by default, the gaussian mean is used as action output. However, this could sometimes cause the evaluation result to be a bit lower than during training, for some environments such as `TurnFaucet-v0` and `PickSingleYCB-v0`. This is because noise could be beneficial for success (e.g. adding noise leads to more object grasping attempts). If you encounter this case, you could add `--cfg-options "eval_cfg.sample_mode=sample"` such that the policy will randomly sample actions during evaluation.



















## More Detailed Workflow

### Converting and Viewing Demonstrations

#### Demonstration Format and Environment Wrapper

For demonstrations converted through `tools/convert_state.py`, the rendered point cloud / RGB-D observations from the environment are post-processed through environment wrappers in `maniskill2_learn/envs/wrappers.py` and saved in the output file. Since training uses the same wrapper as during demonstration convertion, the output demonstration file can be directly loaded into the replay buffer.

The **converted demonstration** (not the original demonstration containing no observation information) has a format similar to the following:

```
from maniskill2_learn.utils.data import GDict
f = GDict.from_hdf5(TRAJ_PATH)
print(f.keys()) # dict_keys(['traj_0', 'traj_1', ...])
traj_0 = f['traj_0']

print(GDict(traj_0).shape)
# Depending on --obs-mode, --n-points, etc, this will print, e.g.
# {'actions': (143, 8), 'dones': (143, 1), 'rewards': (143, 1), 'episode_dones': (143, 1), 
# 'is_truncated': (143, 1), 'obs': {'frame_related_states': (143, 2, 3), 
# 'rgb': (143, 1200, 3), 'state': (143, 24), 'to_frames': (143, 2, 4, 4), 
# 'xyz': (143, 1200, 3)}, 'worker_indices': (143, 1)}
```

#### Further Functionalities on Rendering and Interactively Viewing Demonstrations

We provide further functionalities on rendering and interactively viewing demonstrations in `maniskill2_learn/apis/render_traj.py`. To use the script, modify the corresponding arguments inside the script. See `render_trajectories` and `render_o3d` functions in the script for more details. 

#### Merging Demonstration Trajectories

We provide useful functionalities to merge multiple HDF5 files with a certain pattern into a single file in `tools/merge_h5.py`.

#### Rigid and Soft Body Demonstrations

`tools/convert_state.py` is capable of automatically handling both rigid-body demonstration and soft-body demonstration **source files** published at ManiSkill2022 challenge. For demonstrations in rigid-body environments, the source file provides environment states at each time step, so we set the corresponding environment state at each time step and render to obtain visual observations. For demonstrations in soft-body environments, the environment state is provided only at the first environment state (it would have been too large to store environment states at all steps since the number of particles is large). In this case, we rollout actions from the demonstrations with the same seed as when generating demonstrations to obtain visual observations.

Note that, regardless of whether a **source file** (i.e. the file containing environment state(s)) belongs to a rigid body environment or a soft body environment, the final output file (i.e. the file in `--output-name`) contains trajectories post-processed through the observation wrapper at `maniskill2_learn/env/wrappers.py`.


#### Customizing Demonstrations and Environment Wrappers

The environment and wrapper construction first go through the `make_gym_env` function in `maniskill2_learn/env/env_utils.py`. After the environment and the wrapper have been constructed, during training / evaluation / demonstration generation, the `observation` function under the `ManiSkll2_ObsWrapper` class in `maniskill2_learn/env/wrappers.py` post-processes observations from the environment. Therefore, if you wish to support advanced functionalities, e.g. advanced point cloud downsampling and coordinate transformation, both `maniskill2_learn/env/env_utils.py` and `maniskill2_learn/env/wrappers.py` need to be modified.






### Training and Evaluation

#### Overview

At the high level, `maniskill2_learn/apis/run_rl.py` provides a unified framework for training and evaluation, and is invoked through `python maniskill2_learn/apis/run_rl.py CONFIG_FILE --SOME_ARGS --cfg-options SOME_CONFIG_OPTIONS`. Here `CONFIG_FILE` is a configuration file in `configs/`; `SOME_ARGS` are some custom arguments from `run_rl.py`; `--cfg-options` are config options that override the original configs in `CONFIG_FILE`. 

If `maniskill2_learn/apis/run_rl.py` is invoked with training mode, `maniskill2_learn/apis/run_rl.py` will invoke `maniskill2_learn/apis/train_rl.py` to handle specific training steps. If `maniskill2_learn/apis/run_rl.py` is invoked with evaluation mode (i.e. `--evaluation`), then `maniskill2_learn/env/evaluation.py` will handle the evaluation process. Success rate is reported as evaluation results.

#### Configuration Files, Networks, and Algorithms

We have included several example configuration files in `configs/` for training an agent using various algorithms with RGB-D or point cloud observation modes. Example scripts are in `scripts/example_training`. 

An agent is typically an actor-critic agent (`maniskill2_learn/networks/applications/actor_critic.py`). Algorithms are implemented in `maniskill2_learn/methods/`. Network architectures are implemented in `maniskill2_learn/networks/`. 

#### Simulation and Network Parallelism

ManiSkill2-learn provides an easy-to-use interface to support simulation parallelism and network training / evaluation parallelism. 
- In `maniskill2_learn/apis/run_rl.py`, you can specify `--gpu-ids` for multi-gpu network training and `--sim-gpu-ids` for multi-gpu environment simulation. For example, you can use `--sim-gpu-ids 0` to do simulation and `--gpu-ids 1 2` to do network training. If `--sim-gpu-ids` is not specified, it becomes the same as `--gpu-ids`, i.e. simulation and network training are done on the same GPU.
- In `--cfg-options`, you can use `rollout_cfg.num_procs=X` to specify the number of parallel environments per simulation gpu in `--sim-gpu-ids`. You can use `eval_cfg.num_procs` to specify the number of parallel processes during evaluation (final results are merged).
- Note that during training, if multiple simulation GPUs are used, then some arguments in the configuration file (e.g. `train_cfg.n_steps`, `replay_cfg.capacity`) are effectively multiplied by the number of simulation GPUs. Similarly, if multiple network-training GPUs are used, then some arguments (e.g. `agent_cfg.batch_size`) are effectively multiplied by the number of network-training GPUs.

**Important note for soft-body environments**: soft-body environments typically consume much more memory than a single rigid-body environment and runs much slower. However, as of now, they do not allow multi-gpu training. Thus currently, the only way to train an agent on these environments is by setting `CUDA_VISIBLE_DEVICES=X` and `--gpu-ids=0`. In addition, to keep the same number of parallel processes during rollout, you need to set a smaller batch size than for rigid-body environments. You can also modify our code in order to accumulate gradient steps.

#### Replay Buffer

For the replay buffer used during online agent rollout, its configuration is specified in the `replay_cfg` dictionary of the config file. Some algorithms (e.g. GAIL) also have `recent_traj_replay_cfg` since the recent rollout trajectories are used for discriminator training. 

For algorithms that use demonstrations (e.g. DAPG, GAIL), additional replay buffers need to be used, and these replay buffers exclusively contain demonstrations (`demo_replay_cfg` for DAPG+PPO, and `expert_replay_cfg` for GAIL+SAC). Since demonstration files can contain many trajectories, and loading all of them could result in out-of-memory, we provided a useful argument `num_samples` to limit the number of demonstration trajectories loaded into the replay buffer **for each file in `buffer_filenames`** (in default configurations, `num_samples=-1`, so all trajectories are loaded). Currently, if `num_samples` is used, then the first `num_samples` trajectories are loaded, instead of randomly sampling trajectories.

Alternatively, you could turn on `dynamic_loading` in the replay buffer configuration. In this case, a batch of demonstrations will be first loaded into the replay buffer, and after all its entries have been sampled, a new batch will be loaded. Thus, you don't need to limit the number of trajectories loaded per demonstration file. As an example for DAPG+PPO,

```
demo_replay_cfg=dict(
    type="ReplayMemory",
    capacity=int(2e4),
    num_samples=-1,
    cache_size=int(2e4),
    dynamic_loading=True,
    synchronized=False,
    keys=["obs", "actions", "dones", "episode_dones"],
    buffer_filenames=[
        "PATH_TO_DEMO.h5",
    ],
),
```


#### Some things to keep in mind

When training agents with demonstrations (e.g. DAPG, GAIL), the demonstration should be generated in a way that exactly matches the environment configurations used during training. Pay special attention to e.g. observation mode, point cloud downsampling strategy, observation frame, goal point sampling, etc.









## Submission Example

To facilitate ManiSkill2022 challenge submission, we have added a submission example in `submission_example/` that uses ManiSkill2-Learn. Please refer to the file comments for specific instructions.










## Other Useful Functionalities
### Loading h5 files and handling iterated dictionaries with GDict

We provide `GDict` in `maniskill2_learn.utils.data` for convenient handling of iterated dictionaries in the replay buffer and in HDF5 files. To load a `.h5` trajectory with `GDict` and visualize array shapes & dtypes, you can use the following code snippet:

```
from maniskill2_learn.utils.data import GDict
f = GDict.from_hdf5(TRAJ_PATH)
print(f.keys()) # prints 'traj_0', 'traj_1', etc
traj_0 = GDict(f['traj_0'])

print(traj_0.shape) 
# prints the array shape of all values in the iterated dict
# For example, {'actions': (344, 8), 'dones': (344, 1), 'rewards': (344, 1), 'episode_dones': (344, 1), 
# 'is_truncated': (344, 1),'obs': {'frame_related_states': (344, 2, 3), 
# 'rgb': (344, 1200, 3), 'state': (344, 24), 
# 'to_frames': (344, 2, 4, 4), 'xyz': (344, 1200, 3)}}

print(traj_0.dtype)
# For example: {'/actions': 'float32', '/dones': 'bool', '/episode_dones': 'bool', 
# '/is_truncated': 'bool', '/obs/frame_related_states': 'float32', 
# '/obs/rgb': 'float32', '/obs/state': 'float32', '/obs/to_frames': 'float32', 
# '/obs/xyz': 'float32', '/worker_indices': 'int32'}
```

`GDict` also supports many useful functionalities, e.g. wrapping / unwrapping; type casting; transferring data between numpy & torch and between cpu & gpu; detaching a GDict of torch tensors. For example,

```
import torch, numpy as np
from maniskill2_learn.utils.data import GDict
d = GDict({'a': torch.randn(3,4), 'b': {'c': torch.randn(2,3)}})

# numpy <-> torch
assert d['a'].dtype == torch.float32
d_np = d.to_numpy()
assert d_np['a'].dtype == np.float32
d_2 = d_np.to_torch(device="cpu") # note that tensors in d_2 shares the memory with those in d
assert d_2['a'].dtype == torch.float32
d_2 = d_np.to_torch(device="cpu", non_blocking=True) # to_torch supports 'non_blocking' to accelerate speed
assert d_2['a'].dtype == torch.float32

# Type casting
d_i8 = d.astype('int8')
assert d_i8['a'].dtype == torch.int8
d_np_i16 = d_np.astype('int16')
assert d_np_i16['a'].dtype == np.int16

# cpu <-> gpu
d_cuda = d_np.to_torch(device='cuda:0')
print(d_cuda['a'].device) # cuda:0
d_cuda = d.cuda('cuda:0')
print(d_cuda['a'].device) # cuda:0
d_cpu = d_cuda.cpu()
print(d_cpu['a'].device) # cpu

# Unwrapping from GDict if 'wrapper=False' is taken as an argument in 'to_torch' and 'to_numpy'.
# Use GDict(d) to wrap GDict on top of any dictionary of arrays
d_2 = d_np.to_torch(device="cpu", non_blocking=True, wrapper=False)
assert isinstance(d_2, dict)
d_3 = GDict(d_2).to_torch(device='cuda:0', non_blocking=True, wrapper=False)
assert isinstance(d_3, dict)
d_np_3 = GDict(d_3).to_numpy(wrapper=False)
assert isinstance(d_np_3, dict)

# Detach a GDict of torch tensors
x = GDict({'a': torch.randn(3, requires_grad=True)})
y = torch.randn(3, requires_grad=True)
x_detach = x.detach()
print(x['a'].requires_grad) # True
print(x_detach['a'].requires_grad) # False
loss = x_detach['a'] + y
loss.sum().backward()
print(y.grad) # tensor([1., 1., 1.])
print(x['a'].grad) # None
```


In addition to `GDict`, there is `DictArray` that behaves very similar to `GDict`, except with an additional check that all elements in `DictArray` has the same batch dimension (i.e. `x.shape[0]` is the same for all `x` in a `DictArray`). To import `DictArray`, use `from maniskill2_learn.utils.data import DictArray`.








## Acknowledgements

Some functions (e.g. config system, checkpoint) are adopted from [MMCV](https://github.com/open-mmlab/mmcv):

GAIL implementation and hyperparameters are modified from Hao Shen, Weikang Wan, and He Wang's ManiSkill2021 challenge submission: [paper](https://arxiv.org/abs/2203.02107) and [code](https://github.com/wkwan7/EPICLab-ManiSkill).

## Citation

```
@inproceedings{gu2023maniskill2,
          title={ManiSkill2: A Unified Benchmark for Generalizable Manipulation Skills},
          author={Gu, Jiayuan and Xiang, Fanbo and Li, Xuanlin and Ling, Zhan and Liu, Xiqiaing and Mu, Tongzhou and Tang, Yihe and Tao, Stone and Wei, Xinyue and Yao, Yunchao and Yuan, Xiaodi and Xie, Pengwei and Huang, Zhiao and Chen, Rui and Su, Hao},
          booktitle={International Conference on Learning Representations},
          year={2023}
}
```

## License
ManiSkill2-Learn is released under the Apache 2.0 license, while some specific operations in this library are with other licenses.
