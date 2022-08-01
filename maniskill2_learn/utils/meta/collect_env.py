import os.path as osp, subprocess, sys, cv2, time
from collections import defaultdict
from pathlib import Path
from importlib import import_module


def get_PIL_version():
    try:
        import PIL
    except ImportError:
        return "None"
    else:
        return f"{PIL.__version__}"


def collect_base_env():
    """Collect information from system environments.
    Returns:
        dict: The environment information. The following fields are contained.
            - sys.platform: The variable of ``sys.platform``.
            - Python: Python version.
            - CUDA available: Bool, indicating if CUDA is available.
            - GPU devices: Device type of each GPU.
            - CUDA_HOME (optional): The env var ``CUDA_HOME``.
            - NVCC (optional): NVCC version.
            - GCC: GCC version, "n/a" if GCC is not installed.
            - PyTorch: PyTorch version.
            - PyTorch compiling details: The output of ``torch.__config__.show()``.
            - TorchVision (optional): TorchVision version.
            - OpenCV: OpenCV version.
            - PIL: PIL version.
    """
    env_info = {}
    env_info["sys.platform"] = sys.platform
    env_info["Python"] = sys.version.replace("\n", "")

    import torch

    cuda_available = torch.cuda.is_available()
    env_info["CUDA available"] = cuda_available

    if cuda_available:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, device_ids in devices.items():
            env_info["GPU " + ",".join(device_ids)] = name

        from torch.utils.cpp_extension import CUDA_HOME

        env_info["CUDA_HOME"] = CUDA_HOME

        if CUDA_HOME is not None and osp.isdir(CUDA_HOME):
            try:
                nvcc = osp.join(CUDA_HOME, "bin/nvcc")
                nvcc = subprocess.check_output(f'"{nvcc}" -V | tail -n1', shell=True)
                nvcc = nvcc.decode("utf-8").strip()
            except subprocess.SubprocessError:
                nvcc = "Not Available"
            env_info["NVCC"] = nvcc

        env_info["Num of GPUs"] = torch.cuda.device_count()
    else:
        env_info["Num of GPUs"] = 0
    try:
        gcc = subprocess.check_output("gcc --version | head -n1", shell=True)
        gcc = gcc.decode("utf-8").strip()
        env_info["GCC"] = gcc
    except subprocess.CalledProcessError:  # gcc is unavailable
        env_info["GCC"] = "n/a"

    env_info["PyTorch"] = torch.__version__
    env_info["PyTorch compiling details"] = torch.__config__.show()

    try:
        import torchvision

        env_info["TorchVision"] = torchvision.__version__
    except ModuleNotFoundError:
        pass

    env_info["OpenCV"] = cv2.__version__

    try:
        import maniskill2_learn

        env_info["maniskill2_learn"] = maniskill2_learn.__version__
    except ModuleNotFoundError:
        pass

    return env_info


def collect_env():
    """Collect information from system environments."""
    env_info = collect_base_env()
    return env_info


def get_package_meta(package_name):
    package = import_module(package_name)
    try:
        version = package.__version__
    except:
        version = None

    ret = []
    if version is not None:
        ret.append(f"version: {version}")
    return ", ".join(ret)


def get_meta_info():
    ret = {"meta_collect_time": time.strftime("%Y-%m-%d-%H:%M:%S", time.gmtime(time.time() - 7 * 3600))}  # CA time
    for print_name, package_name in [
        ["PYRL", "maniskill2_learn"],
        ["ManiSkill", "mani_skill"],
        ["ManiSkill-Callback", "maniskill"],
        ["ManiSkill2", "mani_skill2"],
    ]:
        try:
            info_i = get_package_meta(package_name)
            ret[print_name] = info_i
        except:
            pass
    return ret


def log_meta_info(logger, meta_info=None):
    if meta_info is None:
        meta_info = get_meta_info()
    for key in meta_info:
        logger.info(f"{key}: {meta_info[key]}")


if __name__ == "__main__":
    for name, val in collect_env().items():
        print(f"{name}: {val}")
