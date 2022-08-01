import os.path as osp
import numbers, numpy as np
import matplotlib.pyplot as plt


class TensorboardLogger:
    def __init__(self, log_dir=None):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(osp.join(log_dir, "tf_logs"))

    def get_lr_tags(self, runner):
        tags = {}
        lrs = runner.current_lr()
        if isinstance(lrs, dict):
            for name, value in lrs.items():
                tags[f"learning_rate/{name}"] = value[0]
        else:
            tags["learning_rate"] = lrs[0]
        return tags

    def get_momentum_tags(self, runner):
        tags = {}
        momentums = runner.current_momentum()
        if isinstance(momentums, dict):
            for name, value in momentums.items():
                tags[f"momentum/{name}"] = value[0]
        else:
            tags["momentum"] = momentums[0]
        return tags

    @staticmethod
    def is_scalar(val, include_np=True, include_torch=True):
        if isinstance(val, numbers.Number):
            return True
        elif include_np and isinstance(val, np.ndarray) and val.ndim == 0:
            return True
        else:
            import torch

            if include_torch and isinstance(val, torch.Tensor) and len(val) == 1:
                return True
            else:
                return False

    def get_loggable_tags(self, output, allow_scalar=True, allow_text=False, tags_to_skip=("time", "data_time"), add_mode=True, tag_name="train"):
        tags = {}
        for tag, val in output.items():
            if tag in tags_to_skip:
                continue
            if self.is_scalar(val) and not allow_scalar:
                continue
            if isinstance(val, str) and not allow_text:
                continue
            if add_mode and "/" not in tag:
                tag = f"{tag_name}/{tag}"
            tags[tag] = val
        return tags

    def log(self, tags, n_iter, tag_name="train"):
        tags = self.get_loggable_tags(tags, tag_name=tag_name)
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, n_iter)
            elif np.isscalar(val) or val.size == 1:
                self.writer.add_scalar(tag, val, n_iter)
            else:
                if val.ndim == 2:
                    cmap = plt.get_cmap('jet')
                    val = cmap(val)[..., :3]
                assert val.ndim == 3, f"Image should have two dimension! You provide: {tag, val.shape}!"
                self.writer.add_image(tag, val, n_iter, dataformats='HWC')

    def close(self):
        self.writer.close()
