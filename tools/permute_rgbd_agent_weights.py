import torch, numpy as np
import argparse

parser = argparse.ArgumentParser(description="Permute RGBD agent weights due to visual key order changes")
parser.add_argument("--model", type=str, required=True, help="Path to the model file")

args = parser.parse_args()

model = torch.load(args.model)
for k in model['state_dict'].keys():
    if k in ['actor.backbone.visual_nn.stem.weight', 'critic.values.0.backbone.visual_nn.stem.weight']:
        rgb_weights = model['state_dict'][k][:, :6, :, :]
        model['state_dict'][k][:, :6, :, :] = torch.cat([rgb_weights[:, 3:6, :, :], rgb_weights[:, :3, :, :]], dim=1)
        depth_weights = model['state_dict'][k][:, 6:8, :, :]
        model['state_dict'][k][:, 6:8, :, :] = torch.cat([depth_weights[:, 1:2, :, :], depth_weights[:, :1, :, :]], dim=1)
torch.save(model, args.model.replace('.pth', '_permuted_visual.pth'))