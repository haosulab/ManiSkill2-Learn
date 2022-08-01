import numpy as np
import torch
import torch.nn as nn

import time

ln_time = 0.0
ln_ln_time = 0.0
ln_permute1_time = 0.0
ln_permute2_time = 0.0
custom_ln_time = 0.0
bn_time = 0.0

T = 8
device = 'cuda:0'
size=(16,256,1024,64)
weight = torch.rand(size[1]).to(device)
bias = torch.randn(size[1]).to(device)

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-8, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(weight.clone())
        self.bias = nn.Parameter(bias.clone())
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


ln = nn.LayerNorm(size[1], eps=1e-8).to(device)
ln.weight = nn.Parameter(weight.clone())
ln.bias = nn.Parameter(bias.clone())
ln_custom = LayerNorm(size[1], data_format="channels_first").to(device)

a = torch.randn(size).to(device)
b = a.permute(0,2,3,1).contiguous()
ln_a = ln(b)
ln_a = ln_a.permute(0,3,1,2).contiguous()
ln_custom_a = ln_custom(a)
print("|ln - ln_custom| max", torch.max(torch.abs(ln_a - ln_custom_a)))

del a,b



ln_custom = LayerNorm(size[1], data_format="channels_first").to(device)
for t in range(T):
    print(t)
    a = torch.randn(size).to(device)
    torch.cuda.synchronize()
    tt = time.time()
    b=ln_custom(a)
    torch.cuda.synchronize()
    custom_ln_time += time.time() - tt
    del b
print("custom_ln_time", custom_ln_time / T)

ln = nn.LayerNorm(size[1]).to(device)
ln.weight = nn.Parameter(weight.clone())
ln.bias = nn.Parameter(bias.clone())
for t in range(T):
    print(t)
    a = torch.randn(size).to(device)
    torch.cuda.synchronize()

    tt = time.time()
    b=a.permute(0,2,3,1)
    torch.cuda.synchronize()
    ln_permute1_time += time.time() - tt

    tt=time.time()
    b=ln(b)
    torch.cuda.synchronize()
    ln_ln_time += time.time() - tt

    tt=time.time()
    b=b.permute(0,3,1,2)
    torch.cuda.synchronize()
    ln_permute2_time += time.time() - tt
    del b

print("pytorch ln_ln_time", ln_ln_time / T, "ln_permute1_time", ln_permute1_time / T, "ln_permute2_time", ln_permute2_time / T)
print("pytorch ln_total_time", (ln_ln_time + ln_permute1_time + ln_permute2_time) / T)

# bn = nn.BatchNorm2d(size[1]).to(device)
# for t in range(T):
#     print(t)
#     a = torch.randn(size).to(device)
#     torch.cuda.synchronize()
#     tt = time.time()
#     b=bn(a)
#     torch.cuda.synchronize()
#     bn_time += time.time() - tt  
#     del b  
# print("bn_time", bn_time / T)







