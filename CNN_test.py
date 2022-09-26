import torch
import torch.nn as nn

inputs = torch.randn(
    1,5,100,100
)

conv = nn.Conv2d(5,10,(3,3))
print(inputs.shape)
out = conv(inputs)
print(out.shape)
print(conv.weight.shape)