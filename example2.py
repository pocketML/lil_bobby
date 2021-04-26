import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(1, stride=1, dim=1)

    def forward(self, x):
        return self.pool(x)

model = Model()
inp1 = torch.randn(50, 23, 16)
inp2 = torch.randn(50, 44, 16)

out1 = model(inp1)
out2 = model(inp2)
print(out1.shape)
print(out2.shape)