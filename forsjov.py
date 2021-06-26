import torch.nn as nn
from functools import reduce

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(1100, 300)
        self.bilstm = nn.LSTM(300, 150, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(300, 200),
            nn.Linear(200, 2)
        )

    def forward(self, x):
        return x

model = Model()
combined = 0
for name,param in model.named_parameters():
    size = list(param.size())
    combined += reduce(lambda acc, x: acc * x, size, 1)
    print(name, size)

print(f'combined: {combined}')