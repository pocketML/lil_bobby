import torch
import torch.nn as nn

class Model(nn.Module):
  def __init__(self):
    super().__init__()

    self.emb = nn.Embedding(1,300)
    self.bilstm = nn.LSTM(300, 1500, 2, bidirectional=True)
    self.cls = nn.Sequential( # there fancy MLP version
      nn.Linear(1500, 512),
      nn.Linear(512, 512),
      nn.Linear(512, 2)
    )

  def forward(self, x):
    return self.bilstm(x)

