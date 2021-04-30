import torch
import torch.nn as nn
from torch.optim.adam import Adam
from embedding import embeddings
from compression.distillation.student_models.base import StudentModel

class WASSERBLAT_FFN(StudentModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.max_seq_len = 150

        # embedding
        self.embedding = embeddings.get_embedding(cfg)

        self.dropout1 = nn.Dropout(p=self.cfg['dropout']) if self.cfg['dropout'] else lambda x: x

        self.avg_pool = nn.AvgPool1d(2)

        self.classfier = nn.Sequential(
            nn.Dropout(p=self.cfg['dropout']),
            nn.Linear(self.cfg["embedding-dim"], self.cfg['cls-hidden-dim']),
            nn.Dropout(p=self.cfg['dropout']),
            nn.ReLU(),
            nn.Linear(self.cfg['cls-hidden-dim'], self.cfg['num-classes'])
        )

    def forward(self, x, lens):
        emb = self.embedding(x).float()

        # pad_amount = self.max_seq_len - emb.shape[1]

        # # Pad 2nd dimension to match max_seq_len.
        # x = torch.nn.functional.pad(emb, pad=(0, 0, 0, pad_amount, 0, 0), mode="constant")

        #x = self.dropout1(emb)

        #x = self.avg_pool(emb)

        x = torch.mean(emb, dim=1)

        x = self.classfier(x)
        return x
