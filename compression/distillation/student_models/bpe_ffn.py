import torch
import torch.nn as nn
from bpemb import BPEmb
from compression.distillation.student_models.base import StudentModel, StudentConfig

class BPE_FFN(StudentModel):
    def __init__(self, task, use_gpu):
        super().__init__(task, use_gpu)
        self.cfg = StudentConfig(
            task,
            use_gpu,
            embedding_dim=25,
            vocab_size=1000,
            dropout=0.3,
            lr=1e-3,
            weight_decay=1e-5
        )

        self.max_seq_len = 100

        # embedding
        self.bpe = BPEmb(
            lang="en", dim=self.cfg.embedding_dim,
            vs=self.cfg.vocab_size, add_pad_emb=True
        )
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(self.bpe.vectors))

        self.dropout_1 = nn.Dropout(p=self.cfg.dropout) if self.cfg.dropout else lambda x: x

        self.avg_pool = nn.AvgPool1d(2)

        self.hidden_units = 64

        self.dense_1 = nn.Linear(self.max_seq_len * (self.cfg.embedding_dim // 2), self.hidden_units)
        self.relu_1 = nn.ReLU()
        self.dropout_2 = nn.Dropout(p=self.cfg.dropout)
        self.dense_2 = nn.Linear(self.hidden_units, self.hidden_units)
        self.relu_2 = nn.ReLU()
        self.dropout_3 = nn.Dropout(p=self.cfg.dropout)

        self.classifier = nn.Linear(self.hidden_units, self.cfg.num_classes)

    def forward(self, sents, lens):
        emb = self.embedding(sents).float()

        pad_amount = self.max_seq_len - emb.shape[1]

        # Pad 2nd dimension to match max_seq_len.
        emb = torch.nn.functional.pad(emb, pad=(0, 0, 0, pad_amount, 0, 0), mode="constant")

        x = self.dropout_1(emb)

        x = self.avg_pool(x)

        x = x.view(self.cfg.batch_size, -1)

        x = self.dense_1(x)
        x = self.relu_1(x)
        x = self.dropout_2(x)

        x = self.dense_2(x)
        x = self.relu_2(x)
        x = self.dropout_3(x)

        return self.classifier(x)
