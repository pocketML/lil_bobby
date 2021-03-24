from numpy.core.fromnumeric import squeeze
import torch
import torch.nn as nn
from bpemb import BPEmb
from common.task_utils import TASK_INFO, TASK_LABEL_DICT
from compression.distillation.student_models.glue_bilstm import StudentConfig

class BPE_FFN(nn.Module):
    def __init__(self, task, use_gpu):
        super().__init__()
        self.label_dict = TASK_LABEL_DICT[task]
        self.cfg = StudentConfig(
            task,
            use_gpu,
            embedding_dim=25,
            vocab_size=5000
        )

        self.max_seq_len = 150

        # embedding
        self.bpe = BPEmb(
            lang="en", dim=self.cfg.embedding_dim,
            vs=self.cfg.vocab_size, add_pad_emb=True
        )
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(self.bpe.vectors))

        self.avg_pool = nn.AvgPool1d(2)

        self.hidden_units = 500

        self.classifier = nn.Sequential(
            nn.Linear(self.max_seq_len * self.cfg.embedding_dim // 2, self.hidden_units),
            nn.Linear(self.hidden_units, self.cfg.num_classes)
        )

    def forward(self, sents, _):
        emb = self.embedding(sents).float()

        x = self.avg_pool(emb)

        x = x.view(self.cfg.batch_size, -1)

        return self.classifier(x)
