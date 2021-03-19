import torch
import torch.nn as nn
from bpemb import BPEmb
from common.task_utils import TASK_LABEL_DICT
from compression.distillation.student_models.glue_bilstm import ( 
    pack_bilstm_unpack, 
    BILSTMConfig, 
    cat_cmp, 
    get_lstm,
    choose_hidden_state)

# mix implemention of https://arxiv.org/pdf/1903.12136.pdf
#  but with bytepair embeddings instead of the humongously
#  sized word2vec GoogleNews pre-trained word embeddings yay
class TangBILSTM(nn.Module):
    def __init__(self, task):
        super().__init__()
        self.label_dict = TASK_LABEL_DICT[task]
        self.cfg = BILSTMConfig(task)

        # embedding
        self.bpe = BPEmb(lang="en", dim=self.cfg.embedding_dim, vs=self.cfg.vocab_size, add_pad_emb=True)
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(self.bpe.vectors))
        
        # encoding
        self.bilstm = get_lstm(self.cfg)

        # classifier/mlp
        inp_d = self.cfg.enc_hidden_dim * 8 if self.cfg.use_sentence_pairs else self.cfg.enc_hidden_dim * 2
        self.cls = nn.Sequential(
            nn.Linear(inp_d, self.cfg.cls_hidden_dim),
            nn.ReLU(), 
            nn.Linear(self.cfg.cls_hidden_dim, self.cfg.num_classes))

    def forward(self, x, lens):
        if not self.cfg.use_sentence_pairs:
            #embedding
            emb = self.embedding(x).float()
            # encoding
            h = pack_bilstm_unpack(self.bilstm, self.cfg, emb, lens)          
            x = choose_hidden_state(h, lens=lens, decision='last')
        else:
            # embedding
            emb1, emb2 = self.embedding(x[0]).float(), self.embedding(x[1]).float()            
            lens1, lens2 = lens[0], lens[1]
            # encoding
            h1 = pack_bilstm_unpack(self.bilstm, self.cfg, emb1, lens1)
            h2 = pack_bilstm_unpack(self.bilstm, self.cfg, emb2, lens2)
            h_n1 = choose_hidden_state(h1, lens=lens1, decision='last')
            h_n2 = choose_hidden_state(h2, lens=lens2, decision='last')
            x = cat_cmp(h_n1, h_n2)
        # classification
        x = self.cls(x)
        return x
