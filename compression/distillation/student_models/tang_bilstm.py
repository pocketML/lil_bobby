import torch
import torch.nn as nn
from bpemb import BPEmb
import compression.distillation.student_models.glue_bilstm as glue
from compression.distillation.student_models.base import StudentModel, StudentConfig

# mix implemention of https://arxiv.org/pdf/1903.12136.pdf
#  but with bytepair embeddings instead of the humongously
#  sized word2vec GoogleNews pre-trained word embeddings yay
class TangBILSTM(StudentModel):
    def __init__(self, task, use_gpu):
        super().__init__(task, use_gpu)
        self.cfg.batch_size = 50

        # embedding
        self.bpe = BPEmb(lang="en", dim=self.cfg.embedding_dim, vs=self.cfg.vocab_size, add_pad_emb=True)
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(self.bpe.vectors))
        
        # encoding
        self.bilstm = glue.get_lstm(self.cfg)

        # classifier/mlp
        inp_d = self.cfg.enc_hidden_dim * 4 if self.cfg.use_sentence_pairs else self.cfg.enc_hidden_dim
        inp_d = inp_d * 2 if self.cfg.bidirectional else inp_d
        self.classifier = nn.Sequential(
            nn.Linear(inp_d, self.cfg.cls_hidden_dim),
            nn.ReLU(), 
            nn.Linear(self.cfg.cls_hidden_dim, self.cfg.num_classes))

    def forward(self, x, lens):
        def embed_encode_sents(sents, lengths):
            #embedding
            emb = self.embedding(sents).float()
            # encoding
            h = glue.pack_bilstm_unpack(self.bilstm, self.cfg, emb, lengths)
            return glue.choose_hidden_state(h, lens=lengths, decision='last')

        if not self.cfg.use_sentence_pairs:
            x = embed_encode_sents(x, lens)
        else:
            x1 = embed_encode_sents(x[0], lens[0])
            x2 = embed_encode_sents(x[1], lens[1])
            x = glue.cat_cmp(x1, x2)
        # classification
        return self.classifier(x)
