import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from common.task_utils import TASK_INFO, TASK_LABEL_DICT
from bpemb import BPEmb

class BILSTMConfig():
    def __init__(self,
        task,
        use_gpu, 
        batch_size=50, 
        enc_hidden_dim=300, 
        bidirectional=True,
        embedding_dim=25,
        vocab_size=50000,
        num_layers=1,
        batch_first=True,
        dropout=0.2,
        cls_hidden_dim=512,
        ):

        self.num_classes = TASK_INFO[task]['settings']['num-classes']
        self.use_sentence_pairs = TASK_INFO[task]['settings']['use-sentence-pairs']
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.enc_hidden_dim = enc_hidden_dim
        self.bidirectional = bidirectional
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.cls_hidden_dim = cls_hidden_dim
        self.batch_first = batch_first
        self.dropout = dropout

# combines distillation loss function with label loss function
def get_loss_function(alpha, criterion_distill, criterion_label, device=torch.device('cuda')):
    beta = 1 - alpha
    criterion_distill.to(device)
    criterion_label.to(device)
    def loss(pred_logits, target_logits, target_label):
        distill_loss = beta * criterion_distill(pred_logits, target_logits)
        label_loss = alpha * criterion_label(pred_logits, target_label)
        return distill_loss + label_loss
    return loss

# returns the last hidden state (both fw and bw) for each embedded sentence
def pack_bilstm_unpack(bilstm, cfg, embedded, lens):
    def init_hidden():
        h = torch.zeros(2, cfg.batch_size, cfg.enc_hidden_dim)
        c = torch.zeros(2, cfg.batch_size, cfg.enc_hidden_dim)
        if cfg.use_gpu:
            h = h.cuda()
            c = c.cuda()
        return (h, c)

    packed = pack_padded_sequence(embedded, lens, batch_first=cfg.batch_first)
    out, _ = bilstm(packed, init_hidden())
    unpacked, _ = pad_packed_sequence(out, batch_first=cfg.batch_first)
    return unpacked

def cat_cmp(inp1, inp2):
    return torch.cat([inp1, inp2, torch.abs(inp1 - inp2), inp1 * inp2], 1)

def get_lstm(cfg):
    return nn.LSTM(
            batch_first=cfg.batch_first,
            input_size=cfg.embedding_dim,
            hidden_size=cfg.enc_hidden_dim,
            num_layers=cfg.num_layers,
            bidirectional=cfg.bidirectional,
        )

def choose_hidden_state(hidden_states, lens=None, decision='max'):
    if decision == 'max':
        h, _ =  hidden_states.max(dim=1)
        return h
    elif decision == 'last':
        batch_size = hidden_states.shape[0]
        batch_idx = torch.LongTensor([i for i in range(batch_size)])
        return hidden_states[batch_idx, torch.LongTensor(lens) - 1, :] 
    else:
        raise Exception(f'decision {decision} not recognized')

# Model inspired by https://openreview.net/pdf?id=rJ4km2R5t7
# https://github.com/nyu-mll/GLUE-baselines/tree/master/src
class GlueBILSTM(nn.Module):
    def __init__(self, task, use_gpu):
        super().__init__()
        self.label_dict = TASK_LABEL_DICT[task]
        self.cfg = BILSTMConfig(task, use_gpu)
        self.cfg.num_layers = 1
        self.cfg.enc_hidden_dim = 1500
        self.cfg.batch_size = 128
        self.dropout = nn.Dropout(p=self.cfg.dropout) if self.cfg.dropout else lambda x: x

        # embedding
        self.bpe = BPEmb(lang="en", dim=self.cfg.embedding_dim, vs=self.cfg.vocab_size, add_pad_emb=True)
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(self.bpe.vectors))
        
        # encoding
        self.bilstm = get_lstm(self.cfg)

        # classifier/mlp
        inp_d = self.cfg.enc_hidden_dim * 4 if self.cfg.use_sentence_pairs else self.cfg.enc_hidden_dim
        inp_d = inp_d * 2 if self.cfg.bidirectional else inp_d
        self.classifier = nn.Sequential(
            nn.Linear(inp_d, self.cfg.cls_hidden_dim),
            nn.Tanh(), 
            nn.Linear(self.cfg.cls_hidden_dim, self.cfg.num_classes))

    def forward(self, x, lens):
        def embed_enc_sents(sents, lengths):
            #embedding
            emb = self.embedding(sents).float()
            emb = self.dropout(emb)
            # encoding
            h = pack_bilstm_unpack(self.bilstm, self.cfg, emb, lengths)
            h = self.dropout(h)
            return choose_hidden_state(h, decision='max')
        
        if not self.cfg.use_sentence_pairs: 
            x = embed_enc_sents(x, lens)
        else:
            x1 = embed_enc_sents(x[0], lens[0])
            x2 = embed_enc_sents(x[1], lens[1])
            x = cat_cmp(x1, x2)
        # classifier
        x = self.classifier(x)
        return x
