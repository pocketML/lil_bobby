import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam

from abc import abstractmethod
import json
import numpy as np

from common.task_utils import TASK_LABEL_DICT, TASK_INFO
from common.model_utils import get_model_path

class StudentModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.label_dict = TASK_LABEL_DICT[cfg['task']]

    def get_optimizer(self):
        return Adam(
            self.parameters(), 
            lr=self.cfg['lr'],
            weight_decay=self.cfg['weight-decay']
        )

    @abstractmethod
    def forward(self, sents, lens):
        pass

    def save(self, model_name):
        model_path = get_model_path(self.cfg['task'], "distilled")
        torch.save(self.state_dict(), f"{model_path}/{model_name}.pt")
        with open(f'{model_path}/{model_name}.json', 'w') as out:
            json.dump(self.cfg, out)

    # requires that the trained model dict shares the same config values as this one
    def load(self, model_name):
        model_path = get_model_path(self.cfg['task'], "distilled")
        self.load_state_dict(torch.load(f"{model_path}/{model_name}.pt"))
        self.eval()

    def non_embedding_params(self):
        params = []
        for m in self.modules():
            if isinstance(m, nn.Embedding) or isinstance(m, nn.EmbeddingBag):
                continue
            params.extend(p for p in m.parameters() if p.dim() == 2)
        return params

    def init_weights(self, embedding_init_range=None, classifier_init_range=None):
        if not self.embedding.load_pretrained and embedding_init_range is not None:
            self.embedding.init_weight_range(embedding_init_range)

        if classifier_init_range is not None:
            for i in range(len(self.classifier)):
                module = self.classifier[i]
                if isinstance(module, nn.Linear):
                    module.bias.data.zero_()
                    module.weight.data.uniform_(-classifier_init_range, classifier_init_range)

class WarmupOptimizer:
    """Optim wrapper that implements rate."""

    def __init__(self, base_optimizer, warmup_steps=100, final_lr=1e-4, start_lr=1e-6):
        self.base_optimizer = base_optimizer
        self.warmup_steps = warmup_steps
        self.rates = np.linspace(start_lr, final_lr, num=warmup_steps)
        self.final_lr = final_lr
        self._step = 0
        self._rate = start_lr

    def step(self):
        """Update parameters and rate"""
        self._rate = self.rates[self._step] if self._step < self.warmup_steps else self.final_lr
        self._step += 1
        for p in self.base_optimizer.param_groups:
            p["lr"] = self._rate
        self.base_optimizer.step()

    def zero_grad(self):
        self.base_optimizer.zero_grad()

def update_student_config_from_file(cfg, path):
    with open(path, 'r') as f:
        loaded = json.load(f)
        cfg.update(loaded)
    return cfg

def get_default_student_config(task, arch, model_name=None):
    # get base config
    base_path = f'compression/distillation/student_models/configs/base.json'
    cfg = update_student_config_from_file({}, base_path)
    cfg['task'] = task
    cfg['num-classes'] = TASK_INFO[task]['settings']['num-classes']
    cfg['use-sentence-pairs'] = TASK_INFO[task]['settings']['use-sentence-pairs']

    # update with base student model config settings
    if arch is not None:
        filepath = f'compression/distillation/student_models/configs/{arch}.json'
        cfg = update_student_config_from_file(cfg, filepath)

    # update with saved model config settings
    if model_name is not None:
        model_path = get_model_path(task, "distilled")
        cfg = update_student_config_from_file(cfg, f'{model_path}/{model_name}.json')

    return cfg

# combines distillation loss function with label loss function
def get_dist_loss_function(alpha, criterion_distill, criterion_label, device, temperature=1.0):
    beta = 1 - alpha
    criterion_distill.to(device)
    criterion_label.to(device)
    def loss(pred_logits, target_logits, target_label):
        # assuming input is batch x logitsk
        distill_loss = beta * criterion_distill(pred_logits, target_logits)
        label_loss = alpha * criterion_label(pred_logits, target_label)
        return distill_loss + label_loss
    return loss

# returns the last hidden state (both fw and bw) for each embedded sentence
def pack_rnn_unpack(rnn, cfg, embedded, lens, batch_size, enforce_sorted=True):
    def init_hidden():
        h = torch.zeros((1 + int(cfg['bidirectional'])) * cfg['num-layers'], batch_size, cfg['encoder-hidden-dim'])
        c = torch.zeros((1 + int(cfg['bidirectional'])) * cfg['num-layers'], batch_size, cfg['encoder-hidden-dim'])
        if cfg['use-gpu']:
            h = h.cuda()
            c = c.cuda()
        return (h, c)

    packed = pack_padded_sequence(embedded, lens, batch_first=cfg['batch-first'], enforce_sorted=enforce_sorted)
    if cfg['type'] == 'lstm':
        out, _ = rnn(packed, init_hidden())
    else: # we got an rnn
        out, _ = rnn(packed, init_hidden()[0])

    unpacked, _ = pad_packed_sequence(out, batch_first=cfg['batch-first'])
    return unpacked

def cat_cmp(inp1, inp2):
    return torch.cat([inp1, inp2, torch.abs(inp1 - inp2), inp1 * inp2], 1)

def get_lstm(cfg):
    return nn.LSTM(
        batch_first=cfg['batch-first'],
        input_size=cfg['embedding-dim'],
        hidden_size=cfg['encoder-hidden-dim'],
        num_layers=cfg['num-layers'],
        bidirectional=cfg['bidirectional'],
    )

def get_rnn(cfg):
    return nn.RNN(
        batch_first=cfg['batch-first'],
        input_size=cfg['embedding-dim'],
        hidden_size=cfg['encoder-hidden-dim'],
        num_layers=cfg['num-layers'],
        bidirectional=cfg['bidirectional'],
    )

def get_classifier(inp_d, cfg):
    return nn.Sequential(
        nn.Linear(inp_d, cfg['cls-hidden-dim']),
        nn.Dropout(cfg['dropout']),
        nn.ReLU(),
        nn.Linear(cfg['cls-hidden-dim'], cfg['num-classes'])
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
