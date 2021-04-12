from abc import abstractmethod
import json
import torch
from torch import nn
from torch.optim import Adam
from common.task_utils import TASK_LABEL_DICT, TASK_INFO
from common.model_utils import get_model_path
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def update_student_config_from_file(cfg, path):
    with open(path, 'r') as f:
        loaded = json.load(f)
        cfg.update(loaded)
    return cfg

def get_default_student_config(task, arch, model_name=None, use_gpu=True):
    # get base config
    base_path = f'compression/distillation/student_models/configs/base.json'
    cfg = update_student_config_from_file({}, base_path)
    cfg['task'] = task
    cfg['num-classes'] = TASK_INFO[task]['settings']['num-classes']
    cfg['use-sentence-pairs'] = TASK_INFO[task]['settings']['use-sentence-pairs']

    # update with base student model config settings
    filepath = f'compression/distillation/student_models/configs/{arch}.json'
    cfg = update_student_config_from_file(cfg, filepath)

    # update with saved model config settings
    if model_name is not None:
        model_path = get_model_path(task, "distilled")
        cfg = update_student_config_from_file(cfg, f'{model_path}/{model_name}.json')

    return cfg

class StudentModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.label_dict = TASK_LABEL_DICT[cfg['task']]

    def get_optimizer(self):
        return Adam(
            self.parameters(), lr=self.cfg['lr'],
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

    def encode(self, sentence):
        return self.bpe.encode_ids(sentence)

# combines distillation loss function with label loss function
def get_dist_loss_function(alpha, criterion_distill, criterion_label, temperature=1.0, device=torch.device('cuda')):
    beta = 1 - alpha
    criterion_distill.to(device)
    criterion_label.to(device)
    temperature = 1.0 / temperature
    def loss(pred_logits, target_logits, target_label):
        # assuming input is batch x logits
        pred_temp = nn.functional.softmax(pred_logits * temperature, dim=1)
        target_temp = nn.functional.softmax(target_logits * temperature, dim=1)
        distill_loss = beta * criterion_distill(pred_temp, target_temp)
        label_loss = alpha * criterion_label(pred_logits, target_label)
        return distill_loss + label_loss
    return loss

# returns the last hidden state (both fw and bw) for each embedded sentence
def pack_bilstm_unpack(bilstm, cfg, embedded, lens, batch_size):
    def init_hidden():
        h = torch.zeros(2, batch_size, cfg['encoder-hidden-dim'])
        c = torch.zeros(2, batch_size, cfg['encoder-hidden-dim'])
        if cfg['use-gpu']:
            h = h.cuda()
            c = c.cuda()
        return (h, c)

    packed = pack_padded_sequence(embedded, lens, batch_first=cfg['batch-first'])
    out, _ = bilstm(packed, init_hidden())
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
