from abc import abstractmethod
from torch import nn
from torch.optim import Adam
from common.task_utils import TASK_LABEL_DICT, TASK_INFO

class StudentConfig():
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
        lr=1e-4
        ):

        self.num_classes = TASK_INFO[task]['settings']['num-classes']
        self.use_sentence_pairs = TASK_INFO[task]['settings']['use-sentence-pairs']
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.enc_hidden_dim = enc_hidden_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.lr = lr

        # === Bi-LSTM specific settings ===
        self.bidirectional = bidirectional 
        self.num_layers = num_layers
        self.cls_hidden_dim = cls_hidden_dim
        self.batch_first = batch_first

class StudentModel(nn.Module):
    def __init__(self, task, use_gpu):
        super().__init__()
        self.label_dict = TASK_LABEL_DICT[task]
        self.cfg = StudentConfig(task, use_gpu)

    def get_optimizer(self):
        return Adam(self.parameters(), lr=self.cfg.lr)

    @abstractmethod
    def forward(self, sents, lens):
        pass
