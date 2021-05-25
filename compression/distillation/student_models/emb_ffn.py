import torch
import torch.nn as nn

from compression.distillation.student_models import base
from embedding import embeddings

class EmbFFN(base.StudentModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.embedding = embeddings.get_embedding(cfg)

        inp_d = self.cfg['embedding-dim'] * 4 if self.cfg['use-sentence-pairs'] else self.cfg['embedding-dim']
        self.classifier = nn.Sequential(
            nn.Dropout(cfg['dropout']),
            nn.Linear(inp_d, cfg['cls-hidden-dim']),
            nn.Dropout(cfg['dropout']),
            nn.ReLU(),
            nn.Linear(cfg['cls-hidden-dim'], cfg['num-classes'])
        )
        self.init_weights(embedding_init_range=0.1, classifier_init_range=0.1)

    def mean_with_lens(self, x, lens):
        if self.cfg['use-gpu']:
            lens = lens.cuda()
        idx = torch.arange(x.shape[0]) # assumes batch first
        x = x.cumsum(1)[idx, lens - 1, :] # cumulative sum over the channels, dim=1, select the sum for the len of the input
        x = x / lens.view(-1, 1)
        return x

    def forward(self, x, lens):
        if self.cfg['use-sentence-pairs']:
            x1, x2 = self.embedding(x[0]), self.embedding(x[1])
            x1 = self.mean_with_lens(x1, lens[0])
            x2 = self.mean_with_lens(x2, lens[1])
            x = base.cat_cmp(x1, x2)
        else:
            x = self.embedding(x)
            x = self.mean_with_lens(x, lens)
        x = self.classifier(x)
        return x

    def get_optimizer(self):
        warmup_start_lr = self.cfg['lr'] / 100
        base_optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.cfg['lr']     
        )
        optimizer = base.WarmupOptimizer(
            base_optimizer, 
            warmup_steps=100,
            final_lr=self.cfg['lr'],
            start_lr=warmup_start_lr
        )
        return optimizer

    # combines distillation loss function with label loss function
    # weighted with (1 - alpha) and alpha respectively
    def get_combined_loss_function(self, alpha, criterion_distill, criterion_label, device, temperature=3):
        beta = 1 - alpha
        criterion_distill.to(device)
        criterion_label.to(device)
        temperature = 1.0/temperature
        def loss(pred_logits, target_logits, target_label):
            pred_temp = nn.functional.softmax(pred_logits * temperature, dim=1)
            target_temp = nn.functional.softmax(target_logits * temperature, dim=1)
            distill_loss = beta * criterion_distill(pred_temp, target_temp)
            label_loss = alpha * criterion_label(pred_logits, target_label)
            return distill_loss + label_loss
        return loss