import compression.distillation.student_models.char_rnn as c
import torch

emb = c.HashEmbedding(3, 10)

sent = "hej med dig"
input = torch.stack([emb.encode(sent)])
print(input)
print(input.shape)
out = emb(input)

print(out)
print(out.shape)