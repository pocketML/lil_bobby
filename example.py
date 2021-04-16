import compression.distillation.student_models.char_rnn as rnn
import torch

model = rnn.CharRNN({'task':'sst-2','use-gpu': True})
input = ["Hej  ole!! ", "Sikke døjlig vejr i dag :DDD"]
out, hidden = model(input, [len(input)])
pred = torch.argmax(out, dim=1)
print(out)
print(pred)