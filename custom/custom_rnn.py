import torch.nn as nn

def apply_permutation(tensor, permutation, dim: int = 1):
    return tensor.index_select(dim, permutation)

# only single layer
class QuantizableRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True, bidirectional=True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_cell = nn.RNNCell(self.input_size, self.hidden_size)

    # input needs to be a packed sequence
    def forward(self, input, hx):
        input, batch_sizes, sorted_indices, unsorted_indices = input
        max_batch_size = int(batch_sizes[0])

        hx = apply_permutation(hx, sorted_indices)
