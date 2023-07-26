import torch
import torch.nn as nn

device = 'gpu'


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        x = x.unsqueeze(-1)
        if device=='cpu':
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device='cuda').requires_grad_()
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device='cuda').requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])

        return out
