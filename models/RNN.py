import torch
from torch import nn


class RNN(nn.module):
    """
    Vanilla RNN
    """
    def __init__(self, D_in, H, D_out, L=1, nonlinearity='tanh',
                 dropout=0.0, device=None):
        """
        D_in: input feature count
        H: hidden state feature count
        D_out: output feature count
        L: layer count
        nonlinearity: nonlinearity to use (tanh or relu)
        dropout: dropout probability
        device: tensor device
        """
        super(RNN, self).__init__()
        self.L = L
        self.H = H
        self.device = device

        dropout = dropout if L > 1 else 0

        self.rnn = nn.RNN(input_size=D_in, hidden_size=self.H,
                          num_layers=self.L, nonlinearity=nonlinearity,
                          dropout=dropout)
        self.fc1 = nn.Linear(self.H, 2*self.H)
        self.relu = nn.ReLU()
        self.fcout = nn.Linear(2*self.H, D_out)

    def forward(self, X):
        """
        X (batch_size, T, D_in): input minibatch

        returns
        y_pred (batch_size, D_out): output prediction
        """
        X = X.permute(1, 0, 2)
        T, batch_size, D_in = X.size()
        hidden = X.new_zeros((self.L, batch_size, self.H),
                             dtype=torch.float,
                             device=self.device)

        for t in range(T):
            self.rnn.flatten_parameters()
            _, hidden = self.rnn(X[t].unsqueeze(0), hidden)

        y_pred = self.fc1(hidden.squeeze(0))
        y_pred = self.relu(y_pred)
        y_pred = self.fcout(y_pred)
        return y_pred
