import torch
from torch import nn


class LSTM(nn.Module):
    """
    Long Short-Term Memory Model
    """
    def __init__(self, input_dim, hidden_dim, output_dim=10, 
                 num_layers=2):
        """
        input_dim: input feature count
        hidden_dim: hidden state feature count
        output_dim: output feature count
        num_layers: layer count
        batch_size: size of batch
        """
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        #LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        #output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, input_seq):
        """
        input_seq (T, batch_size, input_dim): input minibatch
        lstm_out (input_seq_size, batch_size, hidden_dim): 

        returns
        y_pred (batch_size, output_dim): output prediction
        """
        input_seq = input_seq.permute(1, 0, 2)
        lstm_out, self.hidden = self.lstm(input_seq.view(len(input_seq), len(input_seq[1]), -1))
        y_pred = self.linear(lstm_out[-1].view(len(input_seq[1]), -1))
        return y_pred