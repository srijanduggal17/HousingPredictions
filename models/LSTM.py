import torch
from torch import nn


class LSTM(nn.module):
    """
    Long Short-Term Memory Model
    """
    def __init__(self, input_dim, hidden_dim, output_dim=1, 
                 num_layers=2, batch_size):
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
        self.batch_size = batch_size
        
        #LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        #output layer
        self.linear = mm.Linear(self.hidden_dim, output_dim)

    def forward(self, input_seq):
        """
        input_seq (batch_size, T, input_dim): input minibatch
        lstm_out (input_seq_size, batch_size, hidden_dim): 

        returns
        y_pred (batch_size, D_out): output prediction
        """
        lstm_out, self.hidden = self.lstm.(input_seq.view(len(input_seq), self.batch_size, -1))
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred.view(-1)
    
    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, model.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, model.hidden_dim))