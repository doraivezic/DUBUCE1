import torch
import torch.nn as nn
from torch.nn.modules import dropout

class ModelComparison(nn.Module):
    def __init__(self, embedding_matrix, rnn_type='rnn', num_layers=2, dim1=300, dim2=150, dropout_prob=0.3):
        super(ModelComparison, self).__init__()

        self.embedding = embedding_matrix

        if rnn_type=="gru":
            self.rnn = nn.GRU(dim1, dim2, num_layers=num_layers, batch_first=True, bidirectional=False, dropout=dropout_prob)   
        elif rnn_type=="lstm":
            self.rnn = nn.LSTM(dim1, dim2, num_layers=num_layers, batch_first=True, bidirectional=False, dropout=dropout_prob)  
        else: #vanilla rnn
            self.rnn = nn.RNN(dim1, dim2, num_layers=num_layers, batch_first=True, bidirectional=False, dropout=dropout_prob)
        

        self.fc1 = nn.Linear(dim2, dim2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim2, 1)
        
    def forward(self, text):

        embedded = self.embedding(text)
        
        x, _ = self.rnn(embedded)

        x = self.fc1(x[:, -1, :])
        x = self.relu(x)
        x = self.fc2(x)
        return torch.squeeze(x, dim=1)