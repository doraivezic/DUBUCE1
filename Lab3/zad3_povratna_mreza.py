#Vaš osnovni model RNN ćelije bi trebao biti jednosmjeran i imati dva sloja
#rnn(150) -> rnn(150) -> fc(150, 150) -> ReLU() -> fc(150,1)


import torch
import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(self, embedding_matrix, num_layers=2, dim1=300, dim2=150):
        super(RNNClassifier, self).__init__()

        self.embedding = embedding_matrix

        self.rnn12 = nn.RNN(dim1, dim2, num_layers=num_layers, batch_first=True, bidirectional=False)   #batch_first – If True, then the input and output tensors are provided as (batch, seq, feature) -> da ne moramo transponirati u forwardu
        
        self.fc1 = nn.Linear(dim2, dim2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim2, 1)
        
    def forward(self, text):

        #text je dim: 10 x 34 
        # (10 zbog velicine batch-a) (34 jer je to duljina najduljeg niza u trenutnom batchu, tj on ima 34 rijeci)
        # Svaka instanca je paddana tako da bude duljine 34 (to smo napravili pomocu DataLoadera i njegove pad_collate_fn) 

        embedded = self.embedding(text)
        #embedded je dim: 10 x 34 x 300 (jer je svaka rijec pretvorena u vektor od 300 elemenata)
        
        x, _ = self.rnn12(embedded)

        x = self.fc1(x[:, -1, :])
        x = self.relu(x)
        x = self.fc2(x)
        return torch.squeeze(x, dim=1)




