#Baseline model nam služi za procjenu performansi koje naš stvarni, uobičajeno skuplji model mora moći preći kao plitak potok. 
# Također, baseline modeli će nam pokazati kolika je stvarno cijena izvođenja naprednijih modela.

#avg_pool() -> fc(300, 150) -> ReLU() -> fc(150, 150) -> ReLU() -> fc(150,1)

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


class BaselineClassifier(nn.Module):
    def __init__(self, embedding_matrix, dim1=300, dim2=150, dim3=150):
        super(BaselineClassifier, self).__init__()

        self.embedding = embedding_matrix

        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)
    
        self.fc1 = nn.Linear(dim1, dim2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim2, dim3)
        self.fc3 = nn.Linear(dim3, 1)
        
    def forward(self, text):

        #text je dim: 10 x 34 
        # (10 zbog velicine batch-a) (34 jer je to duljina najduljeg niza u trenutnom batchu, tj on ima 34 rijeci)
        # Svaka instanca je paddana tako da bude duljine 34 (to smo napravili pomocu DataLoadera i njegove pad_collate_fn) 

        embedded = self.embedding(text)
        #embedded je dim: 10 x 34 x 300 (jer je svaka rijec pretvorena u vektor od 300 elemenata)
        embedded = embedded.permute(0,2,1)

        pooled = self.avgpool(embedded) # dim: batch_size x embedding_dim x 1
        pooled = torch.squeeze(pooled, dim=2) # dim: batch_size x embedding_dim

        x = self.fc1(pooled)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return torch.squeeze(x, dim=1)




def calculate_scores(y_real, y_pred):

    conf_matrix = confusion_matrix(y_real, y_pred)
    accuracy = accuracy_score(y_real, y_pred)
    f1 = f1_score(y_real, y_pred)
    
    return accuracy, f1, conf_matrix


def train(model, data, optimizer, criterion):
    model.train()

    cost = 0
    y_pred = []
    y_real = []

    for batch_num, batch in enumerate(data):
        optimizer.zero_grad()
        
        x = batch[0]    #texts
        y = batch[1]    #labels
        #lengths = batch[2]

        logits = model(x)
        loss = criterion(logits, y.float())   #posto koristimo BCELoss ne moramo primjeniti sigmoidu na izlaznim logitima
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)   #Clips gradient norm of an iterable of parameters.
                                                    #The norm is computed over all gradients together, as if they were concatenated into a single vector. Gradients are modified in-place.
        optimizer.step()

        cost += loss.item()

        #prije nego dodamo u predvidene klase moramo primjenti sigmoidu
        logits = torch.sigmoid(logits)
        logits = torch.round(logits).int()

        y_pred.extend(logits.tolist())
        y_real.extend(y.tolist())

    accuracy, f1, conf_matrix = calculate_scores(y_real, y_pred)

    return cost, accuracy, f1, conf_matrix



def evaluate(model, data, criterion):
    model.eval()

    cost = 0
    y_pred = []
    y_real = []

    with torch.no_grad():
        for batch_num, batch in enumerate(data):

            x = batch[0]    #texts
            y = batch[1]    #labels

            logits = model(x)
            loss = criterion(logits, y.float())

            cost += loss.item()
        
        #prije nego dodamo u predvidene klase moramo primjenti sigmoidu
        logits = torch.sigmoid(logits)
        logits = torch.round(logits).int()

        y_pred.extend(logits.tolist())
        y_real.extend(y.tolist())

    accuracy, f1, conf_matrix = calculate_scores(y_real, y_pred)

    return cost, accuracy, f1, conf_matrix
            