from torch.nn.modules import dropout
from zad1_Instance import Instance
from zad1_NLPDataset import NLPDataset
from zad1_Vocab import Vocab
from zad1_Vocab import embedding_matrica

from torch.nn import Embedding 
from torch.utils.data import DataLoader


all_instances = Instance("sst_train_raw.csv")
all_instances.frequencies()




text_vocab = Vocab(all_instances.text_frequencies, max_size=-1, min_freq=0)
text_just_vocab = text_vocab.stoi()

glove = True
embedding_matrix = embedding_matrica(text_just_vocab, use_glove=glove)
embedding_matrix = Embedding.from_pretrained(embedding_matrix, freeze=glove, padding_idx=0) #kako bi vašu matricu spremili u optimizirani omotač za vektorske reprezentacije
                                                                        #freeze stavljamo na True samo ako koristimo glove (txt file)

numercalized_text = text_vocab.encode(all_instances.reviews[0][0])
# print(all_instances.reviews[0][0])
# print(numercalized_text)


label_vocab = Vocab(all_instances.label_frequencies, max_size=-1, min_freq=0, labels=True)
label_just_vocab = label_vocab.stoi()

numercalized_labels = label_vocab.encode(all_instances.reviews[0][1])
# print(all_instances.reviews[0][1])
# print(numercalized_labels)




train_dataset = NLPDataset("sst_train_raw.csv", text_vocab, label_vocab)
numercalized_text, numercalized_labels = train_dataset[0]
#print(numercalized_text, numercalized_labels)


batch_size = 10
shuffle = True
from zad1_method_collate import pad_collate_fn

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                              shuffle=shuffle, collate_fn=pad_collate_fn)
texts, labels, lengths = next(iter(train_dataloader))
# print(texts)
# print(labels)
# print(lengths)


validation_dataset = NLPDataset("sst_valid_raw.csv", text_vocab, label_vocab)
validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=32, 
                              shuffle=False, collate_fn=pad_collate_fn)



#zad2
print("\nZADATAK 2\n")
import torch
import torch.nn as nn
import torch.optim as optim
from zad2_baseline_model import BaselineClassifier, train, evaluate

torch.manual_seed(123)

num_epochs = 5
lr = 1e-4

model = BaselineClassifier(embedding_matrix)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):

    cost, accuracy, f1, conf_matrix = train(model=model, data=train_dataloader, optimizer=optimizer, criterion=criterion)

    cost, accuracy, f1, conf_matrix = evaluate(model=model, data=validation_dataloader, criterion=criterion)

    print(f"Epoch {epoch+1} : accuracy {accuracy}, F1-score {f1}, confusion matrix {conf_matrix}")



test_dataset = NLPDataset("sst_test_raw.csv", text_vocab, label_vocab)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, 
                              shuffle=False, collate_fn=pad_collate_fn)

cost, accuracy, f1, conf_matrix = evaluate(model=model, data=test_dataloader, criterion=criterion)
print(f"Test dataset : accuracy {accuracy}, F1-score {f1}, confusion matrix {conf_matrix}")




#zad3
print("\nZADATAK 3\n")
from zad3_povratna_mreza import RNNClassifier

num_epochs = 5
lr = 1e-4

model = RNNClassifier(embedding_matrix)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):

    cost, accuracy, f1, conf_matrix = train(model=model, data=train_dataloader, optimizer=optimizer, criterion=criterion)

    cost, accuracy, f1, conf_matrix = evaluate(model=model, data=validation_dataloader, criterion=criterion)

    print(f"Epoch {epoch+1} : accuracy {accuracy}, F1-score {f1}, confusion matrix {conf_matrix}")

cost, accuracy, f1, conf_matrix = evaluate(model=model, data=test_dataloader, criterion=criterion)
print(f"Test dataset : accuracy {accuracy}, F1-score {f1}, confusion matrix {conf_matrix}")




#zad4
print("\nZADATAK 4\n")
from zad4_usporedba_modela import ModelComparison

num_epochs = 5
lr = 1e-4

modeli = ["rnn", "gru", "lstm"]
ispisi = ["\nMODEL VANILLA RNN\n", "\nMODEL GRU\n", "\nMODEL LSTM\n"]

for i in range(len(modeli)):

    print(ispisi[i])

    model = ModelComparison(embedding_matrix, rnn_type=modeli[i], dropout_prob=0.1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):

        cost, accuracy, f1, conf_matrix = train(model=model, data=train_dataloader, optimizer=optimizer, criterion=criterion)

        cost, accuracy, f1, conf_matrix = evaluate(model=model, data=validation_dataloader, criterion=criterion)

        print(f"Epoch {epoch+1} : accuracy {accuracy}, F1-score {f1}, cost {cost}, confusion matrix {conf_matrix}")

    cost, accuracy, f1, conf_matrix = evaluate(model=model, data=test_dataloader, criterion=criterion)
    print(f"Test dataset : accuracy {accuracy}, F1-score {f1}, cost {cost}, confusion matrix {conf_matrix}")
