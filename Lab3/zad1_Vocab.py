#služi za pretvorbu tekstnih podataka u indekse (što zovemo numerikalizacija)

#max_size: maksimalni broj tokena koji se sprema u vokabular (uključuje i posebne znakove). -1 označava da se spremaju svi tokeni.
#min_freq: minimalna frekvencija koju token mora imati da bi ga se spremilo u vokabular (\ge). Posebni znakovi ne prolaze ovu provjeru.

#vokabular se izgrađuje samo na train skupu podataka

import numpy as np
import torch


class Vocab:
    def __init__(self, frequencies, max_size, min_freq, labels = False) -> None:
        self.frequencies = frequencies
        self.max_size = max_size
        self.min_freq = min_freq
        self.labels = labels
        
    def itos(self):  #the je na indexu 2 jer ima najvecu frekvenciju   #vraca listu

        d = sorted(self.frequencies.items(), key=lambda x: x[1], reverse=True)
        if self.max_size>=0 and len(d) > self.max_size:
            return Exception("The size fof the vocabulary is too big.")

        lista = []
        index = 0
        if not self.labels:
            lista[0] = "<PAD>"
            lista[1] = "<UNK>"
            index = 2

        for key, value in d:

            if self.min_freq < value:
                lista[index] = key
                index += 1

        return lista


    def stoi(self):   #the je na indexu 2 jer ima najvecu frekvenciju   #vraca dictionary

        d = sorted(self.frequencies.items(), key=lambda x: x[1], reverse=True)
        if self.max_size>=0 and len(d) > self.max_size:
            return Exception("The size fof the vocabulary is too big.")

        vocab = {}
        index = 0
        if not self.labels:
            vocab["<PAD>"] = 0
            vocab["<UNK>"] = 1
            index = 2

        for key, value in d:

            if self.min_freq < value:
                vocab[key] = index
                index += 1

        self.vocab = vocab
        return vocab


    def encode(self, text):    #Numericalized text

        numericalized_text = []

        if type(text) is not list:
            text = [text]

        for el in text:
            if el in self.vocab:
                numericalized_text.append( self.vocab[el] )
            elif len(text)>5:
                numericalized_text.append( 1 )  #UNK charachter

        return torch.tensor(numericalized_text)



def embedding_matrica(vocab, use_glove=True):   #stoi vocab   #matrica ce biti velicine V x dimenzija


    matrica = torch.normal(0, 1, size=(len(vocab), 300))
    matrica[0] = torch.abs (matrica[0] * 0.0 )   #za padding

    if not use_glove:
        return matrica

    f = open('sst_glove_6b_300d.txt', 'r')
    ucitani_dictionary = {}
    for lines in f:
        line = lines.rstrip().split()
        k = line[0]
        line = [float(i) for i in line[1:]]
        ucitani_dictionary[k] = line

    for key, value in sorted(vocab.items(), key=lambda x: x[1], reverse=False):
        
        if key in ucitani_dictionary:
            matrica[value] = torch.FloatTensor(ucitani_dictionary[key])
            
    return matrica