#treba nasljeđivati torch.utils.data.Dataset te implementirati __getitem__ metodu. 
#Ovaj razred služi za spremanje i dohvaćanje podataka te operacije za koje nam je potreban cijeli skup podataka, poput izgradnje vokabulara.


from torch.utils.data import Dataset
import torch
import csv

class NLPDataset(Dataset):
    def __init__(self, file, vocabulary_text, vocabulary_label) -> None:
        super(NLPDataset, self)

        self.vocabulary_text = vocabulary_text
        self.vocabulary_label = vocabulary_label
        
        all_instances = []
        
        with open(file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            for row in csv_reader:
                all_instances.append((row[0].split(), row[1].strip()))

        self.all_instances = all_instances
        

        
    def __getitem__(self, index):
        
        review, label = self.all_instances[index]
        numericalized_review = self.vocabulary_text.encode(review)
        numericalized_label = self.vocabulary_label.encode(label)

        return numericalized_review, numericalized_label



    def __len__(self):
        return len(self.all_instances)