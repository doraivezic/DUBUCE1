#učitavanje podataka tako da se omogući učenje modela za metričko ugrađivanje trojnim gubitkom

from torch.utils.data import Dataset
from collections import defaultdict
from random import choice
import torchvision


class MNISTMetricDataset(Dataset):
    def __init__(self, root="/tmp/mnist/", split='train', remove_class=None):
        super().__init__()
        assert split in ['train', 'test', 'traineval']
        self.root = root
        self.split = split
        mnist_ds = torchvision.datasets.MNIST(self.root, train='train' in split, download=True)
        self.images, self.targets = mnist_ds.data.float() / 255., mnist_ds.targets
        self.classes = list(range(10))

        #zad 3e - omogućite uklanjanje primjera odabrane klase iz skupa za učenje
        self.remove_class = remove_class
        if remove_class is not None:
            # Filter out images with target class equal to remove_class
            # YOUR CODE HERE
            indexes = []
            for i, target in enumerate(self.targets):
                if target == remove_class:
                    indexes.append(i)

            self.images =   [elem   for  i, elem  in enumerate(self.images)   if i not in indexes]
            self.targets =  [elem   for  i, elem  in enumerate(self.targets)  if i not in indexes]

        self.target2indices = defaultdict(list)
        for i in range(len(self.images)):
            self.target2indices[self.targets[i].item()] += [i]

    def _sample_negative(self, index):
        # YOUR CODE HERE
        not_tagret_class = self.targets[index].item()    #klasa anchor primjera -> jedino u toj klasi ne smijemo traziti negativ

        negativ_class = not_tagret_class
        while negativ_class==not_tagret_class:
            negativ_class = choice(self.classes)

            if self.remove_class is not None:
                if negativ_class==self.remove_class:
                    negativ_class = not_tagret_class
        
        return choice(self.target2indices[negativ_class])


    def _sample_positive(self, index):
        # YOUR CODE HERE

        target_class = self.targets[index].item()    #klasa anchor primjera -> trazimo slucano odabrani primjer s istom tom klasom

        pozitiv_index = index
        while index==pozitiv_index: 
            pozitiv_index = choice(self.target2indices[target_class])

        return pozitiv_index    #vracamo index pozitiva


    def __getitem__(self, index):
        anchor = self.images[index].unsqueeze(0)
        target_id = self.targets[index].item()
        if self.split in ['traineval', 'val', 'test']:
            return anchor, target_id
        else:
            positive = self._sample_positive(index)
            negative = self._sample_negative(index)
            positive = self.images[positive]
            negative = self.images[negative]
            return anchor, positive.unsqueeze(0), negative.unsqueeze(0), target_id

    def __len__(self):
        return len(self.images)
