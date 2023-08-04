#Zadatak naše collate funkcije biti će nadopuniti duljine instanci znakom punjenja do duljine najdulje instance u batchu

import torch
from torch.nn.utils.rnn import pad_sequence

def pad_collate_fn(batch, pad_index=0):
    """
    Arguments:
      Batch:
        list of Instances returned by `Dataset.__getitem__`.
      pad_index:
        koji se index koristi kao znak punjenja
    Returns:
      A tensor representing the input batch.
    """
    
    texts, labels = zip(*batch)

    lengths = torch.tensor([len(text) for text in texts]) # Needed for later  #duljine originalnih instanci

    texts = pad_sequence(texts, batch_first=True, padding_value=pad_index)
    labels = torch.tensor(labels)
    
    return texts, labels, lengths
