# provedite klasifikaciju na podskupu za validaciju, ali ovaj put u prostoru slike. 
# Ostvarite taj zadatak oblikovanjem razreda koji u metodi get_features provodi jednostavnu vektorizaciju slike.

# Modificirajte funkciju za učenje tako da se klasifikacija provodi u prostoru slike. 

import torch.nn as nn

class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()

    def get_features(self, img):
        # YOUR CODE HERE
        feats = img.flatten()
        return feats