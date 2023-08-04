import torch
import torch.nn as nn
import torch.nn.functional as F


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True):
        super(_BNReluConv, self).__init__()
        # YOUR CODE HERE -> normalizacije po grupi, aktivacije ReLU i konvolucije
        self.add_module("bnrc_conv", nn.Conv2d(num_maps_in, num_maps_out, kernel_size=(k,k), bias=bias))
        self.add_module("bnrc_relu", nn.ReLU())
        self.add_module("bnrc_batchnorm", nn.BatchNorm2d(num_maps_out))

class SimpleMetricEmbedding(nn.Module):
    def __init__(self, input_channels, emb_size=32):
        super().__init__()
        self.emb_size = emb_size
        # YOUR CODE HERE
        self.bnrc1 = _BNReluConv(num_maps_in=input_channels, num_maps_out=emb_size, k=3)
        self.bnrc23 = _BNReluConv(num_maps_in=emb_size, num_maps_out=emb_size, k=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

    def get_features(self, img):
        # Returns tensor with dimensions BATCH_SIZE, EMB_SIZE - Pripazite da izlazni tenzor u metodi get_features zadrži 
        #                                                       prvu dimenziju koja označava veličinu minigrupe, čak i kada je ona jednaka 1
        # YOUR CODE HERE

        x = self.bnrc1(img)
        x = self.maxpool(x)

        x = self.bnrc23(x)
        x = self.maxpool(x)

        x = self.bnrc23(x)
        
        x = self.avgpool(x)

        return x[:,:,-1,-1]


    def loss(self, anchor, positive, negative):
        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)
        # YOUR CODE HERE - Implementirajte trojni gubitak po uzoru na pytorchev TripletMarginLoss
        dist_ap = torch.linalg.norm(a_x - p_x, dim=1)
        dist_an = torch.linalg.norm(a_x - n_x, dim=1)
        loss = torch.relu(dist_ap - dist_an + 1.0)  #za margin=1

        return torch.mean(loss)

