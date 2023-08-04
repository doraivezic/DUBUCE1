import numpy as np
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import torch

import data

class PTLogreg(nn.Module):
    def __init__(self, D, C):
        """Arguments:
            - D: dimensions of each datapoint 
            - C: number of classes
        """
        super(PTLogreg, self).__init__()
        self.W = nn.Parameter(torch.randn(D, C))
        self.b = nn.Parameter(torch.randn(C))
        # inicijalizirati parametre (koristite nn.Parameter):
        # imena mogu biti self.W, self.b
        # ...
        return

    def forward(self, X):
        # unaprijedni prolaz modela: izračunati vjerojatnosti
        #   koristiti: torch.mm, torch.softmax
        # ...
        out = torch.softmax( torch.mm( X, self.W) + self.b , dim=1)
        return out

    def get_loss(self, X, Yoh_):
        # formulacija gubitka
        #   koristiti: torch.log, torch.mean, torch.sum
        # ...
        return  - torch.mean ( torch.sum( torch.log( self.forward(X) ) * Yoh_ ,  dim=1))


def train(model, X, Yoh_, param_niter=1e5, param_delta=0.01, param_lambda=0.0):
    """Arguments:
        - X: model inputs [NxD], type: torch.Tensor
        - Yoh_: ground truth [NxC], type: torch.Tensor
        - param_niter: number of training iterations
        - param_delta: learning rate
    """

    # inicijalizacija optimizatora
    optimizer = optim.SGD(model.parameters(), lr=param_delta, weight_decay=param_lambda)

    # petlja učenja
    # ispisujte gubitak tijekom učenja
    for epoch in range(int(param_niter)+1):

        loss = model.get_loss(X, Yoh_)      #a ako rucno radimo regularizaciju onda:     #loss = loss + l2_norm * l2_lambda 

        if epoch%100 == 0:
            print("Epoch %d\tLoss: %.04f" %(epoch, loss))

        loss.backward()    #racuna derivaciju od loss po weight
        optimizer.step()    #azurira w i b (prema gradijentnom spustu)

        optimizer.zero_grad()
    return


def eval(model, X):
    """Arguments:
        - model: type: PTLogreg
        - X: actual datapoints [NxD], type: np.array
        Returns: predicted class probabilites [NxC], type: np.array
    """
    # ulaz je potrebno pretvoriti u torch.Tensor  -> napravljeno u mainu
    # izlaze je potrebno pretvoriti u numpy.array
    # koristite torch.Tensor.detach() i torch.Tensor.numpy()

    #detach() -> Returns a new Tensor, detached from the current graph.
                #The result will never require gradient.

    probs = model.forward(X)
    return probs.detach().numpy()



def decfun(model):
    def classify(X):
        return np.argmax( eval(model, torch.Tensor(X) ),  axis=1 )
    return classify

if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(400)

    """
        Y_ : stvarna klasa
        Yoh_ : stvarna klasa u one-hot zapisu

        Y : predviđena klasa
        Yoh : predviđena klasa u one-hot zapisu
    """

    # instanciraj podatke X i labele Yoh_
    X,Y_ = data.sample_gauss_2d(3,100)

    Yoh_ = data.class_to_onehot(Y_) #umjesto koristenja data mogli smo i preko torch
    Yoh_ = torch.Tensor(Yoh_)

    X = torch.Tensor(X)
    
    # definiraj model:
    model = PTLogreg(X.shape[1], Yoh_.shape[1])

    # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
    train(model, X, Yoh_, 1200, 0.5)

    # dohvati vjerojatnosti na skupu za učenje  + pretvaramo u numpy sve Y
    probs = eval(model, X)  
    Y = np.argmax(probs, axis=1)
    Yoh = data.class_to_onehot(Y)

    Yoh_ = Yoh_.numpy()

    # ispiši performansu (preciznost i odziv po razredima)
    C = Yoh_.shape[1]  #broj razreda
    for c in range(C):
        accuracy, recall, precision = data.eval_perf_binary( Yoh[:,c], Yoh_[:,c] )  #saljemo za svaki razred posebno
        print ("Class",c, "Accuracy:",accuracy, "Recall:",recall, "Precision",precision)

    # iscrtaj rezultate, decizijsku plohu

    # graph the decision surface
    f = decfun(model)
    rect=(np.min(X.numpy(), axis=0), np.max(X.numpy(), axis=0))
    data.graph_surface( f, rect)

    # graph the data points
    data.graph_data(X, Y_, Y)

    plt.show()