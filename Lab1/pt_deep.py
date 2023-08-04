import numpy as np
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import torch

import data

class PTDeep(nn.Module):
    def __init__(self, konfiguracijska_lista, aktivacijska_fja=torch.relu):
        super(PTDeep, self).__init__()

        D = konfiguracijska_lista[0]
        C = konfiguracijska_lista[-1]

        H = konfiguracijska_lista[1:-1]   #skriveni slojevi i odgovarajuci broj neurona u svakom
        self.H = H
        self.activ_f = aktivacijska_fja

        #za svaki sloj moramo njegove vrijednosti spremit
        self.weights = nn.ParameterList( nn.Parameter(torch.randn(konfiguracijska_lista[i], konfiguracijska_lista[i+1])) for i in range(len(konfiguracijska_lista)-1) )
        self.biases =  nn.ParameterList( nn.Parameter(torch.randn(konfiguracijska_lista[i+1])) for i in range(len(konfiguracijska_lista)-1))

        return

    def forward(self, X):

        s = torch.clone(X)

        for w,b in zip( self.weights[:-1], self.biases[:-1]):
            s = self.activ_f( torch.mm( s, w ) + b )
            
        w = self.weights.__getitem__(-1)
        b = self.biases.__getitem__(-1)
        out = torch.softmax( torch.mm( s, w) + b , dim=1)
        return out

    def get_loss(self, X, Yoh_):
        return  - torch.mean ( torch.sum( torch.log( self.forward(X) + 1e-15) * Yoh_ ,  dim=1))




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

        loss = model.get_loss(X, Yoh_) 

        if epoch%1000 == 0:
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
    
    probs = model.forward(X)
    return probs.detach().numpy()




def count_params(model):
    #ispis simboličkog imena i dimenzije tenzora   svih parametara

    suma = 0
    for ime, dim in model.named_parameters():
        print(ime, dim.size())
        suma += dim.numel()
    print("Ukupan broj parametara modela je", suma)
    return





def decfun(model):
    def classify(X):
        return np.argmax( eval(model, torch.Tensor(X) ),  axis=1 )
    return classify

if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)

    """
        Y_ : stvarna klasa
        Yoh_ : stvarna klasa u one-hot zapisu

        Y : predviđena klasa
        Yoh : predviđena klasa u one-hot zapisu
    """

    # instanciraj podatke X i labele Yoh_
    #X,Y_ = data.sample_gmm_2d(4,2,40)
    X,Y_ = data.sample_gmm_2d(6,2,10)

    Yoh_ = data.class_to_onehot(Y_)
    Yoh_ = torch.Tensor(Yoh_)

    X = torch.Tensor(X)
    
    #oblik mora biti takav da je prvi element D  i zadnji da je C 
    #konfig_lista = [2,2]
    #konfig_lista = [2,10,2]
    konfig_lista = [2,10,10,2]
    model = PTDeep(konfig_lista, aktivacijska_fja=torch.sigmoid)

    # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
    train(model, X, Yoh_, 1e4, 0.1, 1e-4)

    # dohvati vjerojatnosti na skupu za učenje  + pretvaramo u numpy sve Y
    probs = eval(model, X)  
    Y = np.argmax(probs, axis=1)
    Yoh = data.class_to_onehot(Y)

    Yoh_ = Yoh_.numpy()

    # ispiši performansu (preciznost i odziv po razredima)
    print()
    mean_precision = 0.0

    C = Yoh_.shape[1]
    for c in range(C):
        accuracy, recall, precision = data.eval_perf_binary( Yoh[:,c], Yoh_[:,c] )
        print ("Class",c, "  Accuracy:",accuracy, "Recall:",recall, "Precision",precision)

        mean_precision += precision
    mean_precision = mean_precision / Yoh_.shape[1]
    print("Mean accuracy:", mean_precision)

    print()
    count_params(model)

    # iscrtaj rezultate, decizijsku plohu

    # graph the decision surface
    f = decfun(model)
    rect=(np.min(X.numpy(), axis=0), np.max(X.numpy(), axis=0))
    data.graph_surface( f, rect)

    # graph the data points
    data.graph_data(X, Y_, Y)

    plt.show()