#modul mnist_shootout će usporediti performansu do tada razvijenih klasifikatora na skupu podataka MNIST

import torch
import torchvision
import torch.nn.functional as F

from matplotlib import image
import os

import pt_deep

torch.device('cuda:0')

dataset_root = '/mnist'  # change this to your preference
mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=True)
mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=True)

x_train, y_train = mnist_train.data, mnist_train.targets
x_test, y_test = mnist_test.data, mnist_test.targets
x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)


N = x_train.shape[0]
D = x_train.shape[1] * x_train.shape[2]
C = y_train.max().add_(1).item()

X = x_train.view(N,D)
Yoh = F.one_hot(y_train, C)

X_test = x_test.view(x_test.shape[0],D)
Yoh_test = F.one_hot(y_test, C)

konf_lista = [784,10,10]  #[N, ..., C]
model = pt_deep.PTDeep(konfiguracijska_lista=konf_lista, aktivacijska_fja=torch.relu)

pt_deep.train(model, X, Yoh, param_niter=100, param_lambda=0.15)


#ZAD1  iscrtajte i komentirajte naučene matrice težina za svaku pojedinu znamenku

matrice_tezina = model.weights[0].detach().numpy().T.reshape(-1, 28, 28)

for index, matrice_tezina in enumerate(matrice_tezina):
    image.imsave(os.path.join(f"{index}.png"), matrice_tezina)