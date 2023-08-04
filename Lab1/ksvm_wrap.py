#Modul ksvm_wrap će umatati klasifikator s jezgrenim ugrađivanjem i potpornim vektorima 
#izveden u modulu sklearn.svm biblioteke scikit-learn 
#te omogućiti usporedbu s klasifikatorima temeljenima na dubokom učenju.

import numpy as np
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import torch
from sklearn import svm


import data

class KSVMWrap:

    '''
    Metode:
    __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        Konstruira omotač i uči RBF SVM klasifikator
        X, Y_:           podatci i točni indeksi razreda
        param_svm_c:     relativni značaj podatkovne cijene
        param_svm_gamma: širina RBF jezgre

    predict(self, X)
        Predviđa i vraća indekse razreda podataka X

    get_scores(self, X):
        Vraća klasifikacijske mjere
        (engl. classification scores) podataka X;
        ovo će vam trebati za računanje prosječne preciznosti.

    support
        Indeksi podataka koji su odabrani za potporne vektore
    '''

    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        super(KSVMWrap, self).__init__()

        self.model = svm.SVC(C=param_svm_c, kernel='rbf', gamma=param_svm_gamma, probability=True)
        self.model.fit(X,Y_)
        return
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_scores(self, X):
        return self.model.decision_function(X)
    
    def support(self):
        return self.model.support_
    


if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(400)

    # instanciraj podatke X i labele Yoh_
    X, Y_ = data.sample_gmm_2d(6,2,10)

    Yoh_ = data.class_to_onehot(Y_)
    
    # definiraj model:
    model = KSVMWrap(X,Y_)

    Y = model.predict(X)

    #probs = model.get_scores(X) 

    Yoh_ = data.class_to_onehot(Y_)
    Yoh = data.class_to_onehot(Y)

    mean_precision = 0.0
    for c in range(len(Yoh_[0])):
        accuracy, recall, precision = data.eval_perf_binary( Yoh[:,c], Yoh_[:,c] )
        print ("Class",c, "Accuracy:",accuracy, "Recall:",recall, "Precision",precision)

        mean_precision += precision
    mean_precision = mean_precision / Yoh_.shape[1]
    print("Mean accuracy:", mean_precision)

    # graph the decision surface
    f = model.get_scores
    rect=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface( f, rect)

    # graph the data points
    data.graph_data(X, Y_, Y,   special=model.support())

    plt.show()