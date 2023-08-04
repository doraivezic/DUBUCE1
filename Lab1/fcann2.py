#Modul fcann2 će sadržavati implementaciju dvoslojnog potpuno povezanog modela temeljenog na NumPyjevim primitivima

#probabilistički klasifikacijski modelom s jednim skrivenim slojem

import numpy as np
import matplotlib.pyplot as plt

import data




def fcann2_train(X,Y_, H=10):
    
    N = X.shape[0]
    D = X.shape[1]

    #broj klasa
    C = len(np.unique(Y_))

    w1 = np.random.randn(D, H)
    b1 = np.random.randn(H)
    w2 = np.random.randn(H, C) #visina C, dužina dim
    b2 = np.random.randn(C)
    
    param_niter = int(1e5)
    param_delta = 0.05 #learning rate
    param_lambda = 1e-3  #koeficijent regularizacije

    # gradijentni spust (param_niter iteracija)
    for i in range(param_niter+1):
        
        # klasifikacijske mjere
        # np.matmul (@) recommended for 2D or higher matrix multiplication
        scores1 = X @ w1 + b1    # N x dim
        h1 = np.maximum(0., scores1)   #ReLU = max(0,x)
        scores2 = h1 @ w2 + b2    # N x C

        # vjerojatnosti
        # moramo koristiti softmax (a ne sigmoidu jer je viseklasna logisticka regresija)
        e = np.exp(scores2)
        #suma svakog reda, odnosno svakog primjera
        #sum_e = np.sum(e, axis=1)

        #P(Y=yi|xi)=softmax(s2) tj vjerojatnost da primjer xi pripada klasi yi
        e /= np.sum(e, axis=1)[:,None]  #sada tu imam sve vjerojatnosti i iz njih za svaki primjer trebam uzeti samo one iz stupca y (tocne predikcije)


        y_one_hot = data.class_to_onehot(Y_)
        m = e * y_one_hot
        loss =  - np.mean ( np.sum( np.log( m, out=np.zeros_like(m), where=(m!=0) ), axis=1 ) )


        if i%5000 == 0:
            print("Epoch %d\tLoss: %.04f" %(i, loss))

        if loss < 1e-15:
            break


        #gradijenti

        #parcijalno L po parcijalno s2
        Gs2 = e - y_one_hot

        grad_w2 = 1/N * ( Gs2.T @ h1 ).T  #  (NxC).T x Nxdim => C x dim   .T  =>  dimxC
        grad_b2 = 1/N * np.sum( Gs2, axis=0 ) #1xC

        #parcijalno L po parcijalno s1
        Gs1 = (Gs2 @ w2.T) # NxC x Cxdim   *  Nxdim  = Nxdim
        Gs1[scores1 <= 0] = 0

        grad_w1 = 1/N * ( Gs1.T @ X ).T  #dimxN x NxD => dimxD => Dxdim
        grad_b1 = 1/N * np.sum( Gs1, axis=0 ) #1xdim


        #azuriranje parametara w i b
        w1 -= grad_w1 * param_delta
        b1 -= grad_b1 * param_delta
        w2 -= grad_w2 * param_delta
        b2 -= grad_b2 * param_delta

    return w1, b1, w2, b2


def fcann2_classify(X, w1, b1, w2, b2):
    scores1 = X @ w1 + b1    # N x dim
    h1 = np.maximum(0., scores1)   #ReLU = max(0,x)
    scores2 = h1 @ w2 + b2    # N x C

    e = np.exp(scores2)
    e /= np.sum(e, axis=1)[:,None]

    predictions = np.argmax(e, axis=1)

    return predictions


def decfun(w1, b1, w2, b2):
    def classify(X):
        return fcann2_classify(X, w1, b1, w2, b2)
    return classify

if __name__=="__main__":
    np.random.seed(60)

    # get data
    X,Y_ = data.sample_gmm_2d(6,2,10)
    # X,Y_ = sample_gauss_2d(2, 100)

    # train the model
    w1, b1, w2, b2 = fcann2_train(X, Y_)

    # evaluate the model on the training dataset
    Y = fcann2_classify(X, w1, b1, w2, b2)

    # graph the decision surface
    f = decfun(w1,b1,w2,b2)
    rect=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface( f, rect)

    # graph the data points
    data.graph_data(X, Y_, Y)

    plt.show()