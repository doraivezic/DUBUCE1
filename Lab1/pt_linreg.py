#zadatak3

#optimizacijskog postupka za određivanje parametara pravca y = a * x + b 
# koji prolazi kroz točke (1,3) i (2,5).
# Modificirajte program na način da se pravac može provući kroz proizvoljan broj točaka

import torch
import torch.optim as optim


def pt_linreg(X=torch.tensor([1, 2]), Y=torch.tensor([3, 5])):

    a = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)

    # optimizacijski postupak: gradijentni spust
    optimizer = optim.SGD([a, b], lr=0.1)


    for i in range(100):
        # afin regresijski model
        Y_ = a*X + b

        diff = (Y-Y_)

        # kvadratni gubitak
        loss = torch.mean(diff**2)

        # računanje gradijenata
        loss.backward()
        #ručno računanje gradijenata  dL/da = dL/dY_ * dY_/da = 2*(-1)*(Y-Y_)*X
        with torch.no_grad():
            a_gradijent = 2. * (-1.) * torch.mean( (Y-Y_) * X )
            b_gradijent = 2. * (-1.) * torch.mean(Y-Y_)


        # korak optimizacije
        optimizer.step()

        if(i%25==0):
            print(f'step: {i+1}, loss:{loss:.3f}, Y_:{Y_}, a:{a.item():.3f}, b {b.item():.3f}, a_grad {a.grad.item():.3f}, b_grad {b.grad.item():.3f}, a_grad2 {a_gradijent.item():.3f}, b_grad2 {b_gradijent.item():.3f}')

        # Postavljanje gradijenata na nulu
        optimizer.zero_grad()


    return


if __name__=="__main__":
    X=torch.tensor([1, 2, 3, 4])
    Y=torch.tensor([3, 5, 7, 10])
    pt_linreg(X,Y)