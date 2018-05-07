import neuralnet as n
import numpy as np


def test_deriv():
    nnet = n.NeuralNet(nneuron = (2,3,2))
    X = np.array([[-2],[3]])
    y = np.array([[1,0]])
    nnet.init_data(X, y)
    
    ## get derivatives = fill nnet.layer[i].dw.
    nnet.reset_dw()
    nnet.init_w_random()
    idat = 0
    nnet.forward_prop(idat=idat)
    nnet.backward_prop(idat=idat)
                
        
    ## compute numerical derivatives.
    ## save the num derivatives in
    d = 1.e-4
    numder = []
    cost0 = nnet.costfunction()
        
    for l in np.arange(1, nnet.nlayer):
        w0 = nnet.layer[l].w.copy()
        dw0 = w0*0.0
        for i in np.arange(nnet.layer[l].nn):
            for j in np.arange(nnet.layer[l-1].nn):
                tmp = w0.copy()
                deltaw = tmp[i,j]*d
                tmp[i,j] = tmp[i,j] + deltaw
                nnet.layer[l].w = tmp.copy()
                nnet.forward_prop(idat = idat)
                cost = nnet.costfunction()
                dw0[i,j] = (cost-cost0)/deltaw
                nnet.layer[l].w = w0.copy()
                numder.append(dw0)
                    
                    
    diff = np.sum(np.sum(numder[0] - nnet.layer[1].dw))
    print(diff)
    assert np.abs(diff) < 1.e-2


        
        
