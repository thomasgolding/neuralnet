import pythonml as pml
import numpy as np


def test_deriv():
    nnet = pml.NeuralNet(nneuron = (2,30,24,30,21,32,2))
    X = np.array([[-2],[3]])
    y = np.array([[1,0]])
    nnet.init_data(X, y)
    
    ## get derivatives = fill nnet.layer[i].dw.
    nnet.reset_dw_db()
    nnet.init_w_b_random()
    idat = 0
    nnet.forward_prop(idat=idat)
    nnet.backward_prop(idat=idat)
                
        
    ## compute numerical derivatives.
    ## save the num derivatives in
    d = 1.e-4
    numderw = []
    numderb = []
    cost0 = nnet.costfunction(idat = 0)
        
    for l in np.arange(1, nnet.nlayer):
        w0 = nnet.layer[l].w.copy()
        dw0 = w0*0.0
        b0 = nnet.layer[l].b.copy()
        db0 = b0*0.0
        
        for i in np.arange(nnet.layer[l].nn):
            tmpb = b0.copy()
            deltab = tmpb[i]*d
            tmpb[i] = tmpb[i] + deltab
            nnet.layer[l].b = tmpb.copy()
            nnet.forward_prop(idat = 0)
            cost = nnet.costfunction(idat = 0)
            db0[i] = (cost - cost0)/deltab
            nnet.layer[l].b = b0.copy()
            for j in np.arange(nnet.layer[l-1].nn):
                tmp = w0.copy()
                deltaw = tmp[i,j]*d
                tmp[i,j] = tmp[i,j] + deltaw
                nnet.layer[l].w = tmp.copy()
                nnet.forward_prop(idat = idat)
                cost = nnet.costfunction(idat = 0)
                dw0[i,j] = (cost-cost0)/deltaw
                nnet.layer[l].w = w0.copy()
        numderb.append(db0)
        numderw.append(dw0)

    diff=0.0
    for l in np.arange(1,nnet.nlayer):                 
        diff = diff + np.sum(np.sum(numderw[i] - nnet.layer[i+1].dw))
        diff = diff + np.sum(numderb[i] - nnet.layer[i+1].db)
    print(diff)
    assert np.abs(diff) < 1.e-2

def test_training():
    nnet = pml.NeuralNet(nneuron = (2,3,2))
    X = np.array([[-2],[3]])
    y = np.array([[1,0]])
    nnet.init_data(X, y)
    


        
        
