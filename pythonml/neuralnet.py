import numpy as np
from scipy.stats import norm


class Layer:
    def __init__(self, nneuron, nneuron_m1, type = ''):
        self.first = False
        self.last = False
        if (type == 'first'):
            self.first = True
        if (type == 'last'):
            self.last = True

        self.nn = nneuron
            
        self.a = np.zeros((self.nn))
        self.z = np.zeros((self.nn))
        
        if (not self.first):
            self.nn_m1 = nneuron_m1
            self.w = np.zeros((self.nn, self.nn_m1))
            self.dw = self.w.copy()
            self.b = np.zeros((self.nn))
            self.db = self.b.copy()
            self.delta = np.zeros((self.nn))


class NeuralNet:
    def __init__(self, nneuron, lrate = 0.1,
                     conv_reldiff = 1.e-3, maxiter=100):
        self.nlayer = len(nneuron)
        self.nneuron = nneuron
        self.lrate = lrate
        self.conv_reldiff = conv_reldiff
        self.maxiter = maxiter
        self.cf = np.zeros((self.maxiter))
        self.final_it = 0
        
        self.x = 0
        self.layer = []

        ## add the bias term to the zeroth neuron
        nneuron = np.array(nneuron) 
        
        self.layer.append(
            Layer(nneuron = nneuron[0],
                  nneuron_m1 = 0, type = 'first'))
        for i in np.arange(1,self.nlayer):
            type = ''
            if (i == self.nlayer-1):
                type = 'last'
            self.layer.append(
                Layer(nneuron = nneuron[i],
                      nneuron_m1 = nneuron[i-1],
                      type=type))


        
        ## initiate correct output for training
        ## unsure about these
        self.a = np.zeros(nneuron[-1])
        self.loss = 0.0
        self.damp = 3.
        self.damp2 = 0.01
        
    def actf(self, z):
        return 1./(1+np.exp(-z))

    def dactf(self, a):
        return a*(1-a)

    def predict(self, X):
        self.forward_prop(X = X)
        return self.layer[self.nlayer-1].a

        
    def costfunction(self, idat):
        # assume forward prop is run.
        ## need to add idat-functionality.
        nl = self.nlayer
        aa = self.layer[nl-1].a
        yy = self.y[idat,:]
        cost = -np.sum(     yy *np.log(    aa) +
                       (1.0-yy)*np.log(1.0-aa))
        cost = cost/self.ndata
        
        return cost

        
    def init_data(self, X, y):
        self.X = X
        self.y = y
        self.ndata = self.X.shape[1]
        
    def forward_prop(self, idat=None, X=None):
        if idat == None:
            self.layer[0].a = X
        else:
            self.layer[0].a = self.X[:,idat]
            
        for i in np.arange(1, self.nlayer):
            self.layer[i].z = (self.layer[i].w.dot(self.layer[i-1].a)
                                   + self.layer[i].b)
            self.layer[i].a = self.actf(self.layer[i].z)
            
            
    def backward_prop(self,idat):
        nl = self.nlayer
        # set first delta.
        self.layer[nl-1].delta = self.layer[nl-1].a - self.y[idat,:]
                                      

        # backprop
        for i in np.arange(nl-2, 0, -1):
            delta_tmp = np.dot(self.layer[i+1].w.transpose(),
                self.layer[i+1].delta)*self.dactf(self.layer[i].a)
            self.layer[i].delta = delta_tmp
            
        for i in np.arange(nl-1, 0, -1):
            self.layer[i].dw = self.layer[i].dw + (np.outer(
                self.layer[i].delta,
                self.layer[i-1].a))
            self.layer[i].db = self.layer[i].db + self.layer[i].delta 
        
    def init_w_b_random(self):
        for l in self.layer[1:]:
            nn = l.nn
            nn_m1 = l.nn_m1
            nxn = nn*nn_m1
            w = np.reshape(norm.rvs(0.,1.,nxn), (nn,nn_m1))
            l.w[:,:] = w[:,:]
            l.b = norm.rvs(0,1,nn)


    def reset_dw_db(self):
        for i in np.arange(1,self.nlayer):
            self.layer[i].dw[:,:] = 0.0
            self.layer[i].db[:] = 0.0


    def train_model(self):
        # init training
        self.init_w_b_random()

        ## iteration control
        reldiff = self.conv_reldiff
        maxiter = self.maxiter
        dcost = 2.0*reldiff
        cost = 0.0
        cost_prev = 0.0
        it = 0

        while ((dcost > reldiff) and (it < maxiter)):
            print(it)
            ## compute dw
            self.reset_dw_db()
            cost = 0.0
            for idat in np.arange(self.ndata):
                self.forward_prop(idat = idat)
                self.backward_prop(idat = idat)
                cost = cost + self.costfunction(idat=idat)

            ## update dw
            for layer in self.layer[1:self.nlayer]:
                layer.w = layer.w - self.lrate*layer.dw
                layer.b = layer.b - self.lrate*layer.db

            # do another iteration?
            if (it == 0):
                cost_prev = cost/(2.0*dcost + 1.0)
            dcost = np.abs((cost - cost_prev)/cost_prev)
            dcost_prev = cost
            self.cf[it] = cost
            it = it + 1
            
        self.final_it = it    
            
            
