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
        
        self.a = np.zeros((nneuron))
        self.z = self.a.copy()
        self.b = self.a.copy()
        self.nn = nneuron
        if (not self.first):
            self.nn_m1 = nneuron_m1
            self.w = np.zeros((self.nn, self.nn_m1))
            self.dw = self.w.copy()
            self.delta = np.zeros((self.nn))

        
    def calc_x(self, prevlayer):
        if (not self.first):
           self.g = self.w.dot(prevlayer.x) + self.b
           self.x = 1.0/(1.0 + np.exp(-self.g))
        else:
            print("Can't calc x in first layer.")

    def calc_dxdx_m1(self):
        if (not self.first):
            for i in np.arange(self.nneural):
                self.dxdx_m1[i,:] = self.x[i]*\
                  (1.0-self.x[i])*self.w[i,:]

    def set_x(self, x):
        nx = np.size(x)
        if (nx == self.nn):
            self.x[:] = x
        #else if nx = 1:
        #    self.x = x
        else:
            print('len(x) must be ' + str(self.nn))

    def set_b(self, b):
        if (len(b) == self.nn):
            self.b[:] = b
        else:
            print('len(b) must be ' + str(self.nn))

    def set_w(self, w):
        if (self.first):
            print('No weights for first layer.')
            return
        if (w.shape == self.w.shape):
            self.w[:,:] = w[:,:]
        else:
            print('w.shape must be ' + str(self.w.shape))

            
    def get_w(self):
        if (not self.first):
            return self.w
        else:
            print('No weights in first layer.')

    def get_g(self):
        return self.g

    def get_b(self):
        return self.b
 
class NeuralNet:
    def __init__(self, nneuron):
        self.nlayer = len(nneuron)
        self.x = 0
        self.layer = []
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

    def costfunction(self):
        # assume forward prop is run.
        nl = self.nlayer
        cost = -np.sum(    self.y *np.log(    self.layer[nl-1].a) +
                      (1.0-self.y)*np.log(1.0-self.layer[nl-1].a))
        return cost

        
    def init_data(self, X, y):
        self.X = X
        self.y = y
        self.ndata = self.X.shape[0]
        
    def forward_prop(self,idat):
        self.layer[0].a = self.X[:,idat]
        for i in np.arange(1, self.nlayer):
            self.layer[i].z = self.layer[i].w.dot(self.layer[i-1].a)
            self.layer[i].a = self.actf(self.layer[i].z)
            
    def backward_prop(self,idat):
        nl = self.nlayer
        # set first delta.
        self.layer[nl-1].delta = self.layer[nl-1].a - self.y[idat,:]
                                      

        # backprop
        for i in np.arange(nl-2, 0, -1):
            self.layer[i].delta = np.dot(self.layer[i+1].w.transpose(),
                self.layer[i+1].delta)*self.dactf(self.layer[i].a)

        for i in np.arange(nl-1, 0, -1):
            self.layer[i].dw = self.layer[i].dw + (np.outer(
                self.layer[i].delta,
                self.layer[i-1].a))
        
    def init_w_random(self):
        for il in np.arange(1, self.nlayer):
            nn = self.layer[il].nn
            nn_m1 = self.layer[il].nn_m1
            nxn = nn*nn_m1
            w = np.reshape(norm.rvs(0.,1.,nxn), (nn,nn_m1))
            self.layer[il].w[:,:] = w[:,:] 

            
    def reset_dw(self):
        for i in np.arange(1,self.nlayer):
            self.layer[i].dw[:,:] = 0.0
            
    def init_x0(self, x):
        self.layer[0].set_x(x)

    def set_b(self, il, b):
        self.layer[il].set_b(b)

    def set_w(self, il, w):
        self.layer[il].set_w(w)
        
    def calc_all_x(self):
        for i in np.arange(1,self.nlayer):
            self.layer[i].calc_x(self.layer[i-1])

    def load_training_set(self, x0, a):
        n_in = self.layer[0].nn
        n_out = self.layer[-1].nn
        if ((np.size(x0) != n_in) or (np.size(a) != n_out)):
            print("len(x0) != "+ str(n_in) + ' or len(a) != ' + str(n_out))
            return -1

        self.init_x0(x0)
        if np.size(a) > 1:
            self.a[:] = a[:]
        else:
            self.a = a
            
        
        

    def dxdw(self, lx, jx, lw, jw, iw):
        """Calculate dxdw.
        lx, jx, lw, jw, iw refer to which 
        """        
        if (lx == lw):
            if (jx == jw):
                x0 = self.layer[lx].x[jx]
                xm1 = self.layer[lx-1].x[iw]
                deriv = xm1*x0*(1.0-x0)
                return deriv
            else:
                return 0.0
        if (lx > lw):
            ds = 0.0
            nl_m1 = self.layer[lx-1].nn
            for k in np.arange(nl_m1):
                deriv_m1 = self.dxdw(lx = lx-1, \
                    jx = k, lw = lw, jw = jw, iw = iw)
                w = self.layer[lx].w[jx, k]
                ds = ds + w*deriv_m1
            x0 = self.layer[lx].x[jx]
            deriv = x0*(1.0-x0)*ds
            return(deriv)

            
    def testfunc(self):
         print('sadf')

        
    def train_net(self, x0, a):
        self.load_training_set(x0, a)
        self.calc_all_x()
        loss = np.sum((self.layer[-1].x - a)**2.0)

        nn_last = self.layer[-1].nn
        nl = self.nlayer
        for lw in np.arange(1,nl): ## no weights in first layer.
            #print("Updating weights in layer "+str(lw+1)+' of '+str(nl))
            nn = self.layer[lw].nn
            nn_m1 = self.layer[lw].nn_m1
            for jw in np.arange(nn):
                for iw in np.arange(nn_m1):
                    dlossdw = 0.0
                    for jx in np.arange(nn_last):
                        if nn_last > 1:
                            dlossdw = dlossdw \
                              + 2.0*(self.layer[-1].x[jx] - self.a[jx]) \
                              * self.dxdw(lx=nl-1, jx=jx, lw=lw, jw=jw, iw=iw)
                        else:
                            dlossdw = dlossdw \
                              + 2.0*(self.layer[-1].x - self.a) \
                              * self.dxdw(lx=nl-1, jx=jx, lw=lw, jw=jw, iw=iw)
                             
                    # now update weight:
                    dw = 0.0
                    if np.abs(dlossdw) > 0.0:
                        dw = - loss/dlossdw
                    corr = dw/(1.0 + self.damp*np.abs(dw))
                    corr = corr*self.damp2
                    w = self.layer[lw].w[jw,iw]
                    self.layer[lw].w[jw,iw] =  w + corr


        



