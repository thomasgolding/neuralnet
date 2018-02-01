from unittest import TestCase
import neuralnet as n
import numpy as np
from scipy.stats import norm
from scipy.stats import uniform


class TestNeuralNet(TestCase):


    def test_deriv_analytic(self):
        nnet = n.NeuralNet(nneuron = (2,2,2))
        nnet.init_x0((-2.,3.))

        for il in np.arange(3):
            nnet.set_b(il, (0.0, 0.0))

        #w2 = np.zeros((2,2))
        w2 = np.reshape(norm.rvs(0,1,4), (2,2))
        #w3 = -w2.copy()
        w3 = np.reshape(norm.rvs(0,1,4), (2,2))
        nnet.set_w(il=1, w=w2)
        nnet.set_w(il=2, w=w3)
        nnet.calc_all_x()

        lx=2; jx=0; lw=2; jw=0; iw=0
        deriv = nnet.dxdw(lx=lx, jx=jx, lw=lw, jw=jw, iw=iw)
        # should equal
        x = nnet.layer[lx].x[jw]
        x_m1 = nnet.layer[lx-1].x[iw]
        deriv2 = x*(1.-x)*x_m1
        reldiff = abs((deriv2-deriv)/deriv)
        self.assertTrue(reldiff < 1.e-6)

    def test_deriv_numeric(self):
        nnet = n.NeuralNet(nneuron = (2,2,2))
        nnet.init_x0((-2.,3.))

        for il in np.arange(3):
            nnet.set_b(il, (0.0, 0.0))

        w2 = np.reshape(norm.rvs(1.,0.1,4), (2,2))
        w3 = np.reshape(norm.rvs(-1.,0.1,4), (2,2))

        #w2[:,:] = 0.1
        #w3[:,:] = -0.3
        nnet.set_w(il=1, w=w2)
        nnet.set_w(il=2, w=w3)
        nnet.calc_all_x()

        lx=2; jx=0; lw=1; jw=0; iw=0
        ddw = 1.e-3
        deriv = nnet.dxdw(lx=lx, jx=jx, lw=lw, jw=jw, iw=iw)
        x0 = nnet.layer[lx].x[jx]
        w = nnet.layer[lw].w[jw,iw]
        nnet.layer[lw].w[jw,iw] = w*(1.0 + ddw)
        nnet.calc_all_x()
        x1 = nnet.layer[lx].x[jx]
        deriv2 = (x1-x0)/(ddw*w)
        reldiff = abs((deriv2-deriv)/deriv)
        print("(deriv, deriv2, reldiff)")
        print(deriv, deriv2, reldiff)
        self.assertTrue(reldiff < 1.e-3)

    def test_one_layerneuron(self):
        nneuron = (1,1)
        nnet = n.NeuralNet(nneuron = nneuron)
        nnx = 50
        nntrain = 40
        xx0 = 0.8
        ww0 = -np.log(1./0.99  - 1.)/xx0

        for il in np.arange(len(nneuron)):
            nnet.set_b(il, np.zeros(nneuron[il]))
            
        nnet.init_w_random()
        xin = uniform.rvs(0.00, 1.0, (nnx))
        xout = 1./(1.+np.exp(-ww0*xin))

        for i in np.arange(nntrain):
            nnet.train_net(x0 = xin[i], a = xout[i])

        reldiff = np.abs((nnet.layer[-1].w[0,0] - ww0)/ww0)
        self.assertTrue(reldiff < 1.e-4)

        
        
