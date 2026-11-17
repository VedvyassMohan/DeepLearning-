import numpy as np

x=np.array([-2.,-1.,0.,1.,2.])
sigmoid=1/(1+np.exp(-x)); tanh=np.tanh(x); relu=np.maximum(0,x)
print('Input:   ',x)
print('Sigmoid: ',np.round(sigmoid,3))
print('Tanh:    ',np.round(tanh,3))
print('ReLU:    ',relu)
