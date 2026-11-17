import numpy as np

X=np.array([[0.5,0.8]])
W1=np.array([[0.2,0.4],[0.3,0.1]]); b1=np.array([0.1,0.2])
W2=np.array([[0.5],[0.6]]); b2=np.array([0.1])
sig=lambda z:1/(1+np.exp(-z))
h=sig(X@W1+b1); output=sig(h@W2+b2)
print('Hidden layer output:',np.round(h,4))
print('Output layer value:',np.round(output,4))
