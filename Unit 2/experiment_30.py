import numpy as np

x=np.array([[1.,0.]]) ; target=np.array([[1.]])
W1=np.array([[.2,.4],[.3,.1]]); b1=np.zeros((1,2)); W2=np.array([[.5],[.6]]); b2=np.zeros((1,1)); lr=.1
sig=lambda z:1/(1+np.exp(-z))
h=sig(x@W1+b1); yhat=sig(h@W2+b2)
d2=(yhat-target)*yhat*(1-yhat); dW2=h.T@d2; db2=d2
d1=(d2@W2.T)*h*(1-h); dW1=x.T@d1; db1=d1
W2-=lr*dW2; b2-=lr*db2; W1-=lr*dW1; b1-=lr*db1
print('Prediction before update:',round(float(yhat[0,0]),4))
print('Updated W1:\n',np.round(W1,4)); print('Updated W2:\n',np.round(W2,4))
