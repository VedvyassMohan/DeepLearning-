import numpy as np

# Train a small 2-2-1 network using backpropagation
X=np.array([[0,0],[0,1],[1,0],[1,1]],float); y=np.array([[0],[1],[1],[0]],float)
rng=np.random.default_rng(2); W1=rng.normal(size=(2,4)); b1=np.zeros((1,4)); W2=rng.normal(size=(4,1)); b2=np.zeros((1,1)); lr=.5
sig=lambda z: 1/(1+np.exp(-z))
for _ in range(12000):
    h=sig(X@W1+b1); o=sig(h@W2+b2); dz2=(o-y)*o*(1-o); dW2=h.T@dz2/4; db2=dz2.mean(0,keepdims=True); dz1=(dz2@W2.T)*h*(1-h); dW1=X.T@dz1/4; db1=dz1.mean(0,keepdims=True)
    W2-=lr*dW2; b2-=lr*db2; W1-=lr*dW1; b1-=lr*db1
print('Predicted probabilities:\n',np.round(sig(sig(X@W1+b1)@W2+b2),3))
print('Predicted classes:',(sig(sig(X@W1+b1)@W2+b2)>.5).astype(int).ravel())
