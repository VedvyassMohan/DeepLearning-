import numpy as np

X=np.array([[1,1],[2,1],[1,2],[0,0]],float); y=np.array([0,1,1,0])
w=np.zeros(2); b=0.; lr=.1
for _ in range(30):
    for xi,target in zip(X,y):
        pred=int(np.dot(xi,w)+b>=1.5)
        e=target-pred; w+=lr*e*xi; b+=lr*e
pred=(X@w+b>=1.5).astype(int)
print('Weights:',np.round(w,2),'Bias:',round(b,2))
print('Predictions:',pred,'Actual:',y)
