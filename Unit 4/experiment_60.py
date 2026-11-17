import numpy as np
from sklearn.metrics import mean_squared_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input,SimpleRNN,Dense
rng=np.random.default_rng(60); t=np.arange(450); temp=24+4*np.sin(t/12)+rng.normal(0,.5,450)
for w in [5,10,20]:
 X=np.array([temp[i:i+w] for i in range(len(temp)-w)]); y=temp[w:]; s=int(.8*len(X))
 m=Sequential([Input((w,1)),SimpleRNN(18),Dense(1)]); m.compile(optimizer="adam",loss="mse")
 m.fit(X[:s,...,None],y[:s],epochs=12,verbose=0)
 p=m.predict(X[s:,...,None],verbose=0).ravel()
 print("Sequence length:",w,"Test MSE:",round(mean_squared_error(y[s:],p),4))
