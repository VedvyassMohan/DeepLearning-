import numpy as np
from sklearn.metrics import mean_squared_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input,SimpleRNN,LSTM,GRU,Dense
rng=np.random.default_rng(57); v=np.sin(np.arange(500)/10)+rng.normal(0,.08,500); w=16
X=np.array([v[i:i+w] for i in range(len(v)-w)]); y=v[w:]; s=390
for name,layer in [("RNN",SimpleRNN(20)),("LSTM",LSTM(20)),("GRU",GRU(20))]:
 m=Sequential([Input((w,1)),layer,Dense(1)]); m.compile(optimizer="adam",loss="mse")
 m.fit(X[:s,...,None],y[:s],epochs=15,verbose=0)
 p=m.predict(X[s:,...,None],verbose=0).ravel()
 print(name,"Test MSE:",round(mean_squared_error(y[s:],p),5))
