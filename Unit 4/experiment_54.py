import numpy as np
from sklearn.metrics import mean_squared_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input,SimpleRNN,LSTM,Dense
rng=np.random.default_rng(54); sales=100+15*np.sin(np.arange(450)/8)+rng.normal(0,4,450); w=15
X=np.array([sales[i:i+w] for i in range(len(sales)-w)]); y=sales[w:]; s=340
for name,layer in [("RNN",SimpleRNN(20)),("LSTM",LSTM(20))]:
 m=Sequential([Input((w,1)),layer,Dense(1)]); m.compile(optimizer="adam",loss="mse")
 m.fit(X[:s,...,None],y[:s],epochs=12,verbose=0)
 p=m.predict(X[s:,...,None],verbose=0).ravel()
 print(name,"Test MSE:",round(mean_squared_error(y[s:],p),4))
