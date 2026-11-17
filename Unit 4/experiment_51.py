import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input,SimpleRNN,Dense
rng=np.random.default_rng(51); t=np.arange(450)
v=100+.05*t+4*np.sin(t/12)+rng.normal(0,1,450); w=20
X=np.array([v[i:i+w] for i in range(len(v)-w)]); y=v[w:]; s=340
mu,sd=X[:s].mean(),X[:s].std()
a=((X[:s]-mu)/sd)[...,None]; b=((X[s:]-mu)/sd)[...,None]; yt=(y[:s]-mu)/sd
m=Sequential([Input((w,1)),SimpleRNN(24),Dense(1)]); m.compile(optimizer="adam",loss="mse")
m.fit(a,yt,epochs=15,verbose=0); p=m.predict(b,verbose=0).ravel()*sd+mu
print("MAE:",round(mean_absolute_error(y[s:],p),3))
print("RMSE:",round(mean_squared_error(y[s:],p)**.5,3))
print("Actual:",np.round(y[s:s+5],2)); print("Predicted:",np.round(p[:5],2))
