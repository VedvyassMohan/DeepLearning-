import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input,SimpleRNN,Dense
rng=np.random.default_rng(52); t=np.arange(500)
d=500+50*np.sin(2*np.pi*t/7)+20*np.sin(t/40)+rng.normal(0,10,500); w=14
X=np.array([d[i:i+w] for i in range(len(d)-w)]); y=d[w:]; s=400
mu,sd=X[:s].mean(),X[:s].std()
Xtr=((X[:s]-mu)/sd)[...,None]; Xte=((X[s:]-mu)/sd)[...,None]; yt=(y[:s]-mu)/sd
m=Sequential([Input((w,1)),SimpleRNN(24),Dense(1)]); m.compile(optimizer="adam",loss="mse")
m.fit(Xtr,yt,epochs=15,verbose=0)
print("Forecast demand:",round(float(m.predict(Xte[-1:],verbose=0)[0,0]*sd+mu),2),"units")
