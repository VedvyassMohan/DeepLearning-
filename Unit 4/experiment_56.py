import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input,GRU,Dense
rng=np.random.default_rng(56); t=np.arange(600)
data=np.column_stack([25+4*np.sin(t/20)+rng.normal(0,.4,600),60-8*np.sin(t/20)+rng.normal(0,1,600),1010+3*np.cos(t/25)+rng.normal(0,.6,600)])
w=12; X=np.array([data[i:i+w] for i in range(len(data)-w)]); y=data[w:]; s=480
mu=X[:s].mean((0,1)); sd=X[:s].std((0,1))
m=Sequential([Input((w,3)),GRU(24),Dense(3)]); m.compile(optimizer="adam",loss="mse")
m.fit((X[:s]-mu)/sd,(y[:s]-mu)/sd,epochs=15,verbose=0)
p=m.predict(((X[s:]-mu)/sd)[-1:],verbose=0)[0]*sd+mu
print("Forecast [temperature, humidity, pressure]:",np.round(p,2))
print("Actual:",np.round(y[-1],2))
