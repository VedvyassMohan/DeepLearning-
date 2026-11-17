import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input,SimpleRNN,Dense
rng=np.random.default_rng(58); t=np.arange(500)
d=np.column_stack([50+4*np.sin(t/15)+rng.normal(0,.3,500),100+6*np.cos(t/20)+rng.normal(0,.6,500),3+np.sin(t/7)+rng.normal(0,.1,500)])
w=12; X=np.array([d[i:i+w] for i in range(len(d)-w)]); y=d[w:]; s=400
mu=X[:s].mean((0,1)); sd=X[:s].std((0,1))
m=Sequential([Input((w,3)),SimpleRNN(24,return_sequences=True),SimpleRNN(12),Dense(3)])
m.compile(optimizer="adam",loss="mse"); m.fit((X[:s]-mu)/sd,(y[:s]-mu)/sd,epochs=15,verbose=0)
p=m.predict(((X[s:]-mu)/sd)[-1:],verbose=0)[0]*sd+mu
print("Predicted [temperature, pressure, vibration]:",np.round(p,2)); print("Actual:",np.round(y[-1],2))
