import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Synthetic sequence: each sample has 10 time steps and 1 sensor reading.
rng=np.random.default_rng(12)
raw=np.sin(np.arange(0,500)*0.08)+rng.normal(0,0.05,500)
X=[]; y=[]
for i in range(len(raw)-10):
    X.append(raw[i:i+10]); y.append(raw[i+10])
X=np.array(X).reshape(-1,10,1); y=np.array(y)
model=Sequential([SimpleRNN(16,activation='tanh',input_shape=(10,1)),Dense(1)])
model.compile(optimizer='adam',loss='mse')
model.fit(X,y,epochs=15,batch_size=16,validation_split=0.2,verbose=0)
pred=model.predict(X[-5:],verbose=0).ravel()
print('Last five actual next readings:',np.round(y[-5:],3))
print('RNN predicted readings:',np.round(pred,3))
print('Final training loss:',round(float(model.evaluate(X,y,verbose=0)),5))
