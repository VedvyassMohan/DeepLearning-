import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Example daily temperature series (synthetic for demonstration).
rng=np.random.default_rng(20)
days=np.arange(0,240)
temperature=25+5*np.sin(2*np.pi*days/30)+rng.normal(0,0.8,len(days))
window=7
X=[]; y=[]
for i in range(len(temperature)-window):
    X.append(temperature[i:i+window]); y.append(temperature[i+window])
X=np.array(X).reshape(-1,window,1); y=np.array(y)
split=int(len(X)*0.8)
X_train,X_test=X[:split],X[split:]; y_train,y_test=y[:split],y[split:]
mean,std=X_train.mean(),X_train.std()
X_train=(X_train-mean)/std; X_test=(X_test-mean)/std
y_train=(y_train-mean)/std; y_test_raw=y_test.copy(); y_test=(y_test-mean)/std
model=Sequential([SimpleRNN(24,activation='tanh',input_shape=(window,1)),Dense(1)])
model.compile(optimizer='adam',loss='mse',metrics=['mae'])
model.fit(X_train,y_train,epochs=25,batch_size=16,verbose=0)
pred=model.predict(X_test,verbose=0).ravel()*std+mean
mae=np.mean(np.abs(pred-y_test_raw))
print('Actual next temperatures:',np.round(y_test_raw[:10],2))
print('Predicted temperatures:',np.round(pred[:10],2))
print('Test MAE (degrees):',round(float(mae),3))
