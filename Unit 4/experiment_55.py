import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input,LSTM,Dense
rng=np.random.default_rng(55); t=np.arange(550); price=150+.03*t+6*np.sin(t/17)+rng.normal(0,1,550); w=20
X=np.array([price[i:i+w] for i in range(len(price)-w)]); y=price[w:]; s=420
m=Sequential([Input((w,1)),LSTM(28),Dense(1)]); m.compile(optimizer="adam",loss="mse")
m.fit(X[:s,...,None],y[:s],epochs=18,verbose=0); p=m.predict(X[s:,...,None],verbose=0).ravel()
print("MAE:",round(mean_absolute_error(y[s:],p),3)); print("MSE:",round(mean_squared_error(y[s:],p),3))
print("RMSE:",round(mean_squared_error(y[s:],p)**.5,3)); print("R2:",round(r2_score(y[s:],p),3))
