import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input,LSTM,Dense
rng=np.random.default_rng(61); v=100+12*np.sin(np.arange(400)/9)+rng.normal(0,2,400); w=14
X=np.array([v[i:i+w] for i in range(len(v)-w)]); y=v[w:]; s=310
for units,lr,batch in [(16,.001,16),(32,.001,32),(32,.0005,16)]:
 tf.keras.backend.clear_session()
 m=Sequential([Input((w,1)),LSTM(units),Dense(1)])
 m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),loss="mse")
 m.fit(X[:s,...,None],y[:s],epochs=10,batch_size=batch,verbose=0)
 loss=m.evaluate(X[s:,...,None],y[s:],verbose=0)
 print(f"units={units}, learning_rate={lr}, batch_size={batch}, test_loss={loss:.5f}")
