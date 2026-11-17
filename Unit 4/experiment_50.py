import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input,SimpleRNN,Dense
text=("deep learning learns patterns. recurrent neural networks process sequences. ")*20
chars=sorted(set(text)); c2i={c:i for i,c in enumerate(chars)}; i2c={i:c for c,i in c2i.items()}
L=25; X=[]; y=[]
for i in range(len(text)-L):
 X.append([c2i[c] for c in text[i:i+L]]); y.append(c2i[text[i+L]])
X=np.eye(len(chars))[np.array(X)]; y=np.array(y)
m=Sequential([Input((L,len(chars))),SimpleRNN(48),Dense(len(chars),activation="softmax")])
m.compile(optimizer="adam",loss="sparse_categorical_crossentropy")
m.fit(X,y,epochs=6,batch_size=64,verbose=0)
generated=text[:L]
for _ in range(100):
 ids=[c2i[c] for c in generated[-L:]]
 p=m.predict(np.eye(len(chars))[ids][None,:,:],verbose=0)[0]
 generated+=i2c[int(np.random.choice(len(chars),p=p/p.sum()))]
print("Generated text:\n",generated)
