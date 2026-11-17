import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
texts=["excellent product","very happy purchase","good quality","love this item","terrible product","poor quality","waste of money","very bad experience","works perfectly","highly recommended","damaged item","not satisfied"]
y=np.array([1,1,1,1,0,0,0,0,1,1,0,0])
tok=Tokenizer(num_words=1000,oov_token="<OOV>"); tok.fit_on_texts(texts)
X=pad_sequences(tok.texts_to_sequences(texts),maxlen=6)
model=Sequential([Embedding(1000,16),SimpleRNN(16),Dense(1,activation="sigmoid")])
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
model.fit(X,y,epochs=20,verbose=0)
test=["excellent and useful","poor damaged item"]
Xt=pad_sequences(tok.texts_to_sequences(test),maxlen=6)
for s,p in zip(test,model.predict(Xt,verbose=0).ravel()):
 print(s,"=>","Positive" if p>=.5 else "Negative",round(float(p),3))
