import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
corpus=("language models predict the next word. recurrent networks learn sequences. ")*25
tok=Tokenizer(oov_token="<OOV>"); tok.fit_on_texts([corpus]); ids=tok.texts_to_sequences([corpus])[0]; vocab=len(tok.word_index)+1
seq=[ids[i-3:i+1] for i in range(3,len(ids))]
a=pad_sequences(seq,padding="pre"); X,y=a[:,:-1],a[:,-1]
m=Sequential([Embedding(vocab,16,input_length=3),SimpleRNN(24),Dense(vocab,activation="softmax")])
m.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
m.fit(X,y,epochs=8,verbose=0)
p=pad_sequences(tok.texts_to_sequences(["language models predict"]),maxlen=3)
word=tok.index_word.get(int(m.predict(p,verbose=0)[0].argmax()),"?")
print("Input: language models predict"); print("Suggested next word:",word)
print("Training accuracy:",round(float(m.evaluate(X,y,verbose=0)[1]),3))
