import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Small illustrative parallel dataset: English -> French
source = ["hello", "good morning", "thank you", "how are you", "good night"]
target = ["bonjour", "bonjour", "merci", "comment allez vous", "bonne nuit"]
src_tok = Tokenizer(filters=""); src_tok.fit_on_texts(source)
tgt_tok = Tokenizer(filters=""); tgt_tok.fit_on_texts(target)
src = pad_sequences(src_tok.texts_to_sequences(source), padding="post")
tgt = pad_sequences(tgt_tok.texts_to_sequences(target), padding="post")
# Add start/end markers to decoder sequences
start_id = len(tgt_tok.word_index) + 1
decoder_in = np.zeros((len(target), tgt.shape[1] + 1), dtype="int32")
decoder_out = np.zeros_like(decoder_in)
for i, row in enumerate(tgt):
    seq = [start_id] + [int(x) for x in row if x != 0]
    decoder_in[i, :len(seq)] = seq
    decoder_out[i, :len(seq)-1] = seq[1:]
src_vocab = len(src_tok.word_index) + 1
tgt_vocab = start_id + 1
enc_input = Input(shape=(src.shape[1],))
enc_embed = __import__("tensorflow").keras.layers.Embedding(src_vocab, 32)(enc_input)
_, h, c = LSTM(64, return_state=True)(enc_embed)
dec_input = Input(shape=(decoder_in.shape[1],))
dec_embed = __import__("tensorflow").keras.layers.Embedding(tgt_vocab, 32)(dec_input)
dec_out, _, _ = LSTM(64, return_sequences=True, return_state=True)(dec_embed, initial_state=[h, c])
output = Dense(tgt_vocab, activation="softmax")(dec_out)
model = Model([enc_input, dec_input], output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit([src, decoder_in], decoder_out[..., None], epochs=150, verbose=0)
print("Training samples:", len(source))
print("Final training loss and accuracy:", model.evaluate([src, decoder_in], decoder_out[..., None], verbose=0))
print("Encoder-decoder model trained on the small example parallel corpus.")
