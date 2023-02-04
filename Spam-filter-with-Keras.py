# Analiza spamu na podstawie: K. R. Bokka, S. Hora, T. Jain, M. Wambugu: Deep Learning for Natural Language Processing. Packt Publishing 2019.
# Plik do wykorzystania: spam.csv

import pandas as pd
import numpy as np
from keras.models import Model, Sequential
from keras.layers import LSTM, Dense,Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras_preprocessing.sequence import pad_sequences
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image

from google.colab import files
uploaded=files.upload()

spam = pd.read_csv("spam.csv", encoding="latin")

spam = spam[["v1","v2"]]

spam["v1"].value_counts()

# Przypisanie wartości 0 i 1 do etykiet "ham" i "spam".
lab_map = {"ham":0, "spam":1}
Y = spam["v1"].map(lab_map).values
X = spam["v2"].values

# Tokenizacja tekstu
max_words = 300
mytokenizer = Tokenizer(num_words=max_words,lower=True, split=" ")
mytokenizer.fit_on_texts(X)
text_tokenized = mytokenizer.texts_to_sequences(X)

text_tokenized

# Wszystkie sekwencje, krótsze niż 50 znaków, zostaną wypełnione zerami lub innymi znakami w celu osiągnięcia długości 50 znaków.
max_len = 50
sequences = pad_sequences(text_tokenized, maxlen=max_len)

sequences

# Tworzenie modelu sieci neuronowej.
model = Sequential()
model.add(Embedding(max_words, 20, input_length=max_len))
model.add(LSTM(64))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy',
optimizer='adam',
metrics=['accuracy'])

# Trenowanie modelu
model.fit(sequences,Y,batch_size=128,epochs=10, validation_split=0.2)

# Testowanie modelu
inp_test_seq = "WINNER! U win a 500 prize reward & free entry to FA cup final tickets! Text FA to 34212 to receive award"
test_sequences = mytokenizer.texts_to_sequences(np.array([inp_test_seq]))
test_sequences_matrix = pad_sequences(test_sequences,maxlen=max_len)
model.predict(test_sequences_matrix)
