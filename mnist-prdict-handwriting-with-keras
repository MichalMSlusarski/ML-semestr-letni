import keras 
from keras.datasets import mnist 
import matplotlib.pyplot as plt 
from keras.utils import to_categorical
from keras.models import Sequential 
from keras.layers.core import Dense, Activation
import numpy as np

# ładowanie danych z mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data() 

# standaryzacja:
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32') 
x_train /= 255
x_test /= 255

# 
x_train = x_train.reshape(60000, 784) 
x_test = x_test.reshape(10000, 784) 

y_train = to_categorical(y_train, num_classes=10) 
y_test = to_categorical(y_test, num_classes=10)

# budowa modelu
model = Sequential() 
model.add(Dense(10, activation='sigmoid', input_shape=(784,))) # pierwsza warstwa
model.add(Dense(10, activation='softmax')) # druga warstwa

# funkcja straty
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics = ['accuracy'])

# fitowanie modelu (dane treningowe)
model.fit(x_train, y_train, batch_size=100, epochs=5)

# ewaluacja modelu
test_loss, test_acc = model.evaluate(x_test, y_test)

print('Test accuracy:', test_acc) 

print(np.shape(x_train),np.shape(x_test))
(x_train, y_train), (x_test, y_test) = mnist.load_data() 

plt.imshow(x_test[1], cmap=plt.cm.binary)

#predictions = model.predict(x_test)
#print(np.argmax(predictions[1]) )
#print(predictions[1])
