## Opis przykładowego projektu w Keras:

Na początku importowane są potrzebne biblioteki, w tym Keras - bibliotekę do budowania sieci neuronowych, numpy - bibliotekę do operacji na macierzach, matplotlib - bibliotekę do rysowania wykresów.

```
import keras 
from keras.datasets import mnist 
import matplotlib.pyplot as plt 
from keras.utils import to_categorical
from keras.models import Sequential 
from keras.layers.core import Dense, Activation
import numpy as np
```

Następnie, ładowane są dane MNIST i dzielone na zbiory treningowy i testowy. Zbiór treningowy zawiera 60 000 obrazów cyfr, a zbiór testowy - 10 000 obrazów.

```
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

Poniższe linie to standaryzacja danych, czyli normalizację wartości pikseli do zakresu od 0 do 1.

```
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32') 
x_train /= 255
x_test /= 255
```

Aby móc przetwarzać dane z obrazów 28x28 pikseli przez sieć typu DNN, należy przekształcić ją na jednowymiarową macierz o długości 784 pikseli. Oraz 

```
x_train = x_train.reshape(60000, 784) 
x_test = x_test.reshape(10000, 784)
```

Kolejnym krokiem jest zakodowanie wartości cyfr na wektory binarne za pomocą funkcji `to_categorical` z biblioteki Keras. Dzięki temu możliwe jest przetwarzanie wyników przez sieć neuronową.

```
y_train = to_categorical(y_train, num_classes=10) 
y_test = to_categorical(y_test, num_classes=10)
```

W końcu przychodzi pora na stworzenie modelu: 

```
model = Sequential() 
model.add(Dense(10, activation='sigmoid', input_shape=(784,))) # pierwsza warstwa (wejściowa)
model.add(Dense(10, activation='softmax')) # druga warstwa
```

Tworzony jest model sieci neuronowej, składający się z dwóch warstw. Pierwsza warstwa ma 10 neuronów i używa funkcji aktywacji sigmoid - druga warstwa również ma 10 neuronów i używa funkcji aktywacji softmax. Pierwsza warstwa jest warstwą wejściową, której rozmiar jest równy liczbie pikseli obrazów cyfr.

Wzory funkcji:

**Sigmoid** 

σ(x)=1+e−x1​

**Softmax**

σ(x)i​=∑j=1K​exj​exi​​,dlai=1,…,K

Ostatnim krokiem jest kompilacja i tzw. *fitowanie* modelu:

```
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size=100, epochs=5)
```

Jako funkcja straty ustawiona została `categorical_crossentropy`, optymalizator jako `sgd` (stochastic gradient descent) i metrykę jako `accuracy`.


