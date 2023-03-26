## Zabawy z sieciami neuronowymi

Choć uczenie maszynowe nie należy do najprostszych tematów, podstawowa zasada dziłania sieci neurownowych, może być zawarta w jednym zdaniu:
*Dążymy do znalezienia **minimum funkcji kosztu**, które jest reprezentowane przez wartości wag, dla których funkcja kosztu przyjmuje najmniejszą wartość.*

#### **Przykładowy, prosty model rozpoznawania pisma**

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

Poniższe linie to standaryzacja danych, czyli normalizacja wartości pikseli do zakresu od 0 do 1. Celem normalizacji jest ułatwienie procesu uczenia modelu. Pozwala ona na szybsze zbieganie algorytmów optymalizacyjnych do minimum globalnego. Ponadto, w niektórych zbiorach danych, normalizacja jest wymagana celem zmniejszenia wpływu wartości odstających oraz umożliwienia porównywania danych różniących się skalą/jednostką/zakresem. 

```
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32') 
x_train /= 255
x_test /= 255
```

Aby móc przetwarzać dane z obrazów 28x28 pikseli przez sieć typu DNN, należy przekształcić ją na jednowymiarową macierz o długości 784 pikseli.

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

Tworzony jest model sieci neuronowej, składający się z dwóch warstw. Warstwy te są w pełni połączonymi warstwami neuronów, gdzie każdy neuron w warstwie jest połączony ze wszystkimi neuronami w poprzedniej warstwie. Pierwsza warstwa ma 10 neuronów i używa funkcji aktywacji sigmoid - druga warstwa również ma 10 neuronów i używa funkcji aktywacji softmax. Pierwsza warstwa jest warstwą wejściową, której rozmiar jest równy liczbie pikseli obrazów cyfr. 

Funkcja sigmoid przetwarza dane wejściowe (tj. piksele obrazów cyfr) na wartości z zakresu 0-1, które interpretujemy jako prawdopodobieństwo, że dany neuron reprezentuje daną cyfrę. Funkcja softmax natomiast normalizuje wyniki funkcji sigmoid w taki sposób, że suma wszystkich wartości wynosi 1. Dzięki temu wyniki tej funkcji możemy interpretować jako rozkład prawdopodobieństwa dla wszystkich klas.

Wzory funkcji aktywacji:

**Sigmoid** 

$\sigma(z) = \frac{1} {1 + e^{-z}}$

**Softmax**

$\sigma(z_i) = \frac{e^{z_{i}}}{\sum_{j=1}^K e^{z_{j}}} \ \ \ for\ i=1,2,\dots,K$

gdzie $\mathbf{x}$ to wektor wejściowy, a $K$ to liczba klas. Funkcja softmax normalizuje wartości wejściowe tak, że ich suma wynosi 1, co pozwala na interpretację wyników jako prawdopodobieństw przynależności do poszczególnych klas.

Alternatywnie, zamiast funkcji sigmoidalnej można użyć funkcji ReLU, która lepiej sprawdza się przy większej liczbie warstw:

$Relu(z) = max(0, z)$

Ostatnim krokiem jest kompilacja i tzw. *fitowanie* modelu:

```
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size=100, epochs=5)
```

Jako funkcja straty ustawiona została `categorical_crossentropy`, optymalizator jako `sgd` (stochastic gradient descent) i metrykę jako `accuracy`. Dokładne znaczenie tych terminów wyjaśniam poniżej.

___

**Co to jest funkcja straty?** 

Najogólniej są to funkcje pozwalające porównać etykiety klas predykowanych i rzeczywistych. Przytaczana funkcja cross-entropi jest popularnym narzędziem stosowanym w problemach klasyfikacji wieloklasowej. 

**Co to jest optymalizator?**

Optymalizatory w sieciach neuronowych to algorytmy służące do aktualizacji wag sieci w trakcie procesu uczenia się, w celu minimalizacji funkcji kosztu. 

**Co to jest metryka?**

Metryki w uczeniu maszynowym to liczbowe wskaźniki służące do oceny jakości modelu uczącego się na podstawie zestawu danych. Metryki są używane do mierzenia, jak dobrze model dopasowuje się do danych treningowych oraz jak dobrze generalizuje na nowych, niewidzianych danych testowych.

Zastosowana w ewaluacji modelu metryka `accuracy` oznacza procentową liczbę poprawnie sklasyfikowanych przykładów w zbiorze testowym:

$Accuracy = \frac{TP+TN}{TP+TN+FP+FN}$


