# Machine Learning (notatki)

Choć uczenie maszynowe nie należy do najprostszych tematów, podstawowa zasada działania sieci neuronowych, może być zawarta w jednym zdaniu:
*Dążymy do znalezienia **minimum funkcji straty**, czyli takich wartości **wag**, dla których funkcja kosztu przyjmuje najmniejszą wartość.*

### Zawartość repozytorium

- Sieć od podstaw
- Wprowadzenie do Keras
- Word Embeddings

## Sieć od podstaw

Choć współcześnie krajobraz uczenia maszynowego zdominowany jest przez wysokopoziomowe biblioteki, takie jak Tensorflow czy PyTorch, na samym początku warto poeksperymentować z sieciami pisanymi od podstaw.  Implementacja prostej sieci neuronowej od podstaw, pozwala poznać podstawowe koncepcje, takie jak propagacja wsteczna, funkcje aktywacji, ważenie synaptyczne itp.

Implementacje prostych sieci znajdują się w folderze: ![Neural-Network-From-Scratch](https://github.com/MichalMSlusarski/ML-semestr-letni/tree/main/Neural-Network-From-Scratch)

Opis poszczególnych etapów tworzenia sieci, eksperymentów i definiowanych funkcji.

### Przygotowanie danych

Do treningu i modyfikacji sieci otrzymaliśmy zbior testowy basic.csv, zawiarający podstawowe metryki postów na Fb. Naszym zadaniem jest sprawdzeni, oczyszczenie i podział zbioru oraz zdefiniowanie wartości przewidywanej. **Od razu warto zaznaczyć, że zbiór jest bardzo mały - dla normalnej sieci zbyt mały aby uzyskać dobre wyniki.**

Pracę z danymi zaczynam od wizualnego przejrzenia zbioru. Określiłem, że znajdują się tam dwa rzędy wyjaśniające nazwę i typ prezentowanej zmiennej. Jest też sporo wartości pustych, najprawdopodobniej reprezentujących brak wartości.

Celem oczyszczenia zbioru, utworzyłem skrypt clean_data. W ramach operacji, ładuję nieoczyszczony zbiór basic.csv. Przeprowadzam podstawowe czyszczenie: usunięcie rzędów wyjaśniających oraz kolumny Z13; uzupełnienie braków danych zerami oraz profilaktyczne unormowanie typu danych w tabeli (to_numeric).
```python
import pandas as pd
import numpy as np

df = pd.read_csv('basic.csv')

df = df.drop([0, 1])
df = df.reset_index(drop=True)
df = df.drop('Z13', axis=1)
df = df.fillna(0)
df = df.apply(pd.to_numeric)
```
Następnym krokiem jest standaryzacja danych, tak aby wartości mieściły się w podobnym zakresie. Bez standaryzacji wysokie wartości mogłyby zdominować proces uczenia się sieci.

Do standaryzacji wybieram kolumny liczbowe (za kategoryczne uznaję wszystkie: 0-1). Korzystam z metody max-min, która przeskalowuje wartości w określonych kolumnach do standardowego zakresu od 0 do 1. Odbywa się to poprzez odjęcie wartości minimalnej w kolumnach i jej podzielenie przez zakres wartości tam znajdowanych.
```python
columns_to_standardize = ['Z11', 'Z14', 'Z15', 'Z16', 'Z17', 'Z18', 'Z19', 'Z20']
df[columns_to_standardize] = round((df[columns_to_standardize] - df[columns_to_standardize].min()) / (df[columns_to_standardize].max() - df[columns_to_standardize].min()), 6)
```
Zamieniam zmienną Z21 z numerycznej na kategoryczną, za pułap przyjmując medianę wartości liczby komentarzy. Dzięki temu można podzielić zbiór na dwie mniej więcej równe klasy.
```python
median = df['Z21'].median()
df['Z21'] = np.where(df['Z21'] > median, 1, 0)
```
Następnie sprawdzam, czy uzyskane klasy rzeczywiście są w miarę równoliczne.
```python
print(df['Z21'].value_counts())
```
Dla mediany wychodzą następujące liczebności klas:
0	341
1	301

Dla porównania, przy grupowaniu po średniej, klasy były znacząco nierówne:
0	488
1	154

Wartość mediany to 7. To znaczy, że 341 postów ma równo 7 lub mniej komentarzy. 301 postów ma więcej. Klasy nie mogą być dokładnie równe, gdyż wartości zachodziłyby na siebie. Zadaniem sieci będzie przewidzieć, czy przy danym poście liczba komentarzy jest większa niż mediana zbioru.

Pod koniec procesu przetwarzania, 50 ostatnich pozycji ze zbioru przeznaczam na zbiór testowy. Usuwam je ze zbioru treningowego.
```python
test_df = df[-50:]
df = df[:-50]
```
Zapisuję zbiory.
```python
df.to_csv('training.csv')
test_df.to_csv('test.csv')
```

### Budowa sieci neuronowej

Na początku przetestowałem architekturę opartą o kod z zajęć.
```python
import pandas as pd
import numpy as np

class NeuralNetwork():


    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((11, 1)) - 1


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def sigmoid_derivative(self, x):
        return x * (1 - x)


    def train(self, training_inputs, training_outputs, training_iterations):


        for iteration in range(training_iterations):
            output = self.think(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments


    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output
```

Konstruktor __init__(self) inicjalizuje wagi synaptyczne (synaptic_weights) losowymi wartościami z przedziału [-1, 1].

Funkcja aktywacji sigmoidalnej sigmoid(self, x) oblicza wartość funkcji sigmoidalnej dla danej wartości x.

**Wzór funkcji sigmoidalnej** 

$\sigma(z) = \frac{1} {1 + e^{-z}}$

gdzie $z$ to wektor wejściowy - ważona suma wejść neuronu. Funkcja ta zwraca wartość bliską 0 dla bardzo dużych wartości ujemnych $x$, a wartość bliską 1 dla bardzo dużych wartości dodatnich $x$. Dla $x$ równego zero funkcja sigmoid zwraca wartość 0,5.

Pochodna funkcji sigmoidalnej sigmoid_derivative(self, x) oblicza pochodną funkcji sigmoidalnej dla danej wartości x. Pochodna ta jest używana podczas obliczania dostosowań wag w procesie treningu, wykorzystując propagację wsteczną.

Metoda treningowa train(self, training_inputs, training_outputs, training_iterations) przechodzi przez określoną liczbę iteracji treningowych. Dla każdej iteracji:

- Wykonuje krok "w przód" (forward pass) przez sieć, obliczając wyjście na podstawie wejść treningowych.
- Oblicza błąd jako różnicę między oczekiwanymi wyjściami treningowymi a rzeczywistymi wyjściami sieci.
- Oblicza dostosowania wag na podstawie iloczynu transponowanej macierzy wejść treningowych, błędu oraz pochodnej funkcji sigmoidalnej z wyjścia.
- Aktualizuje wagi synaptyczne, dodając dostosowania.

Metoda predykcji think(self, inputs) przyjmuje wejścia inputs i oblicza wyjście sieci neuronowej. Przekształca wejścia na typ float, wykonuje ważoną sumę wejść (np.dot(inputs, self.synaptic_weights)) i stosuje funkcję sigmoidalną do wyniku. Zwraca obliczone wyjście, które jest interpretowane jako prawdopodobieństwo klasy pozytywnej w przypadku klasyfikacji binarnej.

### Trenowanie modelu

Tym co uległo zmianie, w relacji do skryptu z zajęć, jest sposób wprowadzania i walidacji danych. 

W deklaracji if ```__name__ == "__main__":``` zaczynam od zaimportowania danych z setu treningowego.
```python
df = pd.read_csv("training.csv")
```
Wyciągam zmienne Z9-Z20 jako zmienne objaśniające/predykcyjne i Z21 jako objaśnianą (labels).
```python
input_features = df[['Z9', 'Z10','Z11', 'Z12', 'Z14', 'Z15', 'Z16', 'Z17', 'Z18', 'Z19', 'Z20']]
output_labels = df[['Z21']]
```
Tak jak w oryginalnym skrypcie, wartości z poszczególnych kolumn są przypisywane do wejść i wyjścia sieci.
```python
training_inputs = input_features.values
training_outputs = output_labels.values
```
Inicjalizowany jest obiekt sieci i wywoływana jest metoda trenująca.
```python
neural_network = NeuralNetwork()
neural_network.train(training_inputs, training_outputs, 11000)
```
Uzyskane wagi zapisuję w pliku weights.npy. Rozszerzenie .npy jest częścią biblioteki numpy.
```python
np.save("weights.npy", neural_network.synaptic_weights)
```

### Testowanie modelu

Przechodzę do walidacji modelu. Na początku inicjuję sieć, wczytując zapisane wagi.
```python
neural_network = NeuralNetwork()
neural_network.synaptic_weights = np.load("weights.npy")
```
Analogicznie do zbioru treningowego przygotowuję zbiór testowy.
```python
test_df = pd.read_csv("test.csv")
test_input_features = test_df[['Z9', 'Z10','Z11', 'Z12', 'Z14', 'Z15', 'Z16', 'Z17', 'Z18', 'Z19', 'Z20']]
test_output_labels = test_df[['Z21']]
test_inputs = test_input_features.values
test_outputs = test_output_labels.values
```
Wartości testowe zapuszczane są do modelu i przypisane do zmiennej predictions. Jako że są to wartości zmiennoprzecinkowe, zamieniam je na binarne, wywołując metodę astype(). Wartości, które przekroczą próg 0.5 zliczane są jako 1 - pozostałe jako 0.
```python
predictions = neural_network.think(test_inputs)
binary_predictions = (predictions > 0.5).astype(int)
```
Metryką, którą posłużę się do sprawdzenia modelu jest accuracy, czyli dokładność. Obliczam dokładność modelu wyciągając średnią z przypadków gdzie predykcja jest równa rzeczywistości. Następnie, wynik mnożony jest przez 100, aby otrzymać wartość w procentach.
Na przykład, jeśli accuracy wynosi 85, to oznacza, że model poprawnie sklasyfikował 85% próbek ze zbioru testowego.
```python
accuracy = np.mean(binary_predictions == test_outputs) * 100
print("Mean accuracy:", accuracy)
```
Średnia dokładność modelu wyniosła 70%. Nie jest to wynik wybitny, ale całkiem przyzwoity jak na jedną warstwę i **bardzo mały zbiór treningowy**.

### Modyfikacja modelu

#### Dodawanie warstwy pośredniej

Pierwszą modyfikacją architektury, która mogłaby potencjalnie usprawnić jego dokładność, jest warstwa pośrednia, zwana też ukrytą. Dodanie takiej warstwy rozpoczynam od zadeklarowania nowego zestawu wag dla każdej z warstw. 

Warstwa wejściowa ma 11 neuronów przyjmujących input treningowy. Tych 11 neuronów połączonych jest z 5 neuronami w warstwie ukrytej, które z kolei łączą się z jednym neuronem wyjściowym. Liczba 5 neuronów jest wynikiem dalszych eksperymentów.
```python
def __init__(self):
    np.random.seed(1)
    self.synaptic_weights1 = 2 * np.random.random((11, 5)) - 1
    self.synaptic_weights2 = 2 * np.random.random((5, 1)) - 1
```
Modyfikacji ulega funkcja treningowa. Dodaję output warstwy ukrytej. Jego deklaracja jest taka sama jak outputu całej funkcji w wersji jednowarstwowej. Tam występuje jednak tylko w funkcji think. Tutaj musi być uzgadniany co iterację z warstwą ukrytą. Należy pamiętać, że output całej funkcji przyjmuje teraz output ukryty, a nie treningowy.
```python
def train(self, training_inputs, training_outputs, training_iterations):
    for iteration in range(training_iterations):
        hidden_output = self.sigmoid(np.dot(training_inputs, self.synaptic_weights1))
        output = self.sigmoid(np.dot(hidden_output, self.synaptic_weights2))
```
Tak jak wcześniej, błąd w warstwie wyjściowej jest obliczany na podstawie różnicy między oczekiwanymi wynikami a przewidywanymi wynikami. Następnie dostosowania/korekty (output_adjustments) są obliczane przy użyciu pochodnej funkcji aktywacji warstwy wyjściowej, tak samo jak w przypadku jednowarstwowej sieci. 

Na podstawie tych dostosowań oblicza się błąd w warstwie ukrytej przy użyciu macierzy wag między warstwą ukrytą a wyjściową. Mnożąc te korekty przez wagi synaptic_weights2, błąd z warstwy wyjściowej przenoszony jest z powrotem do warstwy ukrytej. Ten krok pozwala przypisać część błędu w warstwie wyjściowej do każdego neuronu w warstwie ukrytej.
```python
output_error = training_outputs - output
output_adjustments = output_error * self.sigmoid_derivative(output)
hidden_error = np.dot(output_adjustments, self.synaptic_weights2.T)
hidden_adjustments = hidden_error * self.sigmoid_derivative(hidden_output)
```
Output ukryty deklarowany jest tak samo, jak wyjściowy:
```python
def think(self, inputs):
    inputs = inputs.astype(float)
    hidden_output = self.sigmoid(np.dot(inputs, self.synaptic_weights1))
    output = self.sigmoid(np.dot(hidden_output, self.synaptic_weights2))
    return output
```
Testowanie modelu odbywa się na tej samej zasadzie co poprzednio. Okazuje się jednak, że model traci na dokładności. Po kilku eksperymentach z liczbą neuronów pośrednich, najlepsze wyniki uzyskałem z 5 neuronami. Wyniki te są i tak gorsze niż w jednowarstwowej sieci, wynosząc jedynie 54%.

Najprawdopodobniej, dodatkowa warstwa wprowadza niepotrzebny poziom skomplikowania do prostej zależności, którą obserwuje się w danych. Ponadto, modele wielowarstwowe mogą wymagać większej ilości danych treningowych. Ewentualnie, można spróbować zdefiniować inną, prostszą funkcję aktywacji w warstwie ukrytej.

#### Zmiana funkcji aktywacji

Prostą funkcją aktywacji, którą można zastosować w warstwie ukrytej jest ReLU. Przyporządkowuje ona wartościom dodatnim ich wartość w sposób liniowy, a wartościom ujemnym wartość 0. 

**Wzór funkcji ReLU**

$Relu(z) = max(0, z)$

Do definicji funkcji wykorzystuję metodę maximum z biblioteki numpy.
```python
def relu(self, x):
    return np.maximum(0, x)
```
Pochodna z ReLU to pochodna z X dla liczb dodatnich i pochodna z 0 dla liczb ujemnych. Jej wartości to więc odpowiednio 1 i 0.
```python
def relu_derivative(self, x):
    return np.where(x > 0, 1, 0)
```

Jedyną zmianą w stosunku do sieci dwuwarstwowej jest konieczność podmienienia funkcji sigmoid na ReLU we wszystkich miejscach, w których figurowała jako warstwa ukryta. Przykład poniżej:
```python
hidden_output = self.sigmoid(np.dot(inputs, self.synaptic_weights1))
```
Zamieniam na:
```python
hidden_output = self.relu(np.dot(training_inputs, self.synaptic_weights1))
```
Eksperymentując z liczbą neuronów pośrednich, przy 7 neuronach model osiągnął maksymalny wynik 64% dokładności. Lepiej, niż w przypadku dwóch warstw z aktywacją sigmoidalną, ale wciąż gorzej niż w przypadku jednowarstwowej sieci. Jak widać prostsze, nie znaczy gorsze.


## Wprowadzenie do Keras

Keras to wysokopoziomowa biblioteka do tworzenia i trenowania sieci neuronowych. Została zaprojektowana w celu zapewnienia prostego i intuicyjnego interfejsu programistycznego dla budowy modeli sieci neuronowych. 

Pierwszym ćwiczeniem będzie rozpoznawanie cyfr pisanych.

```python
import keras 
from keras.datasets import mnist 
import matplotlib.pyplot as plt 
from keras.utils import to_categorical
from keras.models import Sequential 
from keras.layers.core import Dense, Activation
import numpy as np
```

Następnie, ładowane są dane MNIST[^1] i dzielone na zbiory treningowy i testowy. Zbiór treningowy zawiera 60 000 obrazów cyfr, a zbiór testowy - 10 000 obrazów.

[^1]: MNIST to baza danych autorstwa Y. Le Cunn et al. zawierająca cyfry pisane odręcznie. Wchodzi w skład pakietu Keras.

```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

Poniższe linie to standaryzacja danych, czyli normalizacja wartości pikseli do zakresu od 0 do 1. Celem normalizacji jest ułatwienie procesu uczenia modelu. Pozwala ona na szybsze zbieganie algorytmów optymalizacyjnych do minimum globalnego. Ponadto, w niektórych zbiorach danych, normalizacja jest wymagana celem zmniejszenia wpływu wartości odstających oraz umożliwienia porównywania danych różniących się skalą/jednostką/zakresem. 

```python
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32') 
x_train /= 255
x_test /= 255
```

Aby móc przetwarzać dane z obrazów 28x28 pikseli przez sieć typu DNN, należy przekształcić ją na jednowymiarową macierz o długości 784 pikseli.

```python
x_train = x_train.reshape(60000, 784) 
x_test = x_test.reshape(10000, 784)
```

Kolejnym krokiem jest zakodowanie wartości cyfr na wektory binarne za pomocą funkcji `to_categorical` z biblioteki Keras. Dzięki temu możliwe jest przetwarzanie wyników przez sieć neuronową.

```python
y_train = to_categorical(y_train, num_classes=10) 
y_test = to_categorical(y_test, num_classes=10)
```

W końcu przychodzi pora na stworzenie modelu: 

```python
model = Sequential() 
model.add(Dense(10, activation='sigmoid', input_shape=(784,))) # pierwsza warstwa (wejściowa)
model.add(Dense(10, activation='softmax')) # druga warstwa
```

Tworzony jest model sieci neuronowej, składający się z dwóch warstw. Warstwy te są w pełni połączonymi warstwami neuronów, gdzie każdy neuron w warstwie jest połączony ze wszystkimi neuronami w poprzedniej warstwie. Pierwsza warstwa ma 10 neuronów i używa funkcji aktywacji sigmoid - druga warstwa również ma 10 neuronów i używa funkcji aktywacji softmax. Pierwsza warstwa jest warstwą wejściową, której rozmiar jest równy liczbie pikseli obrazów cyfr. 

Funkcja sigmoid przetwarza dane wejściowe (tj. piksele obrazów cyfr) na wartości z zakresu 0-1, które interpretujemy jako prawdopodobieństwo, że dany neuron reprezentuje daną cyfrę. Funkcja softmax natomiast normalizuje wyniki funkcji sigmoid w taki sposób, że suma wszystkich wartości wynosi 1. Dzięki temu wyniki tej funkcji możemy interpretować jako rozkład prawdopodobieństwa dla wszystkich klas.

**Wzór funkcji softmax**

$\sigma(x_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} \quad \text{for } i=1,2,\dots,K$

gdzie $x$ to wektor wejściowy, a $K$ to liczba klas. Funkcja softmax normalizuje wartości wejściowe tak, że ich suma wynosi 1, co pozwala na interpretację wyników jako prawdopodobieństw przynależności do poszczególnych klas.

Alternatywnie, zamiast funkcji sigmoidalnej można użyć funkcji ReLU, która lepiej sprawdza się przy większej liczbie warstw.

Ostatnim krokiem jest kompilacja i tzw. *fitowanie* modelu:

```python
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size=100, epochs=5)
```

Jako funkcja straty ustawiona została `categorical_crossentropy`(entropia krzyżowa), optymalizator jako `sgd` (stochastic gradient descent) i metrykę jako `accuracy`.

**Wzór funkcji entropii krzyżowej**

$L(y,ŷ)=−(y⋅log(ŷ)+(1−y)⋅log(1−ŷ))$

*Gdzie:*

- *y to prawdziwa etykieta (0 lub 1)*
- *ŷ to przewidziana etykieta przez model (wartość z zakresu 0 do 1)*

## Przetwarzanie tekstu (Word Embeddings)

### Przygotowanie danych

```python
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()

    return text

def clean_doc(doc):
    tokens = doc.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]

    return tokens
```

Funkcja `load_doc` służy do wczytania zawartości pliku tekstowego o podanej nazwie. Funkcja `clean_doc` otrzymuje tekst, który dzieli na pojedyncze słowa (tokeny). Następnie usuwa znaki interpunkcyjne, słowa niebędące literami, *stopwords* oraz słowa równe bądź krótsze niż jeden znak. Zwraca listę przetworzonych tokenów.

Następnym krokiem będzie **zdefiniowanie słownika** słów używanych w dokumentach, których używamy do analizy. Do tego zadania importujemy dwie dodatkowe biblioteki:

```
from keras.preprocessing.text import Tokenizer
```

Moduł `keras.preprocessing.text.Tokenizer` służy do przetwarzania tekstu na wektory numeryczne, które mogą być wykorzystane do modelowania tekstu.

```python
def add_doc_to_vocab(filename, vocab):
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    vocab.update(tokens)

def process_docs(directory, vocab):
    for filename in listdir(directory):
        if filename.startswith('11'):
            continue

    path = directory + '/' + filename
    add_doc_to_vocab(path, vocab)

def save_list(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

vocab = Counter()

process_docs('/content/drive/MyDrive/Colab Notebooks/pos', vocab)
process_docs('/content/drive/MyDrive/Colab Notebooks/neg', vocab)

print(len(vocab))

min_occurance = 2
tokens = [k for k,c in vocab.items() if c >= min_occurance]
print(len(tokens))

save_list(tokens, 'vocab.txt')
```

    Funkcje te są odpowiedzialne za przygotowanie słownika wykorzystanego w modelu. Funkcja `add_doc_to_vocab` wczytuje dokument i wywołuje, opisaną wcześniej, funkcję `clean_doc`. Funkcja `process_docs` przetwarza wszystkie dokumenty w katalogu, wykorzystując funkcję `add_doc_to_vocab` do dodania słów do słownika. Natomiast funkcja `save_list` zapisuje listę słów do pliku o podanej nazwie. Na końcu kod wyświetla długość słownika oraz liczbę tokenów, które występują co najmniej dwukrotnie (są to słowa, które trafią do słownika).

Tym sposobem został zdefiniowany słownik o nazwie ‘**vocab.txt**’.

Teraz przystępujemy do przygotowania reprezentacji tekstów według modelu Bag-of-Words. Będziemy musieli wykonać dwa kroki:

- utworzenie linii tokenów z naszych zasobów
- zakodowanie tekstów modelem reprezentacji Bag-of-Words

Na początku zaczniemy od funkcji, która utworzy nam linie tokenów:

```python
def doc_to_line(filename, vocab):
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)
```

Funkcja o nazwie `doc_to_line` przyjmuje dwa argumenty - nazwę pliku oraz zbiór słów, których słownictwo ma być użyte do filtrowania tokenów. Funkcja najpierw wczytuje dokument z pliku, a następnie usuwa z niego znaki specjalne i dzieli tekst na pojedyncze tokeny. Następnie filtruje listę tokenów, aby pozostawić tylko te, które znajdują się w zbiorze słów (przekazanym jako drugi argument). Na koniec łączy listę tokenów w jeden ciąg znaków i zwraca ten ciąg.

Następnym krokiem jest już zakodowanie tekstów:

```python
def process_docs(directory, vocab, is_train):
    lines = list()
    for filename in listdir(directory):
        if is_train and filename.startswith('11'):
            continue
        if not is_train and not filename.startswith('11'):
            continue
        path = directory + '/' + filename
        line = doc_to_line(path, vocab)
        lines.append(line)
    return lines

def load_clean_dataset(vocab, is_train):
    neg = process_docs('/content/drive/MyDrive/Colab Notebooks/neg', vocab, is_train)
    pos = process_docs('/content/drive/MyDrive/Colab Notebooks/pos', vocab, is_train)
    docs = neg + pos
    labels = [0 for _ in range(len(neg))] + [1 for _ in range(len(pos))]
    return docs, labels

def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())
train_docs, ytrain = load_clean_dataset(vocab, True)
test_docs, ytest = load_clean_dataset(vocab, False)
tokenizer = create_tokenizer(train_docs)
Xtrain = tokenizer.texts_to_matrix(train_docs, mode='freq')
Xtest = tokenizer.texts_to_matrix(test_docs, mode='freq')
```

Funkcje, które zostają tutaj użyte, oprócz wcześniej wspomnianej `process_docs`, to `load_clean_dataset`, która ładuje zbiór danych z dwóch folderów: "neg" i "pos" i dokonuje ich przetworzenia, używając funkcji `process_docs`, a następnie łączy je w jedną listę. Potem funkcja przygotowuje etykiety dla dokumentów - "0" dla negatywnych dokumentów i "1" dla pozytywnych dokumentów - i zwraca listę dokumentów i odpowiadających im etykiet. Następnie funkcja `create_tokenizer` tworzy obiekt Tokenizer, który służy do przetwarzania tekstu na sekwencje liczb.

### Model

Oczyściwszy dokumenty, rozpoczynam budowę modelu.

**Budowa i trenowanie modelu**

Importujemy moduł `keras`, a wraz z nim typ sekwencyjny `sequential` oraz warstwy typu `dense`, `flatten`, `embedding` i `convolutional` (konwolucyjne):

```
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
```

Zaczytujemy dane treningowe i dokonujemy ich wektoryzacji.

```python
def load_clean_dataset(vocab, is_train):
    neg = process_docs(r'C:\Users\mslus\Desktop\Projects\rafinacja-zaliczenie\neg', vocab, is_train)
    pos = process_docs(r'C:\Users\mslus\Desktop\Projects\rafinacja-zaliczenie\pos', vocab, is_train)
    docs = neg + pos
    labels = array([0 for _ in range(len(neg))] + [1 for _ in range(len(pos))])
    return docs, labels

def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer
```

Kluczem do stworzenia modelu Word Embeddings jest wyrównanie długości sekwencji wejścia - *padding*. Modele Word Embeddings wymagają stałej długości sekwencji wejściowych. Dzięki *paddingowi*, można dostosować zróżnicowane długości zdaniowych sekwencji do oczekiwanej długości, co pozwala na bezproblemowe wykorzystanie tych modeli. Niestety, podczas importu modułu do paddingu z Keras wystąpił nieoczekiwany błąd zgodności wersji. Na szczęście, łatwo można stworzyć taką funkcję od podstaw - `pad_sequences`. Wywołujemy ją w funkcji `encode_docs`.

```python
def pad_sequences(sequences, maxlen=None, padding_value=0):
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)

    padded_sequences = []
    for seq in sequences:
        padded_seq = seq[:maxlen] + [padding_value] * max(0, maxlen - len(seq))
        padded_sequences.append(padded_seq)
    return padded_sequences

def encode_docs(tokenizer, max_length, docs):
    encoded = tokenizer.texts_to_sequences(docs)
    padded = pad_sequences(encoded, maxlen=max_length)
    return padded
```

Podczas treningu sieci neuronowej z wykorzystaniem warstwy `embedding`, wagi tej warstwy są aktualizowane, aby nauczyć się odpowiednich reprezentacji słów w kontekście konkretnego zadania. W trakcie propagacji wstecznej, modele aktualizują wektory osadzeń w taki sposób, aby słowa o podobnym znaczeniu lub kontekście były reprezentowane przez podobne wektory.

**Definiowanie modelu**

Zaczynamy od stworzenia modelu sekwencyjnego za pomocą klasy `Sequential()`. Następnie dodawana jest warstwa `Embedding`, która tworzy wektory osadzenia dla słów.`vocab_size` oznacza rozmiar słownika, czyli liczbę unikalnych słów w zbiorze danych. Argument `100` określa rozmiar wektora osadzenia dla każdego słowa. `input_length` wskazuje na maksymalną długość sekwencji wejściowej.

```
def define_model(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model
```

W trzecim kroku dodajemy warstwę konwolucyjną `Conv1D`. Warstwa konwolucyjna w modelu Word Embeddings pozwala na wykrycie lokalnych wzorców w tekście, takich jak sekwencje słów o określonym znaczeniu lub kontekście. Filtry konwolucyjne działają jako detektory cech, które uczą się reprezentować różne aspekty tekstu, na przykład słowa kluczowe, frazy lub gramatyczne konstrukcje. Te cechy mogą następnie być wykorzystane przez kolejne warstwy modelu do wykonania konkretnych zadań związanych z przetwarzaniem języka naturalnego.

Dodawana jest warstwa `MaxPooling1D`, której zadaniem jest warstwa redukcja wymiarowości macierzy cech. Przyczynia się ona do bardziej efektywnego i skutecznego przetwarzania tekstu, poprawiając reprezentację cech, redukując wymiarowość i zwiększając odporność na zmienność.

Przed wejściem do warstwy `Dense`, dane z poprzednich warstw muszą zostać spłaszczone do jednego wymiaru. Służy do tego warstwa `Flatten`. Warstwę wyjściową stanowi Dense, aktywowane funkcją *sigmoid*, która zwraca wartości z zakresu od 0 do 1, interpretowane jako prawdopodobieństwo przynależności do klasy. Kompilacja modelu odbywa się na parametrach analogicznych, do tych, które zastosowaliśmy w modelu Bag-of-Words. Jako funkcja straty wywoływana jest `binary_crossentropy`. Optymalizator `adam`, a miarą ewaluacji modelu jest `accuracy`.

Na podstawie skompilowanego modelu rozpoczyna się trening, polegający na wywołaniu uprzednio zdefiniowanych funkcji.

Na szybko można sprawdzić model na przykładowych zdaniach testowych:

```python
text = 'This is the best movie I have ever seen'
percent, sentiment = predict_sentiment(text, vocab, tokenizer, max_length, model)
print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text, sentiment, percent*100))
# test negative text
text = 'This is the worst movie ever.'
percent, sentiment = predict_sentiment(text, vocab, tokenizer, max_length, model)
print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text, sentiment, percent*100))
```


 

___
