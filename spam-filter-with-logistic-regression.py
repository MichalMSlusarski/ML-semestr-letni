# Filtrowanie spamu. Zastosowanie regresji logistycznej. 
# Na podstawie: S. Halder, S. Ozdemir: Hands-On Machine Learning for Cybersecurity. Packt Publishing 2018.
# Analizowany korpus - https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
# Plik do wykorzystania smsspam.csv

from google.colab import files
uploaded=files.upload()

!pip install sklearn

!pip install scikit-learn

import pandas as pd
import numpy as np

import sklearn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

spam = pd.read_csv('smsspam.csv', delimiter='\t',header=None)

spam.head()

# Podział zbioru danych na część treningową i testową
X_train_dataset, X_test_dataset, y_train_dataset, y_test_dataset = train_test_split(spam[1],spam[0])

# Konwersja tekstu na wektory cech
vectorizer = TfidfVectorizer()

# Funkcja vectorizer.fit_transform() przekształca dane wejściowe w wektor cech. Wektor cech to skalowane i normalizowane dane używane do treningu modelu.

X_train_dataset = vectorizer.fit_transform(X_train_dataset)

#LogisticRegression() to klasa, która definiuje model regresji logistycznej.
classifier_log = LogisticRegression()

# Funkcja "fit" dopasowuje model klasyfikatora (classifier_log) do zbioru danych treningowych.
classifier_log.fit(X_train_dataset, y_train_dataset)

# Import funkcji obliczającej skuteczność predykcji.
from sklearn.metrics import accuracy_score

#Vectorizer służy do transformacji danych testowych złożonych z dwóch ciągów tekstowych.

X_test_dataset = vectorizer.transform( ['Wah lucky man... Then can save money... Hee....', 'WINNER!! As a valued network customer you have been selected to receivea ĺŁ900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.'] )

# "classifier_log.predict (X_test_dataset)" wykorzystuje model regresji logistycznej do przewidywania wyników dla zestawu danych testowych.
predictions_logistic = classifier_log.predict(X_test_dataset)
print(predictions_logistic)
