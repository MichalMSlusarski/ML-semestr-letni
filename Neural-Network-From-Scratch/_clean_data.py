import pandas as pd
import numpy as np

df = pd.read_csv('basic.csv')

# podstawowe czyszczenie: usunięcie rzędów tekstowych, kolumny Z13, uzupełnienie braków zerami oraz profilaktycznie ustalenie typu
df = df.drop([0, 1])
df = df.reset_index(drop=True)
df = df.drop('Z13', axis=1)
df = df.fillna(0)
df = df.apply(pd.to_numeric)

# wybieram kolumny do standardyzacji
columns_to_standardize = ['Z11', 'Z14', 'Z15', 'Z16', 'Z17', 'Z18', 'Z19', 'Z20']

# do standardyzacji używam metody max-min. Wynik zaokrąglam
df[columns_to_standardize] = round((df[columns_to_standardize] - df[columns_to_standardize].min()) / (df[columns_to_standardize].max() - df[columns_to_standardize].min()), 6)

# zamieniam zmienną Z21 z numerycznej na kategoryczną, za pułap przyjmując medianę wartości liczby komentarzy. Dzięki temu mozna podzielic zbiór na dwie mniej więcej równe klasy
median = df['Z21'].median()
df['Z21'] = np.where(df['Z21'] > median, 1, 0)
print(median)

# sprawdzam czy kategorie są w przybliżeniu równe
#print(df['Z21'].value_counts())

# 50 ostatnich pozycji ze zbioru przeznaczam na zbiór testowy
test_df = df[-50:]
df = df[:-50]

# zapisuję zbiory
df.to_csv('training.csv')
test_df.to_csv('test.csv')
