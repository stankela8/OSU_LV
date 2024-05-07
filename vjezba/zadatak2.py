""""Datoteka titanic.csv sadrži podatke o putnicima broda Titanic, koji je potonuo
1912. godine. Upoznajte se s datasetom. Uˇcitajte dane podatke. Podijelite ih na ulazne podatke X
predstavljene stupcima Pclass, Sex, Fare i Embarked i izlazne podatke y predstavljene stupcem
Survived. Podijelite podatke na skup za uˇcenje i skup za testiranje modela u omjeru 70:30.
Izbacite izostale i null vrijednosti. Skalirajte podatke. Dodajte programski kod u skriptu pomo´cu
kojeg možete odgovoriti na sljede´ca pitanja:"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn . model_selection import train_test_split
from sklearn . preprocessing import MinMaxScaler
from sklearn . neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

data=pd.read_csv("titanic.csv")

#izbacivanje null i izostalih vrijednosti
data.dropna(inplace=True)

#podjela na test i train
X=data[['Pclass', 'Sex', 'Fare','Embarked']]
y=data['Survived']

#pretvaranje kategorickih varijabli u numericke
X=pd.get_dummies(X, columns=["Sex", "Embarked"])

#podjela 70:30
X_train , X_test , y_train , y_test = train_test_split (X, y, test_size = 0.3, random_state =1)

#skaliranje
sc=MinMaxScaler()
X_train_n=sc.fit_transform( X_train )
X_test_n=sc.transform( X_test )

# a)Izradite algoritam KNN na skupu podataka za uˇcenje (uz K=5). Vizualizirajte podatkovne
# primjere i granicu odluke.

# izrada KNN 
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

KNN_model = KNeighborsClassifier( n_neighbors = 5 )
KNN_model.fit( X_train_n , y_train )
y_train_p_KNN = KNN_model.predict(X_train_n)
y_test_p_KNN = KNN_model.predict( X_test_n )

# b) Izraˇcunajte toˇcnost klasifikacije na skupu podataka za uˇcenje i skupu podataka za testiranje.

print("KNN(neighbours = 5): ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_KNN))))

#Sto je k manji model je slozeniji, a sto je k veci model postaje prejednostavan. K=n prejednostavno

#c) Pomo´cu unakrsne validacije odredite optimalnu vrijednost hiperparametra K algoritma
# KNN.

