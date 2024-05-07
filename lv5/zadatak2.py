#Zadatak 5.5.2 Skripta zadatak_2.py ucitava podatkovni skup Palmer Penguins[1]. Ovaj podatkovni skup sadrži mjerenja provedena na tri razlicite vrste pingvina (’Adelie’, ’Chinstrap’,
#’Gentoo’) na tri razlicita otoka u podrucju Palmer Station, Antarktika. Vrsta pingvina odabrana je kao izlazna velicina i pri tome su klase oznacene s cjelobrojnim vrijednostima
#0, 1 i 2. Ulazne velicine su duljina kljuna (’bill_length_mm’) i duljina peraje u mm (’flipper_length_mm’). Za vizualizaciju podatkovnih primjera i granice odluke u skripti je dostupna
#funkcija plot_decision_region.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    edgecolor = 'w',
                    label=labels[cl])

# ucitaj podatke
df = pd.read_csv("lv5/penguins.csv")

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}, inplace = True)

print(df.info())

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()

# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

#a) Pomocu stupcastog dijagrama prikažite koliko primjera postoji za svaku klasu (vrstu pingvina) u skupu podataka 
#za ucenje i skupu podataka za testiranje. Koristite numpy funkciju unique.

plt.bar(np.unique(y_train), np.bincount(y_train[:,0]))
plt.bar(np.unique(y_test), np.bincount(y_test[:,0]))
plt.show()

#b) Izgradite model logisticke regresije pomocu scikit-learn biblioteke na temelju skupa podataka za ucenje.

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

#c) Pronadite u atributima izgradenog modela parametre modela. Koja je razlika u odnosu na binarni klasifikacijski problem iz prvog zadatka?

print(model.coef_)
print(model.intercept_)

#Glavna razlika u odnosu na binarni klasifikacijski problem iz prvog zadatka je broj parametara, u prvom zadatku postojala su samo 3 parametra, a u ovom 9 tj 3*3 zato sto postoje3 razlicite klase.

#d) Pozovite funkciju plot_decision_region pri cemu joj predajte podatke za ucenje i izgradeni model logisticke regresije. Kako komentirate dobivene rezultate?

plot_decision_regions(X_train, y_train[:,0], model, resolution=0.01)
plt.show()

#e) Provedite klasifikaciju skupa podataka za testiranje pomocu izgradenog modela logistike regresije. Izracunajte i prikažite matricu zabune na testnim podacima.
#Izracunajte tocnost. Pomocu classification_report funkcije izracunajte vrijednost cetiri glavne metrike na skupu podataka za testiranje.

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

y_test_p = model.predict(X_test)

cm = confusion_matrix (y_test,y_test_p)
print("Matrica zabune: ",cm)
disp = ConfusionMatrixDisplay(confusion_matrix (y_test,y_test_p))
disp.plot()
plt.show()

print("Tocnost: ", accuracy_score(y_test, y_test_p))
print(classification_report(y_test, y_test_p))

#f) Dodajte u model još ulaznih velicina. Što se dogada s rezultatima klasifikacije na skupu podataka za testiranje?

print ('f:')
input_variables2 = ['bill_length_mm',
                    'flipper_length_mm',
                    'bill_depth_mm',
                    ]
X2 = df[input_variables2].to_numpy()
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size = 0.2, random_state = 123)

model2 = LogisticRegression()
model2.fit(X2_train, y2_train)

y2_test_p = model2.predict(X2_test)

cm = confusion_matrix (y2_test,y2_test_p)
print("Matrica zabune: ",cm)
disp = ConfusionMatrixDisplay(confusion_matrix (y2_test,y2_test_p))
disp.plot()
plt.show()

print("Tocnost: ", accuracy_score(y2_test, y2_test_p))
print(classification_report(y2_test, y2_test_p))

input_variables3 = ['bill_length_mm',
                    'flipper_length_mm',
                    'bill_depth_mm',
                    'body_mass_g',
                    ]

X3 = df[input_variables3].to_numpy()
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y, test_size = 0.2, random_state = 123)

model3 = LogisticRegression()
model3.fit(X3_train, y3_train)

y3_test_p = model3.predict(X3_test)

cm = confusion_matrix (y3_test,y3_test_p)
print("Matrica zabune: ",cm)
disp = ConfusionMatrixDisplay(confusion_matrix (y3_test,y3_test_p))
disp.plot()
plt.show()

print("Tocnost: ", accuracy_score(y3_test, y3_test_p))
print(classification_report(y3_test, y3_test_p))

#Dodavanjem vise ulaznih velicina povecava se tocnost modela sto je vidljivo preko matrica zabune. Dodavanjem trece ulazne velicine povecava se tocnost, ali dodavanjem i cetvrte tocnost klasifikacije prve vrste se smanjuje
#a druge povecava a ukupna tocnost modela ostaje relativno ista.
