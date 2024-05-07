#import bibilioteka
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier 
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
from tensorflow import keras
from keras import layers
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
##################################################
#1. zadatak
##################################################

#učitavanje dataseta
dataset = pd.read_csv('IspitTitanic/titanic.csv')
#a)
brojZena=dataset[dataset['Sex'] == 'female']
print(len(brojZena))

#b)
umrli=dataset[dataset['Survived'] == 0]
umrliPostotak=len(umrli)/(len(dataset))
print(f"Nije prezivejelo {umrliPostotak*100:.3f} % ljudi.")

#c)
males=dataset[dataset['Sex'] == 'male']
malesSurvived=males[males['Survived']==1]
malesSurvivedPercentage=round(len(malesSurvived)/len(males)*100,2)

femalesSurvived=brojZena[brojZena['Survived']==1]
femalesSurvivedPercentage=round(len(femalesSurvived)/len(brojZena)*100,2)

x = ['Male percent', 'Female percent']
y = [malesSurvivedPercentage,femalesSurvivedPercentage]
colors=['green','yellow']
plt.figure()
plt.bar(x,y, color = colors)
plt.xlabel('Spol')
plt.ylabel('Postotak prezivjelih')
plt.title('Dijagram prezivjelih po spolu u postotcima')
plt.show()

#d)
averageAgeOfFemaleSurvivors=femalesSurvived['Age'].mean()
averageAgeOfMaleSurvivors=malesSurvived['Age'].mean()

print(round(averageAgeOfFemaleSurvivors))
print(round(averageAgeOfMaleSurvivors))
#e)
klasa1=malesSurvived[malesSurvived['Pclass']==1]
klasa2=malesSurvived[malesSurvived['Pclass']==2]
klasa3=malesSurvived[malesSurvived['Pclass']==3]

print(klasa1['Age'].max())
print(klasa2['Age'].max())
print(klasa3['Age'].max())
##################################################
#2. zadatak
##################################################

#učitavanje dataseta
data = pd.read_csv('IspitTitanic/titanic.csv')

#train test split
data.dropna(axis = 0)
data.drop_duplicates()
data.reset_index(drop = True)

X = data[['Pclass', 'Sex', 'Fare', 'Embarked']]
X = pd.get_dummies(X, columns = ['Sex', 'Embarked'])
y = data['Survived'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_test_s = sc.transform(X_test)

#a)
KNN_model = KNeighborsClassifier(n_neighbors = 5)
KNN_model.fit(X_train_s, y_train)

y_train_p = KNN_model.predict(X_train_s)
y_test_p = KNN_model.predict(X_test_s)

#b)
print('Tocnost na skupu podataka za ucenje: ', accuracy_score(y_train, y_train_p))
print('Tocnost na skupu podataka za testiranje: ', accuracy_score(y_test, y_test_p))


#c)
k_values = [i for i in range (1,100)]
scores = []
for k in k_values:
    KNN_model = KNeighborsClassifier( n_neighbors = k )
    score = cross_val_score( KNN_model , X_train_s , y_train , cv =5)
    scores.append(np.mean(score))

sns.lineplot(x = k_values , y = scores, marker = 'o')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.show()

print(f"Najbolji k = {k_values[np.argmax(scores)]} , Accuracy = {np.max(scores)}")

#d)
KNN_model_optimalni = KNeighborsClassifier(n_neighbors = 22)
KNN_model_optimalni.fit(X_train_s, y_train)

y_train_p_optimalni = KNN_model_optimalni.predict(X_train_s)
y_test_p_optimalni = KNN_model_optimalni.predict(X_test_s)
print('Tocnost na skupu podataka za ucenje: ', accuracy_score(y_train, y_train_p_optimalni))
print('Tocnost na skupu podataka za testiranje: ', accuracy_score(y_test, y_test_p_optimalni))
##################################################
#3. zadatak
##################################################

#učitavanje podataka:
data = pd.read_csv('IspitTitanic/titanic.csv')

data.dropna(axis = 0)
data.drop_duplicates()
data.reset_index(drop = True)

X = data[['Pclass', 'Sex', 'Fare', 'Embarked']]
X = pd.get_dummies(X, columns = ['Sex', 'Embarked'])
y = data['Survived'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_test_s = sc.transform(X_test)
#a)
model=keras.Sequential()
model.add(layers.Input(shape = (7, )))
model.add(layers.Dense(12, activation = 'relu'))
model.add(layers.Dense(8, activation = 'relu'))
model.add(layers.Dense(4, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))
model.summary()
#b)
model.compile(loss="binary_crossentropy",
optimizer ="adam",
metrics = ["accuracy",])
#c)
epochs = 100
batch_size = 5
history = model.fit(X_train_s, y_train, batch_size = batch_size, epochs = epochs, validation_split = 0.1)
predictions = model.predict(X_test)
y_test_predicted = np.argmax(predictions, axis = 1)
#d)
model.save('model.keras')
#e)
model = load_model("model.keras")

test_loss, test_acc = model.evaluate(X_test_s, y_test, verbose=0)
print(test_acc)
print(test_loss)

y_test_p = model.predict(X_test_s)
y_test_p = (y_test_p > 0.5).astype(int)
cm = confusion_matrix(y_test, y_test_p)
disp = ConfusionMatrixDisplay ( cm )
disp.plot()
plt.show()

print(classification_report(y_test, y_test_p))
