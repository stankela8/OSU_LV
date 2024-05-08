import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import Accuracy
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.cluster import KMeans
#učitavanje dataseta
iris = datasets.load_iris()


##################################################
#1. zadatak
##################################################
#a)
data = iris.data
names = iris.target
classes = iris.target_names

virginica = data[names == 2]
versicolour = data[names == 1]
setosa = data[names == 0]

virginica_sepal_lenght = virginica[:,0]
virginica_petal_lenght = virginica[:,2]
setosa_sepal_lenght = setosa[:,0]
setosa_petal_lenght = setosa[:,2]

plt.scatter(virginica_petal_lenght, virginica_sepal_lenght, color = 'green', label = 'Virginica')
plt.scatter(setosa_petal_lenght, setosa_sepal_lenght, color = 'grey', label = 'setosa')
plt.xlabel('Duljina latica')
plt.ylabel('Duljina čašice')
plt.title('Naslov')
plt.legend()
plt.show()


#b)

max_sepal_width = [max(data[names == i][:, 1]) for i in range(3)]

plt.bar(classes, max_sepal_width)
plt.xlabel('Klase')
plt.ylabel('Najveća širina čašice')
plt.title('Najveća širina čašice za svaku klasu cvijeta')
plt.show()

#c)

avgSetosa=setosa[:,1].mean()
greaterThanAvgSetosa=sum(setosa[:,1]>avgSetosa)
print(greaterThanAvgSetosa)
##################################################
#2. zadatak
##################################################

iris = datasets.load_iris()
X = iris.data
y = iris.target 
#a)

wcss = []
for i in range(1, 11):  # Testiranje za K od 1 do 10
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
optimal_k = np.argmin(np.diff(wcss)) + 2
print('Optimalna velicina K dobivena lakat metodom: ', optimal_k)

# Crtanje lakat grafikona
plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Lakat metoda za Iris dataset')
plt.xlabel('Broj klastera (K)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()
#b)

#c)
km = KMeans(n_clusters=optimal_k, init='random', n_init=5, random_state=0)
km.fit(X)
labels=km.predict(X)
#d)
colormap=np.array(['green','yellow','orange'])
plt.scatter(iris.data[:, 0], iris.data[:, 1], c=colormap[iris.target])
plt.title("Grupiranje podataka pomoću algoritma K-means")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.show()
#e)

##################################################
#3. zadatak
##################################################
iris = datasets.load_iris()
X = iris.data  # podaci o latama i čašicama
y = iris.target  # oznake klasa

# Podjela podataka na skup za učenje i skup za testiranje u omjeru 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train_categorical = np.eye(3)[y_train]
y_test_categorical = np.eye(3)[y_test]

#a)
model = Sequential([
    Dense(12, activation='relu', input_shape=(4,)),
    Dropout(0.2),
    Dense(7, activation='relu'),
    Dropout(0.3),
    Dense(5, activation='relu'),
    Dense(3, activation='softmax')
])
model.summary()
#b)

#c)

#d)

#e)

#f)