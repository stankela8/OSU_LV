import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

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
                    label=cl)


# ucitaj podatke
data = pd.read_csv("lv6/Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()

#1. Izradite algoritam KNN na skupu podataka za ucenje (uz K=5). Izracunajte tocnost klasifikacije na skupu podataka za ucenje i skupu podataka za testiranje. Usporedite
#dobivene rezultate s rezultatima logisticke regresije. Što primjecujete vezano uz dobivenu granicu odluke KNN modela?

KNN = KNeighborsClassifier(5)
KNN.fit(X_train_n,y_train)
y_train_p = KNN.predict(X_train_n)
y_test_p = KNN.predict(X_test_n)

print("KNN(5): ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

plot_decision_regions(X_train_n, y_train, classifier=KNN)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()
#Tocnost klasifikacije je veca s algoritnom KNN od logisticke regresije u ovom slucaju. Granica odluke u ovom algoritmu nije ravna linija nego nekakva cudna krivulja.

#2. Kako izgleda granica odluke kada je K =1 i kada je K = 100?

KNN = KNeighborsClassifier(1)
KNN.fit(X_train_n,y_train)
y_train_p = KNN.predict(X_train_n)
y_test_p = KNN.predict(X_test_n)

print("KNN(1): ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

plot_decision_regions(X_train_n, y_train, classifier=KNN)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()

KNN = KNeighborsClassifier(100)
KNN.fit(X_train_n,y_train)
y_train_p = KNN.predict(X_train_n)
y_test_p = KNN.predict(X_test_n)

print("KNN(100): ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

plot_decision_regions(X_train_n, y_train, classifier=KNN)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()
#Tocnost KNN algoritma s premalim i prevelikim vrijednostima K je dosta gora od logisticke regresije i od KNN algoritma s prikladnim parametrima. Granica odluke kad je K=1 su mala podrucja oko svakog podatka,
#a u slucaju K=100 granica je dosta pomaknuta na crvenu stranu zato sto je vise crvenih podataka u skupu

#6.5.2
k_values = [i for i in range (1,100)]
scores = []
for k in k_values:
    KNN_model = KNeighborsClassifier( n_neighbors = k )
    score = cross_val_score( KNN_model , X_train_n , y_train , cv =5)
    scores.append(np.mean(score))

sns.lineplot(x = k_values , y = scores, marker = 'o')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.show()

print(f"Najbolji k = {k_values[np.argmax(scores)]} , Accuracy = {np.max(scores)}")

#6.5.3 i 4
def plotting(classifier,X_train_n, y_train, y_train_p, title):
    plot_decision_regions(X_train_n, y_train, classifier=classifier )
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.legend(loc='upper left')
    plt.title(title + " Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
    plt.tight_layout()

def make_SVC_model(C, gama):
    SVC_model = svm.SVC(kernel='rbf', C=C, random_state=42, gamma=gama)
    scores = cross_val_score(SVC_model, X_train, y_train, cv=5)
    print("----------------------------------------------\nRBF:")
    print(scores)
    SVC_model.fit(X_train_n, y_train)
    # Evaluacija modela SVC
    y_train_p_SVC = SVC_model.predict(X_train_n)

    plotting(SVC_model,X_train_n, y_train, y_train_p_SVC, "RBF, C="+str(C)+" gamma=" + str(gama))

make_SVC_model(1, 0.01)
make_SVC_model(1, 0.1)
make_SVC_model(100, 0.1)
make_SVC_model(1, 1)
plt.show()

# Pomoću unakrsne validacije odredite optimalnu vrijednost hiperparametra C i γ
# algoritma SVM za problem iz Zadatka 1.
pipe = Pipeline([('scaler', StandardScaler()), ('svm', svm.SVC(kernel='rbf'))])
param_grid = {'svm__C': [0.1, 1, 10], 'svm__gamma': [0.1, 1, 10]}
grid = GridSearchCV(pipe, param_grid, cv=10)
grid.fit(X_train, y_train)
print("Najbolji parametri: ", grid.best_params_)
print("Tocnost: ", grid.best_score_)
print("Tocnost na testnom skupu: ", grid.score(X_test, y_test))
