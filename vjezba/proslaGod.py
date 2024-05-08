import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, recall_score, precision_score
from tensorflow import keras
from keras import layers

# ucitavanje csv datoteke
dataframe = pd.read_csv('./diabetes.csv')
data = dataframe

#1-a
print("---------------------------------")
print("Broj osoba: " + str(len(data)))

#1-b
print("---------------------------------")
print(data.isnull().sum())
print(data.duplicated().sum())
data = data.drop_duplicates()
data = data.dropna()
data = data[data['BMI'] != 0] # removes rows with BMI = 0
data = data.reset_index(drop = True)
print("Broj osoba nakon brisanja duplikata: " + str(len(data)))

#1-c
print("---------------------------------")
plt.figure()
plt.scatter(data['Age'], data['BMI'], color="blue",  s=5)
plt.xlabel('Age')
plt.ylabel('BMI')
plt.title("AGE VS BMI")
plt.show()

#1-d
print("---------------------------------")
print("Max BMI: " + str(data['BMI'].max()))
print("Min BMI: " + str(data['BMI'].min()))
print("Mean BMI: " + str(data['BMI'].mean()))

#1-e
print("---------------------------------")
has_diabetes = data[data['Outcome'] == 1]
no_diabetes = data[data['Outcome'] == 0]
print("has diabetes data:")
print("Broj osoba: " + str(len(has_diabetes)))
print("Max BMI: " + str(has_diabetes['BMI'].max()))
print("Min BMI: " + str(has_diabetes['BMI'].min()))
print("Mean BMI: " + str(has_diabetes['BMI'].mean()))
print()
print("no diabetes data:")
print("Broj osoba: " + str(len(no_diabetes)))
print("Max BMI: " + str(no_diabetes['BMI'].max()))
print("Min BMI: " + str(no_diabetes['BMI'].min()))
print("Mean BMI: " + str(no_diabetes['BMI'].mean()))

#2
print("---------------------------------")
print("---------------------------------")
print("---------------------------------")
X = dataframe[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = dataframe['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#2-a
print("---------------------------------")
log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)

#2-b
print("---------------------------------")
y_test_pred = log_reg.predict(X_test)

#2-c
print("---------------------------------")
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()

#2-d
print("---------------------------------")
print("tocnost: " + str(accuracy_score(y_test, y_test_pred)))
print("odziv: " + str(recall_score(y_test, y_test_pred)))
print("preciznost: " + str(precision_score(y_test, y_test_pred)))

#3
print("---------------------------------")
print("---------------------------------")
print("---------------------------------")
X = dataframe[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = dataframe['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#3-a
print("---------------------------------")
model = keras.Sequential()
model.add(layers.Input(shape=(8,)))  # 8 ulaznih varijabli
model.add(layers.Dense(12, activation='relu'))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary())

#3-b
print("---------------------------------")
model.compile(optimizer='adam',
                    loss='binary_crossentropy', # categorical_crossentropy
                    metrics=['accuracy'])

#3-c
print("---------------------------------")
model.fit(X_train,
                y_train,
                epochs = 150,
                batch_size = 10,
                validation_split = 0.1)

#3-d
model.save('model.h5')
# model = load_model('model.h5')

#3-e
score = model.evaluate(X_test, y_test)
print("Evaluacija na testnom skupu: " + str(score))

#3-f
print("---------------------------------")
y_test_pred = model.predict(X_test)
# potrebno jer se radi 0 kategorickim a ne regresijskim vrijednostima
y_test_pred[y_test_pred > 0.5] = 1
y_test_pred[y_test_pred <= 0.5] = 0

cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()