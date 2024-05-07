from sklearn.preprocessing import OneHotEncoder
import sklearn.linear_model as lm
import pandas as pd

# ucitavanje podataka
dataframe = pd.read_csv('LV4/data_C02_emission.csv')

# 1-od-K kodiranje
ohe = OneHotEncoder()
X_encoded = ohe.fit_transform(dataframe[['Fuel Type']]).toarray()
print(X_encoded)
print("---------------------------------------------------")

X = dataframe[['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)', 'Fuel Consumption Comb (mpg)']]
X = pd.concat([pd.DataFrame(X_encoded, columns=ohe.get_feature_names_out(['Fuel Type'])), X], axis=1)
y = dataframe['CO2 Emissions (g/km)']

# linearna regresija
linearModel = lm.LinearRegression()
linearModel.fit(X, y)
y_expect = linearModel.predict(X)

# najveca pogreska u procjeni
max_error = max(abs(y - y_expect))
print(max_error)

# index najvece greske --> index neke vrijednosti u nizu
max_error_index = abs(y - y_expect).argmax()

# trazenje podatka koji ima najvecu procjenjenu gresku
data_point_X = X.iloc[max_error_index]
data_point_y = y.iloc[max_error_index]
data_point = dataframe.iloc[max_error_index]
print("X: ", data_point_X)
print("y: ", data_point_y)
print(data_point)
