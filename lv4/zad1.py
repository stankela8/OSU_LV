from sklearn . model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sklearn . linear_model as lm
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


dataframe = pd.read_csv('lv4/data_C02_emission.csv')


#a) 
# rastavljanje dataframe-a na ulazne i izlazne velicine
X = dataframe[['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)', 'Fuel Consumption Comb (mpg)']]
y = dataframe['CO2 Emissions (g/km)']
# podjela skupa na skup za ucenje i testiranje
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#b)
plt.figure()
# iscrtavanje izlaza u ovisnosti o jednoj ulaznoj velicini
#iscrtavanje test i train podataka
plt.scatter(y_train, X_train['Fuel Consumption City (L/100km)'], color="blue",  s=10, alpha= 0.5)
plt.scatter(y_test, X_test['Fuel Consumption City (L/100km)'], color="red",  s=10, alpha= 0.5)
plt.show()

#c)
sc = MinMaxScaler()
X_train_n = sc.fit_transform(X_train) # parametri
scaled_X_train = pd.DataFrame(X_train_n, columns=X_train.columns) # transformacija
scaled_X_test = pd.DataFrame(X_train_n, columns=X_train.columns) # transformacija s parametrima X_train skaliranja

# prikaz podataka prije i poslije skaliranja
X_train['Fuel Consumption City (L/100km)'].plot(kind='hist', bins=25)
plt.show()
scaled_X_train['Fuel Consumption City (L/100km)'].plot(kind='hist', bins=25)
plt.show()

#d)
linearModel = lm.LinearRegression ()
linearModel.fit( X_train_n , y_train )
print(linearModel.coef_)
print(linearModel.intercept_)

#e)
y_test_expect = linearModel.predict(sc.transform(X_test))

# prikaz stvarnioh izlaznih podataka i predvidjenih izlaznih podataka
plt.figure(figsize=(8, 6))
plt.scatter(y_test, X_test['Fuel Consumption City (L/100km)'], label='Real data', alpha=0.5, s=10)
plt.scatter(y_test_expect, X_test['Fuel Consumption City (L/100km)'], label='Predicted data', alpha=0.5, s=10)
plt.legend()
plt.show()

#g)
MSE = mean_squared_error(y_test, y_test_expect)
print(f"Mean squared error(MSE): {MSE}")
MAE = mean_absolute_error(y_test, y_test_expect)
print(f"Mean absolute error(MAE): {MAE}")
MAPE = mean_absolute_percentage_error(y_test, y_test_expect)
print(f"Mean absolute percentage error(MAPE): {MAPE}")
