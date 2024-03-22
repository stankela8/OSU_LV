import pandas as pd
data = pd.read_csv("data_C02_emission.csv")

#1.a
print("Broj mjerenja: ",len(data))
print(data.info())
print ( data . isnull (). sum ())
data.dropna(axis =0)
data.dropna(axis =1)
data.drop_duplicates()
data = data.reset_index(drop = True)

print(data.head(5))
print(data.describe())

#1.b
potrosnja=data.sort_values("Fuel Consumption City (L/100km)")
print("Najmanja potrosnja: ",potrosnja[["Make","Model","Fuel Consumption City (L/100km)"]].head(3))
print("Najveca potrosnja: ",potrosnja[["Make","Model","Fuel Consumption City (L/100km)"]].tail(3))

#1.c
velicinaMotora=(data[(data["Engine Size (L)"] >= 2.5) & (data ["Engine Size (L)"] <= 3.5)])
print(len(velicinaMotora))
zbrojEmisije=(velicinaMotora[(velicinaMotora["CO2 Emissions (g/km)"])].sum())
print(zbrojEmisije)

#1.d
audi=data[(data["Make"] == "Audi")]
print(len(audi))

