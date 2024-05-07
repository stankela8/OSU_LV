import numpy as np
from tensorflow import keras
from PIL import Image

# Učitavanje izgrađene mreže
model = keras.models.load_model("lv8/zadatak1_model.keras")

# Učitavanje slike za testiranje
img_path = "slika.png"
img = Image.open(img_path).convert('L')  # Otvori sliku i pretvori je u crno-bijelu
img = img.resize((28, 28))  # Promijeni veličinu slike na 28x28 piksela
img_array = np.array(img)  # Pretvori sliku u numpy array

img_array = img_array.astype("float32") / 255.0  # Skaliraj vrijednosti piksela na raspon [0, 1]
img_array=np.expand_dims(img_array,axis=0)
img_array=np.expand_dims(img_array,axis=-1)


# Klasificiranje slike
predictions = model.predict(img_array)
predicted_label = np.argmax(predictions)

# Ispis rezultata u terminal
print("Predviđena klasa:", predicted_label)