import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from  keras.models import load_model

model=load_model("zadatak1_model.keras")

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

#indeksiranje krivih
y_pred=model.predict(x_test_s)
pred_labels=np.argmax(y_pred,axis=1)
print(pred_labels)
print(y_test)
wrong_labels=np.where(pred_labels!=y_test)[0]
print(wrong_labels)

for i in range(3):
    plt.figure()
    index=wrong_labels[i]
    plt.imshow(x_test[index].reshape(28,28),cmap='gray')
    plt.title(f"Real label: {y_test[index]}, Predicted label: {pred_labels[index]}")
plt.show()