import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(train_images.shape)

#one pixel
print(train_images[0, 23, 23])

print(train_labels[:10])

class_names = ['T-shirt/top', 'Trousers', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneakers', 'Bag', 'Ankle boot']

# plt.figure()
# plt.imshow(train_images[2])
# plt.colorbar()
# plt.grid(False)
# plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape= (28, 28)), #input layer (1)
    keras.layers.Dense(128, activation = 'relu' ), #hidden layer (2)
    keras.layers.Dense(10, activation = 'softmax') #output layer (3)
])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(train_images, train_labels, epochs = 10)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 1)

print('Test accuracy:', test_acc)

predictions = model.predict(test_images)
print(np.argmax(predictions[0]))





