import numpy as np #self-explainatory
import matplotlib.pyplot as plt #helps us visualize stuff
import tensorflow as tf #neural network tool we're using
import keras
import random

(train_img, train_lbl), (test_img, test_lbl) = keras.datasets.mnist.load_data()

model=tf.keras.models.load_model('digit_recognizer.keras')

predictions = model.predict(test_img)

rnum=random.randint(0,9999)
plt.imshow(test_img[rnum], cmap=plt.cm.binary)
plt.title(f"Predicted: {np.argmax(predictions[rnum])}")
plt.show()