import numpy as np #self-explainatory
import matplotlib.pyplot as plt #helps us visualize stuff
import tensorflow as tf #neural network tool we're using
import keras
from keras import layers, models
import os #tells us stuff abt the directory and whatnot
import random

(train_img, train_lbl), (test_img, test_lbl) = keras.datasets.mnist.load_data()
#splits the data into two sets: testing and training. keras is doing all of this

print(len(test_img), len(test_lbl))

print(len(test_img), len(test_lbl))

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    #2d->1d array
    layers.Dense(64, activation='sigmoid'),
    #64 neurons, decided to go with a sigmoid function bc why not
    layers.Dense(64, activation='sigmoid'),
    #i decided to add two layers bc i can mess with it better later on
    layers.Dense(10, activation='softmax') #this is last layer there are ten numbers 0-9
    #softmax gives the likelihood for any neuron 1-10 actually being the number 1-10. tells us probability basically of which digit it thinks it is
])

print(len(test_img), len(test_lbl))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#adam updates the weight of the models. idk what this means so it's in my comments for future learning
#sparse_categorical_crossentropy calculates how far off the model's predictions are
#metrics tells us how accurate the nn is

print(len(test_img), len(test_lbl))

model.fit(train_img, train_lbl, epochs=5)
#puts the training stuff into our model. also it trains 3 times bc why not

print(len(test_img), len(test_lbl))

test_loss, test_acc = model.evaluate(test_img, test_lbl)
print(f"Test accuracy: {test_acc}")

predictions = model.predict(test_img)

# this program _just_ trains the model and creates a FILE that CONTAINS THE MODEL. if you wanna just see the results go to the other python file :]
# i moved the part that actually picks a random image and 'reads' what number it is to a different python script so i wouldn't have to re-train the model every single time i want to see a new image 
# look jellyfiSHES: <:[=    <:[=   <:[=     <:[=