# Neural Network Project--Handwritten Digit Recognizer

This is a simple handwritten digit recognizer I made using tf/keras, matplot, and numpy. For the images themselves, I used mnist, a library that contains samples of a bunch of handwritten digits. Each image is 28x28 and in grayscale to make everything easier.

Here's a breakdown of the ideas that make this work:

This is a supervised learning model, so first, we split the msnit samples into two datasets: training, and testing. We have two variables: the images themselves and the labels. For example, we have an image of the number seven, and the label attached to it would be '7', and this would be either in the testing or the training pool.

The 'image' itself is stored in the computer as an array of values from 0-255, with 0 being white, and 255 being black, with everything else in between being shades of gray. 0-255 is a HUGE range of values and computational-heavy, so we divide all the numbers in the array by 255, so we get values between 0 and 1, which is easier to work with.

The next step is to flatten the image. An image is basically a 2D representation of an array with shades of white/black/gray rather than values 0-1. We flatten this into 1 dimension, so now all the values are lined up, one after the other, something like this: [0, 0, 0, 0.1, 0.5, 0.9...] 

Because each number is different, each number will have 0s (no value) and 1s (darkest value). This means that we can analyze each 'type' of number using statistics. For example, every time I write the number '7', there will usually be a horizontal line around the top-ish area, and a vertical line in the right-ish area. This ends up returning the most intense values in a certain range in the array. The NN sees this, and then recognizes the pattern that a '7' ends up being. Tensorflow takes care of all the complicated math stuff, so all we have to worry about is the high-level concepts.

All that's left now is to compile the model, and run it. 

All the display stuff that you can see in Jupyter is done by matplotlib.

That's a really quick overview, feel free to download the python or jupyter files to mess around with them. 

AB
