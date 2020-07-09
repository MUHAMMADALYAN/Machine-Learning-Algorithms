import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense

mnist = tf.keras.datasets.mnist
#it will load 60,000 images
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
#shape indicates number of elements in each dimension .
# it will have 60000 arrays in outer dimension within these array we have 28 arrays per array and these 28 arrays has 28 element
print('training_labels=',training_labels.shape)
print('training_images=',training_images.shape)
print('test_images=',test_images.shape)
print('test_labels=',test_labels.shape)
#seeing labels
"""print('test_labels=',test_labels[0])
#seeing set of all unique labels
print('training_labels=',set(training_labels))

import matplotlib.pyplot as plt
#it will read image and color map is binary because we have black and white images
plt.imshow(training_images[0],cmap='binary')
#ploting images
plt.show()"""

#So now we are changing represeation of labels in a way that particular class will be 1 and all other
#classes will be zero in the list . This is called One Hot Encoding
#the reason for doing is that now our neural network will predict which switch
# is on out of all the switches instead of predicting numerical value (which is basically linear regression and basically used for predicting numerical value)
# but here it will tell us the class
# this is the classification because for ery class there is switch
from tensorflow.keras.utils import to_categorical
training_labels_ecoded=to_categorical(training_labels)
test_labels_ecoded=to_categorical(test_labels)
print('training_labels=',training_labels_ecoded.shape)
print('training_images=',test_labels_ecoded.shape)

#proeessig on traing data
 #we will make a neural network that will take 784 dimensional vectors and it will output 10 dimensional vector
# for 10 different classes
#no we convert our each inputs which are 28 by 28  into 784x1

mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images.reshape(60000, 28, 28,1)
training_images=training_images / 255.0
test_image_ploting=test_images
test_images = test_images.reshape(10000, 28, 28,1)
test_images=test_images/255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28,1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])


#‘Activation’ is the activation function for the layer.
# An activation function allows models to take into account nonlinear relationships.
#The activation function we will be using is ReLU or Rectified Linear Activation.
#it converts net value in 2019 other activation function is softmax with gives us the probability for various nodes
# (in this case  classes)

#add model layers

#in order to optimise the weights and the intercept of the datasets we define optimizer algorithm

model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
model.fit(training_images, training_labels_ecoded, epochs=2)
acc=model.evaluate(test_images,test_labels_ecoded)
print("Alyan accuracy on test data=",acc)

preds=model.predict(test_images)
plt.figure(figsize=(24,24))

start_index=0
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    pred=np.argmax(preds[start_index+i])
    y_real=test_labels[start_index+i]


    plt.xlabel("i={},prediction={},real_Value{}".format(start_index+i,pred,y_real))
    plt.imshow(test_image_ploting[start_index+1],cmap='binary')
plt.show()

"""import numpy as np
import random
from   tensorflow.keras.preprocessing.image import img_to_array, load_img

#model.layers  api is used to display outputs of convolutions and iterate through the
successive_outputs = [layer.output for layer in model.layers[1:]]

# Let's define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model after
# the first.
#visualization_model = Model(img_input, successive_outputs)
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
# Let's prepare a random input image of a cat or dog from the training set."""







