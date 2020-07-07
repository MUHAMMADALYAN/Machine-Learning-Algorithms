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

training_images_rehape=np.reshape(training_images,(60000,784))
test_images_reshape=np.reshape(test_images,(10000,784))

#so now we have  lists of all inputs x that maps to all possible gven lists of labels one input by one and learn
#normalizing data
x_trainig_images_norm=training_images_rehape/255
x_testig_images_norm=test_images_reshape/255

#we will define the number of nodes according to the number of feachers hair we have 784 features therefore we will use 128 or
# something
model=Sequential()

#‘Activation’ is the activation function for the layer.
# An activation function allows models to take into account nonlinear relationships.
#The activation function we will be using is ReLU or Rectified Linear Activation.
#it converts net value in 2019 other activation function is softmax with gives us the probability for various nodes
# (in this case  classes)

#add model layers
model.add(Dense(128,activation='relu',input_shape=(784,)))
model.add(Dense(512, activation='relu'))
#output layer with 1 node
model.add(Dense(10,activation='softmax'))

#in order to optimise the weights and the intercept of the datasets we define optimizer algorithm

model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
model.fit(training_images_rehape, training_labels_ecoded, epochs=3)
acc=model.evaluate(test_images_reshape,test_labels_ecoded)
print("acc=",acc)

preds=model.predict(x_testig_images_norm)
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
    plt.imshow(test_images[start_index+1],cmap='binary')
plt.show()

