import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
data=pd.read_csv("student-mat.csv",sep=";")
# Since our data is seperated by semicolons we need to do sep=";"

#trimming data
data=data[["G1","G2","G3","absences","studytime","failures"]]
test_data=data.iloc[100:150]

data=data.iloc[1:99]

#To see our data frame we can type:

predict="G3"
#seperating data
x=data.drop(["G3"],axis=1)
columns=x.shape[1]

y=data["G3"]

x_testdata=test_data.drop(["G3"],axis=1)
y_testdata=test_data["G3"]
print(y)
print(x)

model=Sequential()
# here is 10 neurons
#‘Activation’ is the activation function for the layer.
# An activation function allows models to take into account nonlinear relationships.
#The activation function we will be using is ReLU or Rectified Linear Activation.
# Although it is two linear pieces, it has been proven to work well in neural networks.

#add model layers
model.add(Dense(10,activation='relu',input_shape=(columns,)))
model.add(Dense(10, activation='relu'))
#output layer with 1 node
model.add(Dense(1))
#compile model using mse as a measure of model performance
#Once it's done training -- you should see an accuracy value at the end of the final epoch.
model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])


#set early stopping monitor so the model stops training when it won't improve anymore here we set for 3 iterations
early_stopping_monitor = EarlyStopping(patience=3)
#train model

model.fit(x, y, validation_split=0.2, epochs=100, callbacks=[early_stopping_monitor])
#
predictions = model.predict(x_testdata)

#converting data frame to
for i in range(len(predictions)):
    print(predictions[i],x_testdata.iloc[i],y_testdata.iloc[i])

