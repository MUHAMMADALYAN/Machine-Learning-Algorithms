import tensorflow
import keras
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
import pickle
from sklearn.utils import shuffle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style

#loading data
data=pd.read_csv("student-mat.csv",sep=";")
# Since our data is seperated by semicolons we need to do sep=";"

#trimming data
data=data[["G1","G2","G3","absences","studytime","failures"]]
#To see our data frame we can type:

predict="G3"
#seperating data
x=np.array(data.drop(["G3"],axis=1))
y=np.array(data["G3"])

#scikit provides many unsupervised and supervised learning algorithms
#spliting our data into testing and training data
x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)
print(x_train[0])
best=0
for i in range(30):

    x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)
    # defing LinearRegression Algorithm
    #Implementing LinearRegression Algorithm that will generate a line upon given dat
    linear=linear_model.LinearRegression()
    # train our model on given data and trainin it to  perform x--> y mapping we will tell both input and ouput to train
    linear.fit(x_train,y_train)
    # checking accuracy (LR algoritm will generate straight line on the basis of trainindata and then tell accuracy== how many points lie below
    acc = (linear.score(x_test, y_test))
    print("Accuracy: " + str(acc))
    if acc > best:
        best=acc
    with open("student.pickle","wb") as f:
        pickle.dump(linear,f)

pickle_in= open("student.pickle","rb")
linear=pickle.load(pickle_in)

predictions=linear.predict(x_test)# Gets a list of all predictions
for i in range(len(predictions)):
    print(predictions[i],x_test[i],y_test[i])

plot="G2"
plt.scatter(data[plot],data["G3"])
plt.xlabel(plot)
plt.ylabel("final grades")
plt.show()