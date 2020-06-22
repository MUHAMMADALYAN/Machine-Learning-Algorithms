import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
import matplotlib.pyplot as plt

#KNN stands for K-Nearest Neighbors. KNN is a machine learning algorithm used for classifying data. Rather than
#coming up with a numerical prediction such as a students grade or stock price it attempts to classify data into
# certain categories

data=pd.read_csv("car.data",sep=",")

#We will start by creating a label encoder object and then use that to encode each column of our data into integers
le=preprocessing.LabelEncoder()
#The method fit_transform() takes a list (each of our columns) and will return to us an array containing our new values.
buying=le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

#seperate data input and output
x=list(zip(buying,maint,door,persons,lug_boot,safety,cls))
y=list(cls)

#split between training and test

x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)
model=KNeighborsClassifier(n_neighbors=9)
model.fit(x_train,y_train)

predict=model.predict(x_test)
cls=["unacc", "acc", "good", "vgood"]
for i in range(len(predict)):
    print("predicted",cls[predict[i]],"test data:",x_test[i],"actual_data:",cls[y_test[i]])
    #neighbors method takes 2D as input, this means if we want to pass one data point we need surround it with []
    n=model.kneighbors([x_test[i]],9,True)
    print("neighbours:",n)

A = model.kneighbors_graph(x_test, 9)
plot="values"
plt.scatter(buying,y)
plt.xlabel(plot)
plt.ylabel("classes")
plt.show()