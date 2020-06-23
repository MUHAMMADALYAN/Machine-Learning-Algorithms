import sklearn
from sklearn import svm,metrics
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

ds=datasets.load_breast_cancer()

x=ds.data
y=ds.target
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

#we are calling its calssifier
cls = svm.SVC(kernel="linear")
cls.fit(x_train,y_train)
pred=cls.predict(x_test)
acc=metrics.accuracy_score(y_test,pred)
print(acc)