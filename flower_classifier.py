import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
iris_dataset = load_iris()
print("keys of iris_dataset: \n{}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193]+"\n...")
print("target names: {}".format(iris_dataset['target_names']))
print("Feature names: {}".format(iris_dataset['feature_names']))
print("types of data: {}".format(type(iris_dataset['data'])))
print("shape of data: {}".format(iris_dataset['data'].shape))
print("First five column of data: \n{}".format(iris_dataset['data'][:5]))
print("type of target: {}".format(type(iris_dataset['target'])))
print("shape of target: {}".format(iris_dataset['target'].shape))
print("target:\n{}".format(iris_dataset['target']))
#0 meaning setosa/1 means versicolor/2 means virginica
X_train,X_test,y_train,y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)
print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))
#iris_dataframe = pd.DataFrame(X_train,columns=iris_dataset.feature_names)
#grr = pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
#hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
#make a new prediction where the sepal length = 5,sepal width = 2.9,petal length = 1,petal width = .2
X_new = np.array([[5,2.9,1,.2]])
print('x_new shape : {} '. format(X_new.shape))
prediction = knn.predict(X_new)
print('Prediction:{}'.format(prediction))
print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))
pred = knn.predict(X_test)
print('X_test prediction : {}'.format(pred))
print('X_test target : {}'.format(iris_dataset['target_names'][pred]))
print("Test set score: {:.2f}".format(np.mean(pred == y_test)))
