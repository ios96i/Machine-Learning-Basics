
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:, 13].values

# Encoding the Independent Variable
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
lb_X_1=LabelEncoder()
X[:,1]=lb_X_1.fit_transform(X[:,1])
X[:,2]=lb_X_1.fit_transform(X[:,2])
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)
X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#part 2 - now lets make ANN!

#importing ANN laibraries 
import keras
from keras.models import Sequential
from keras.layers import Dense

#initilising the ANN (defined as sequenial)
classifier= Sequential()

#Adding the input layers, Hidden first Layers 
classifier.add(Dense(units=6 ,kernel_initializer='uniform', activation='relu',input_dim=11))#relu is rectifier function

#Adding the second hidden layers 
classifier.add(Dense(units =6, kernel_initializer='uniform', activation='relu'))

#Adding the output layers 
classifier.add(Dense(units=1, kernel_initializer ='uniform',activation='sigmoid'))

#compiling the ANN to the training set
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fitting the ANN to the Training set
classifier.fit(X_train, y_train ,batch_size=10 , nb_epoch= 100)

#predict the test result
y_pred= classifier.predict(X_test)
y_pred=(y_pred>0.5)
# Making confusion matrix
from sklearn.metrics import confusion_matrix 
cm =confusion_matrix(y_test, y_pred)
