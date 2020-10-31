import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""

#fitting simple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
Regressor = LinearRegression() # do not import anything becuase it simple
Regressor.fit(X_train,y_train)

#predictting the test set result
y_pred= Regressor.predict(X_test)

#visualiaing the Training set result
plt.scatter(X_train, y_train , color='green')
plt.plot(X_train, Regressor.predict(X_train) , color ='blue')
plt.title('salary Vs Experience (training set)')
plt.xlabel('years of Experience')
plt.ylabel('Salary')
plt.show()

#visualiaing the Test set result
plt.scatter(X_test, y_test , color='red')
plt.plot(X_train, Regressor.predict(X_train) , color ='yellow')
plt.title('salary Vs Experience (testing set)')
plt.xlabel('years of Experience')
plt.ylabel('Salary')
plt.show()