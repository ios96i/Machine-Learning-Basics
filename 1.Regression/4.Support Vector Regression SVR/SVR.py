#SVR
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X= sc_X.fit_transform(X)
y= sc_y.fit_transform(y.reshape(-1,1))

#fitting SVR to the dataset 
from sklearn.svm import SVR
regressor = SVR(kernel= 'rbf') #gussian plot
regressor.fit(X,y)

#predict SVR
y_pred= regressor.predict([[6.5]])
y_pred= sc_y.inverse_transform(y_pred)

#visulising the SVRegression 
plt.scatter(X, y, color='red')
plt.plot(X,regressor.predict(X), color='blue')
plt.title('truth or buff (SVR)')
plt.xlabel('position level')
plt.ylabel('salaries')
plt.show()

