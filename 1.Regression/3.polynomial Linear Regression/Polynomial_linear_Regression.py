
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#fitting linear Regression to the dataset 
from sklearn.linear_model import LinearRegression
lin_reg= LinearRegression()
lin_reg.fit(X,y)

#fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures(degree=4)
X_poly= poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2= LinearRegression()
lin_reg_2.fit(X_poly, y)

#visulising the linear Regression 
plt.scatter(X, y, color='red')
plt.plot(X,lin_reg.predict(X), color='blue')
plt.title('truth or buff (linear Regression)')
plt.xlabel('position level')
plt.ylabel('salaries')
plt.show()

#visulising the polynomial Regression 
X_grid=np.arange(min(X),max(X), 0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y , color='green')
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth or buff (Polynomial Regression)')
plt.xlabel('position Label')
plt.ylabel('salaries')
plt.show()

#prediction linear
lin_reg.predict([[6.5]])

#prediction polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))


