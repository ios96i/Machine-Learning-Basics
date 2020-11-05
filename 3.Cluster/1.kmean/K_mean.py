
# k means clustering 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

#using the Elbow method to find optimal K clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans= KMeans(n_clusters=i ,init= 'k-means++',max_iter= 300, n_init=10 , random_state=0 )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clustor')
plt.ylabel('Wcss')
plt.show()

# Applying K-mean to the mall dataset
kmeans= KMeans(n_clusters=5, init='k-means++',max_iter=300 , n_init=10 ,random_state=0)#depend on Elebow table 
y_means= kmeans.fit_predict(X)


#visulaising the clusters 
plt.scatter(X[y_means==0,0], X[y_means==0,1], s=100 ,c='red' ,label='mid')
plt.scatter(X[y_means==1,0], X[y_means==1,1], s=100 ,c='green' ,label='cl')
plt.scatter(X[y_means==2,0], X[y_means==2,1], s=100 ,c='blue' ,label='cluster 3')
plt.scatter(X[y_means==3,0], X[y_means==3,1], s=100 ,c='purple' ,label='cluster 4')
plt.scatter(X[y_means==4,0], X[y_means==4,1], s=100 ,c='violet' ,label='cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s=300 , c='yellow', label='centroids')
plt.title('clusters of clients')
plt.xlabel('Annual Income')
plt.ylabel('spending score (0-100)')
plt.legend()
plt.show()
