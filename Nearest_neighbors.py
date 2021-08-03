import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# define the input data
X = np.array([[3.1, 2.3], [2.3, 4.2], [3.9, 3.5], [3.7, 6.4], [4.8, 1.9],
             [8.3, 3.1], [5.2, 7.5], [4.8, 4.7], [3.5, 5.1], [4.4, 2.9],])


# define the nearest neighbors
k = 3

# test data from which NN is to be found
test_data = [3.3, 2.9]

# visualize
plt.figure()
plt.title('Input data')
plt.scatter(X[:,0], X[:,1], marker='o', s=100, color = 'black')

# build the kNN model
knn_model = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X)
distances, indices = knn_model.kneighbors([test_data])

# print the kNNs
print("\n K Nearest Neighbors")
for rank, index in enumerate(indices[0][:k], start=1):
    print(str(rank)+ " is ", X[index])

# visualize the NNs with test data point
plt.figure()
plt.title('Nearest Neighbors')
plt.scatter(X[:,0],X[:,1],marker='o', s=100, color='k')
plt.scatter(X[indices][0][:][:,0],X[indices][0][:][:,1],
            marker='o', s=250, color='k', facecolors='none')
plt.scatter(test_data[0], test_data[1],
            marker='x', s=100, color = 'k')
plt.show()
