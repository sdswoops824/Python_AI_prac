import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets._samples_generator import make_blobs

# generate a 2=D dataset containing 4 blobs
X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=0.40, random_state=0)
plt.scatter(X[:,0], X[:,1], s=50)
plt.show()

# initialize kmeans to be the KMeans algorithm with required number of clusters
kmeans = KMeans(n_clusters=4)

# Train the K-Means model with input data
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
plt.scatter(X[:,0], X[:,1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_

# Plot another graph to visualize the clusters
plt.scatter(centers[:, 0], centers[:, 1], c = 'black', s = 200, alpha = 0.5)
plt.show()