import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics


# same code as K-Means clustering
from sklearn.datasets._samples_generator import make_blobs
X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=0.40, random_state=0)

# Initialize variables
scores = []
values = np.arange(2, 10)

# iterate the K-means model and train with input data
for num_clusters in values:
    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
    kmeans.fit(X)

# estimate the silhouette score
score = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean', sample_size=len(X))

# display the number of clusters and silhouette score
print("\nNumber of clusters =", num_clusters)
print("Silhouette score = ", score)
scores.append(score)

# optimal number of clusters
num_clusters = np.argmax(scores) + values[0]
print('\nOptimal number of clusters =', num_clusters)

# Plot another graph to visualize the clusters
y_kmeans = kmeans.predict(X)
plt.scatter(X[:,0], X[:,1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c = 'black', s = 200, alpha = 0.5)
plt.show()