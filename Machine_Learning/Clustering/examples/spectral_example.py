from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from collections import Counter
from ClusterPlot import *

# Generate synthetic data
X, true_labels = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# SpectralClustering crucial parameters:
# n_clusters: number of clusters to find
# affinity: how to construct the affinity matrix
# n_neighbors: Number of neighbors to use when constructing the affinity matrix using the nearest neighbors method. Ignored for affinity='rbf'.
# n_init: number of times the algorithm will be run with different centroid seeds
spectral_clustering = SpectralClustering(n_clusters=4,eigen_solver=None,n_components=4,random_state=0,n_init=10,gamma=1.0,affinity='rbf',
                                            n_neighbors=10,eigen_tol=0.0,assign_labels='kmeans',degree=3,
                                            coef0=1,kernel_params=None,n_jobs=-1)  # -1 means using all processors)
cluster_labels = spectral_clustering.fit_predict(X)

### The algorithm end here, the following code calculates the number of each clusters and plots it
cluster_counts = Counter(cluster_labels)
print(f"Algorithm found {len(cluster_counts)} clusters.")
print("Points per cluster:", cluster_counts)
true_label_counts = Counter(true_labels)
print(f"True labels have {len(true_label_counts)} clusters.")
print("Points per cluster for true labels:", true_label_counts)

# Silhouette Score
print(f"Silhouette Score: {silhouette_score(X, cluster_labels)}\n")

# Plot true labels vs. clustering results
plot_true_vs_clustered(X, true_labels, cluster_labels)
