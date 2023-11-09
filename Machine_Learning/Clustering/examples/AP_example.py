from sklearn.cluster import AffinityPropagation
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from collections import Counter
from ClusterPlot import *

# Generate synthetic data
X, true_labels = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# AffinityPropagation crucial parameters: damping, max_iter, convergence_iter, copy, preference, affinity, verbose
affinity_propagation = AffinityPropagation(damping=0.9, max_iter=200, convergence_iter=15, copy=True, preference=-100, affinity='euclidean', verbose=False)
cluster_labels = affinity_propagation.fit_predict(X)

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
