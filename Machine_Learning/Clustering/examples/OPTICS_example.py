from sklearn.cluster import OPTICS
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from collections import Counter
from ClusterPlot import *

# Generate synthetic data
X, true_labels = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# OPTICS crucial parameters: min_samples, max_eps, metric, cluster_method, eps, xi, 
# predecessor_correction, min_cluster_size, algorithm, leaf_size, p
optics = OPTICS(min_samples=30, max_eps=np.inf, metric='minkowski', cluster_method='xi', eps=None, xi=0.01, 
                predecessor_correction=True, min_cluster_size=30, algorithm='auto', leaf_size=30, p=2)
cluster_labels = optics.fit_predict(X)

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
