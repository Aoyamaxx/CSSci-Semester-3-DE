from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from collections import Counter
from ClusterPlot import *

# Generate synthetic data
X, true_labels = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# AgglomerativeClustering crucial parameters: n_clusters, affinity, memory, 
# connectivity, compute_full_tree, linkage, distance_threshold
agg_cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', 
                                    memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None)
cluster_labels = agg_cluster.fit_predict(X)

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
