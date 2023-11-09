import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
import pandas as pd

# Generate synthetic data
X, true_labels = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Convert the true labels to a color palette
lut = sns.color_palette('tab10', n_colors=len(set(true_labels)))
row_colors = pd.Series(true_labels).map(dict(zip(range(len(set(true_labels))), lut)))

# Perform hierarchical clustering
pairwise_dists = pdist(X)
linkage_matrix = linkage(pairwise_dists, method="ward")

# Compute the distance matrix for plotting
distance_matrix = squareform(pairwise_dists)

# Heatmap of the distance matrix with the previous clustering tree
sns.clustermap(distance_matrix, row_cluster=True, col_cluster=True,
               row_linkage=linkage_matrix, col_linkage=linkage_matrix,
               figsize=(10, 10), cmap='viridis', xticklabels=False, yticklabels=False)

plt.show()
