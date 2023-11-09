from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# Function to plot true labels and cluster labels side by side
def plot_true_vs_clustered(X, true_labels, cluster_labels, cluster_centers=None):
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Plot true labels
    axs[0].set_title('True Labels')
    unique_true_labels = set(true_labels)
    colors_true = plt.cm.rainbow(np.linspace(0, 1, len(unique_true_labels)))
    for k, col in zip(unique_true_labels, colors_true):
        class_member_mask = (true_labels == k)
        xy = X[class_member_mask]
        axs[0].plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=10)

    # Plot clustered labels
    axs[1].set_title('Clustered Labels')
    unique_cluster_labels = set(cluster_labels)
    colors_cluster = plt.cm.rainbow(np.linspace(0, 1, len(unique_cluster_labels)))
    for k, col in zip(unique_cluster_labels, colors_cluster):
        class_member_mask = (cluster_labels == k)
        xy = X[class_member_mask]
        axs[1].plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=10)
        if cluster_centers is not None and k != -1:
            center = cluster_centers[k]
            axs[1].plot(center[0], center[1], 'o', markerfacecolor=col, markeredgecolor='k', markeredgewidth=2, markersize=14)

    plt.show()

# Function to plot the first two columns
def plot_clustered(X, cluster_centers):
    X = np.asarray(X)
    nfeature = X.shape[-1]
    if nfeature > 2:
        X = X[:,:2]
    if nfeature < 2:
        print("Error")
    # Plot clustered labels
    unique_cluster_labels = set(cluster_labels)
    colors_cluster = plt.cm.rainbow(np.linspace(0, 1, len(unique_cluster_labels)))
    for k, col in zip(unique_cluster_labels, colors_cluster):
        class_member_mask = (cluster_labels == k)
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col)
    plt.show()

# Generate synthetic data
X, true_labels = make_blobs(n_samples=300, centers=4, cluster_std=0.80, random_state=0)

# KMeans crucial parameters: n_clusters, init, n_init, max_iter, random_state
# This is the method
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=0)
cluster_labels = kmeans.fit_predict(X)

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
plot_clustered(X, cluster_labels)
#plot_true_vs_clustered(X, true_labels, cluster_labels)
