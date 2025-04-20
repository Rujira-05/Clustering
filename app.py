import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Page settings
st.set_page_config(page_title="K-Means Clustering App with Iris", layout="centered")

# App title
st.markdown("<h2 style='text-align: center;'>🔍 K-Means Clustering App with Iris Dataset</h2>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Configure Clustering")
k = st.sidebar.slider("Select number of clusters (K)", min_value=2, max_value=10, value=4)

# Load dataset
iris = load_iris()
X = iris.data

# Apply KMeans
model = KMeans(n_clusters=k, random_state=42)
y_kmeans = model.fit_predict(X)

# Reduce to 2D with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
centers_pca = pca.transform(model.cluster_centers_)

# Plotting (adjust size)
fig, ax = plt.subplots(figsize=(6, 4))  # ⬅️ Smaller figure size
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='tab10', s=40)

# Plot centroids
ax.scatter(centers_pca[:, 0], centers_pca[:, 1], c='black', s=100, marker='X', label='Centroids')

# Labeling
ax.set_title("Clusters (2D PCA Projection)", fontsize=12)
ax.set_xlabel("PCA1", fontsize=10)
ax.set_ylabel("PCA2", fontsize=10)

# Legend
handles, _ = scatter.legend_elements()
labels = [f"Cluster {i}" for i in range(k)]
ax.legend(handles, labels, title="Clusters", fontsize=9)

# Show plot
st.pyplot(fig)
