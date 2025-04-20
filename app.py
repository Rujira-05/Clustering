# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Page config
st.set_page_config(page_title="K-Means Clustering App with Iris", layout="wide")

# Title
st.title("🔍 K-Means Clustering App with Iris Dataset")

# Load data
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Sidebar
st.sidebar.header("⚙️ Configure Clustering")
k = st.sidebar.slider("Select number of clusters (K)", min_value=2, max_value=10, value=3)

# K-Means clustering
model = KMeans(n_clusters=k, random_state=42)
y_kmeans = model.fit_predict(X)

# Reduce dimensions using PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Plotting
fig, ax = plt.subplots()
scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_kmeans, cmap='tab10', s=50)
centers_reduced = pca.transform(model.cluster_centers_)
ax.scatter(centers_reduced[:, 0], centers_reduced[:, 1], c='black', s=200, marker='X', label='Centroids')
ax.set_title("Clusters (2D PCA Projection)")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.legend(*scatter.legend_elements(), title="Cluster")

# Show plot
st.pyplot(fig)
