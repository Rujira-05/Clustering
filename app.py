import streamlit as st
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Load model
with open('kmeans_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

st.set_page_config(page_title="K-Means Clustering App", layout="centered")
st.title("💠 K-Means Clustering Visualizer")
st.subheader("📊 Example Data for Visualization")
st.markdown("This demo uses example data (2D) to illustrate clustering results.")

# Create synthetic data
X, _ = make_blobs(n_samples=300, centers=loaded_model.n_clusters,
                  cluster_std=0.60, random_state=0)

# Predict with model
y_kmeans = loaded_model.predict(X)

# Plot the result
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = loaded_model.cluster_centers_
ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.75, marker='X', label='Centroids')
ax.legend()

# Show the plot
st.pyplot(fig)
