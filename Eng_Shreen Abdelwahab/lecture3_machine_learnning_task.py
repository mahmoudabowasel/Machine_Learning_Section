# محمود كمال ابو واصل عوض
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

data = {
    "Point": ["P1", "P2", "P3", "P4", "P5", "P6", "P7"],
    "x": [1.0, 1.5, 3.0, 5.0, 3.5, 4.5, 3.5],
    "y": [1.0, 2.0, 3.5, 7.0, 5.0, 5.0, 4.5]
}

df = pd.DataFrame(data)
X = df[["x", "y"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []
silhouette_scores = []
k_values = range(2, min(7, len(df)))

for k in k_values:
    model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = model.fit_predict(X_scaled)
    inertias.append(model.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

best_k = list(k_values)[int(np.argmax(silhouette_scores))]
print("Best K according to silhouette score:", best_k)

kmeans = KMeans(n_clusters=best_k, init="k-means++", n_init=10, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

centers_scaled = kmeans.cluster_centers_
centers = scaler.inverse_transform(centers_scaled)

print("\nData with cluster assignments:")
print(df.to_string(index=False))

print("\nCluster centers:")
for i, center in enumerate(centers):
    print(f"Cluster {i}: x = {center[0]:.2f}, y = {center[1]:.2f}")

print("\nInertia (WCSS):", round(kmeans.inertia_, 4))
print("Silhouette score:", round(silhouette_score(X_scaled, df["Cluster"]), 4))

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), inertias, marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method")
plt.xticks(list(k_values))
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), silhouette_scores, marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method")
plt.xticks(list(k_values))
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
for cluster in sorted(df["Cluster"].unique()):
    points = df[df["Cluster"] == cluster]
    plt.scatter(points["x"], points["y"], s=100, label=f"Cluster {cluster}")
plt.scatter(centers[:, 0], centers[:, 1], s=250, marker="X", label="Centroids")
for _, row in df.iterrows():
    plt.annotate(row["Point"], (row["x"], row["y"]), xytext=(5, 5), textcoords="offset points")
plt.xlabel("x")
plt.ylabel("y")
plt.title("K-means Clustering Result")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
