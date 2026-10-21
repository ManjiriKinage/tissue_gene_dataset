import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("tissue_gene_expression.csv")

# Separate labels and features
y = df["tissue"]
X = df.drop(columns=["tissue"])

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means with K = 7
kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Compare clusters with actual tissue types
comparison = pd.crosstab(df["Cluster"], df["tissue"])

print("\nCluster vs Actual Tissue:")
print(comparison)

# Run several times
print("\nMultiple Runs:")

for seed in [1, 10, 20, 30, 42]:
    model = KMeans(n_clusters=7, random_state=seed, n_init=10)
    labels = model.fit_predict(X_scaled)

    print(f"\nRandom State: {seed}")

    for cluster in range(7):
        counts = y[labels == cluster].value_counts()

        if len(counts) > 0:
            print(
                f"Cluster {cluster} -> "
                f"{counts.index[0]} ({counts.iloc[0]} samples)"
            )