from sklearn.cluster import KMeans


def run_clustering(X):
    model = KMeans(n_clusters=3, random_state=42)
    clusters = model.fit_predict(X)

    print("\n=== CLUSTERING ===")
    print("Primeros clusters:", clusters[:10])

    return clusters