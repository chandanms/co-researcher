import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def imports():
    import numpy as np
    import pandas as pd
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.datasets import load_iris
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    return DBSCAN, KMeans, StandardScaler, load_iris, np, pd, silhouette_score


@app.cell
def load_data(load_iris, pd):
    # Load the iris dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    df
    return df, iris


@app.cell
def prepare_features(StandardScaler, df):
    # Select features for clustering (exclude target)
    features = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]
    X = df[features].values

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, features, scaler


@app.cell
def kmeans_clustering(KMeans, X_scaled, silhouette_score):
    # KMeans clustering with 3 clusters (matching iris species)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    # Calculate silhouette score
    kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)

    kmeans_result = {
        "algorithm": "KMeans",
        "n_clusters": 3,
        "silhouette_score": kmeans_silhouette,
        "labels": kmeans_labels,
    }
    return kmeans, kmeans_labels, kmeans_result, kmeans_silhouette


@app.cell
def dbscan_clustering(DBSCAN, X_scaled, np, silhouette_score):
    # DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)

    # Calculate silhouette score (only if more than 1 cluster found)
    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)

    if n_clusters_dbscan > 1:
        # Exclude noise points (-1) from silhouette calculation
        mask = dbscan_labels != -1
        if mask.sum() > n_clusters_dbscan:
            dbscan_silhouette = silhouette_score(X_scaled[mask], dbscan_labels[mask])
        else:
            dbscan_silhouette = np.nan
    else:
        dbscan_silhouette = np.nan

    n_noise = (dbscan_labels == -1).sum()

    dbscan_result = {
        "algorithm": "DBSCAN",
        "n_clusters": n_clusters_dbscan,
        "n_noise_points": int(n_noise),
        "silhouette_score": dbscan_silhouette,
        "labels": dbscan_labels,
    }
    return (
        dbscan,
        dbscan_labels,
        dbscan_result,
        dbscan_silhouette,
        n_clusters_dbscan,
        n_noise,
    )


@app.cell
def compare_results(dbscan_result, df, kmeans_result, pd):
    # Create comparison DataFrame
    comparison = pd.DataFrame(
        [
            {
                "Algorithm": kmeans_result["algorithm"],
                "Clusters Found": kmeans_result["n_clusters"],
                "Silhouette Score": round(kmeans_result["silhouette_score"], 4),
                "Noise Points": 0,
            },
            {
                "Algorithm": dbscan_result["algorithm"],
                "Clusters Found": dbscan_result["n_clusters"],
                "Silhouette Score": round(dbscan_result["silhouette_score"], 4)
                if not pd.isna(dbscan_result["silhouette_score"])
                else "N/A",
                "Noise Points": dbscan_result["n_noise_points"],
            },
        ]
    )

    # Add cluster labels to original dataframe for analysis
    df_with_clusters = df.copy()
    df_with_clusters["kmeans_cluster"] = kmeans_result["labels"]
    df_with_clusters["dbscan_cluster"] = dbscan_result["labels"]

    comparison
    return comparison, df_with_clusters


@app.cell
def summary(comparison, kmeans_silhouette, dbscan_silhouette, pd):
    # Determine the winner
    if pd.isna(dbscan_silhouette):
        winner = "KMeans"
        reason = "DBSCAN did not find valid clusters for silhouette comparison"
    elif kmeans_silhouette > dbscan_silhouette:
        winner = "KMeans"
        reason = f"Higher silhouette score ({kmeans_silhouette:.4f} vs {dbscan_silhouette:.4f})"
    else:
        winner = "DBSCAN"
        reason = f"Higher silhouette score ({dbscan_silhouette:.4f} vs {kmeans_silhouette:.4f})"

    summary_text = f"""
    Clustering Comparison Summary
    =============================

    Winner: {winner}
    Reason: {reason}

    Note: KMeans requires specifying the number of clusters (set to 3 for iris).
    DBSCAN automatically determines cluster count based on density parameters.
    """

    print(summary_text)
    return reason, summary_text, winner


if __name__ == "__main__":
    app.run()
