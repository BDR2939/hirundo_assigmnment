import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.mixture import GaussianMixture


def select_clean_subset(df: pd.DataFrame, method: str = "percentile", frac: float = 0.9):
    """Select a subset of samples based on tracin self-influence score (potentially mislabeled).

    Args:
        df: DataFrame containing an "influence" column.
        method: One of {"percentile", "knee", "gmm"}.
        frac: For percentile method, the quantile to keep (e.g., 0.9 keeps bottom 90%).

    Returns:
        clean_df: DataFrame filtered to exclude high-influence samples.
        cutoff_info: Dict describing the cutoff used.
    """
    if method == "percentile":
        cutoff = df["influence"].quantile(frac)
        clean_df = df[df["influence"] <= cutoff]
        cutoff_info = {"method": "percentile", "cutoff": float(cutoff)}
    elif method == "knee":
        sorted_scores = np.sort(df["influence"].values)[::-1]
        knee = KneeLocator(range(len(sorted_scores)), sorted_scores, curve="convex", direction="decreasing")
        cutoff = sorted_scores[knee.knee]
        clean_df = df[df["influence"] <= cutoff]
        cutoff_info = {"method": "knee", "cutoff": float(cutoff)}
    elif method == "gmm":
        gmm = GaussianMixture(n_components=2, random_state=0).fit(df["influence"].values.reshape(-1, 1))
        labels = gmm.predict(df["influence"].values.reshape(-1, 1))
        high_cluster = int(np.argmax(gmm.means_))
        clean_df = df[labels != high_cluster]
        cutoff_info = {"method": "gmm", "cluster_means": gmm.means_.ravel().tolist()}
    else:
        raise ValueError(f"Unknown method: {method}")

    return clean_df, cutoff_info


