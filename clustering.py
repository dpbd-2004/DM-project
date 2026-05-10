"""
=============================================================================
MODULE 5 - CLUSTERING
Project : Spatial-Temporal Mining of Urban Heatwave Precursors and
          Regional Vulnerability Clustering
Course  : Data Mining (6th Semester)
File    : clustering.py
=============================================================================
Responsibilities
----------------
1. Load city_attributes.csv (Lat/Lon) and compute per-city average
   temperature, humidity, pressure, and wind_speed from preprocessed_data.csv.
2. Build a 5-dimensional feature matrix per city for clustering:
      [Latitude, Longitude, avg_temperature, avg_humidity, avg_wind_speed]
3. Implement three clustering techniques:
      a) k-Means        - centroid-based; k=4 chosen via Elbow method.
      b) Hierarchical   - agglomerative, Ward linkage; dendrogram plotted.
      c) DBSCAN         - density-based; handles noise/outliers automatically.
4. Generate visualisations:
      - Elbow curve (k-Means inertia vs k)
      - Dendrogram (Hierarchical)
      - 2D geographic scatter maps for all three algorithms
      - 3D scatter (Lat, Lon, avg_temperature) coloured by cluster
5. Print a comparative analysis: DBSCAN outlier handling vs k-Means.
6. Save cluster assignments to outputs/cluster_assignments.csv.
=============================================================================
"""

import os
import time
import warnings
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm     as cm
from   matplotlib        import gridspec
import seaborn           as sns
from   mpl_toolkits.mplot3d import Axes3D                   # noqa: F401

from sklearn.preprocessing  import StandardScaler
from sklearn.cluster        import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics        import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR        = os.path.dirname(__file__)
CITY_CSV        = os.path.join(BASE_DIR, "datasets",  "city_attributes.csv")
PREPROCESSED    = os.path.join(BASE_DIR, "outputs",   "preprocessed_data.csv")
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR       = os.path.join(BASE_DIR, "plots")
CLUSTER_OUT_CSV = os.path.join(OUTPUT_DIR, "cluster_assignments.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

# k-Means parameters
KMEANS_K       = 4        # final number of clusters (confirmed by elbow)
KMEANS_K_RANGE = range(2, 10)
RANDOM_STATE   = 42

# DBSCAN parameters  (tuned for geographic + thermal feature space)
DBSCAN_EPS     = 1.2      # neighbourhood radius (on scaled features)
DBSCAN_MIN_PTS = 2        # minimum points to form a core point

# Features used for clustering
CLUSTER_FEATURES = ["Latitude", "Longitude",
                    "avg_temperature", "avg_humidity", "avg_wind_speed"]

# Colour palette for clusters
PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0",
           "#FF9800", "#00BCD4", "#F44336", "#8BC34A"]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _header(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _subheader(title: str) -> None:
    print(f"\n  --- {title} ---")


def _cluster_metrics(X_scaled, labels, name):
    """Print silhouette score and Davies-Bouldin index for a clustering."""
    mask = labels >= 0          # DBSCAN marks noise as -1; exclude from scoring
    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        print(f"  [{name}] Insufficient clusters for metric computation.")
        return
    sil = silhouette_score(X_scaled[mask], labels[mask])
    dbi = davies_bouldin_score(X_scaled[mask], labels[mask])
    print(f"  [{name}]  Silhouette Score  : {sil:.4f}  "
          f"(higher = better separated)")
    print(f"  [{name}]  Davies-Bouldin    : {dbi:.4f}  "
          f"(lower  = better)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 : LOAD & BUILD PER-CITY FEATURE MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def build_city_features() -> pd.DataFrame:
    """
    Merge city_attributes.csv with per-city averages computed from
    preprocessed_data.csv to produce one row per city.

    Returns
    -------
    pd.DataFrame  with columns:
        City, Country, Latitude, Longitude,
        avg_temperature, avg_humidity, avg_pressure, avg_wind_speed
    """
    _header("STEP 1 - LOADING DATA & BUILDING PER-CITY FEATURE MATRIX")

    # City meta-data
    cities = pd.read_csv(CITY_CSV)
    print(f"  [OK] City attributes loaded : {cities.shape}")

    # Load preprocessed data (only columns we need to save RAM)
    needed = ["city", "temperature", "humidity", "pressure", "wind_speed"]
    df = pd.read_csv(PREPROCESSED, usecols=needed)
    print(f"  [OK] Preprocessed data loaded : {df.shape}")

    # Per-city means
    city_agg = (
        df.groupby("city")
          .agg(
              avg_temperature = ("temperature", "mean"),
              avg_humidity    = ("humidity",    "mean"),
              avg_pressure    = ("pressure",    "mean"),
              avg_wind_speed  = ("wind_speed",  "mean"),
          )
          .reset_index()
          .rename(columns={"city": "City"})
    )
    print(f"  [OK] Per-city aggregation shape : {city_agg.shape}")

    # Merge
    merged = pd.merge(cities, city_agg, on="City", how="inner")
    print(f"  [OK] Merged city feature matrix : {merged.shape}")

    print("\n  Per-city feature summary (avg values):")
    print(merged[["City",
                  "avg_temperature", "avg_humidity",
                  "avg_pressure",    "avg_wind_speed"
                  ]].round(2).to_string(index=False))

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 : SCALE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def scale_features(df: pd.DataFrame):
    """
    Z-score standardise the clustering feature matrix so that
    geographic and meteorological scales are comparable.

    Returns
    -------
    X_scaled (ndarray), scaler (StandardScaler)
    """
    _header("STEP 2 - FEATURE STANDARDISATION (Z-Score)")

    X = df[CLUSTER_FEATURES].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"  Features used  : {CLUSTER_FEATURES}")
    print(f"  Data matrix    : {X_scaled.shape}")
    print(f"  Post-scale mean: {X_scaled.mean(axis=0).round(4)}")
    print(f"  Post-scale std : {X_scaled.std(axis=0).round(4)}")

    return X_scaled, scaler


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 : ELBOW METHOD (k selection for k-Means)
# ─────────────────────────────────────────────────────────────────────────────

def plot_elbow(X_scaled) -> None:
    """
    Compute k-Means inertia for k in KMEANS_K_RANGE and save the
    Elbow curve to plots/elbow_curve.png.
    """
    _header("STEP 3 - ELBOW METHOD (Choosing Optimal k for k-Means)")

    inertias = []
    for k in KMEANS_K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE,
                    n_init=10, max_iter=300)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        print(f"  k={k:2d}  inertia={km.inertia_:.2f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(list(KMEANS_K_RANGE), inertias, "o-",
            color="#2196F3", linewidth=2, markersize=7, markerfacecolor="white",
            markeredgewidth=2)
    ax.axvline(KMEANS_K, color="#FF5722", linestyle="--",
               linewidth=1.5, label=f"Chosen k = {KMEANS_K}")
    ax.set_xlabel("Number of Clusters  k", fontsize=12)
    ax.set_ylabel("Inertia  (Within-cluster SSE)", fontsize=12)
    ax.set_title("Elbow Curve - k-Means Cluster Selection",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out = os.path.join(PLOTS_DIR, "elbow_curve.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [OK] Saved -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 : k-MEANS CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def run_kmeans(X_scaled) -> np.ndarray:
    """
    Apply k-Means with KMEANS_K clusters.

    Parameters
    ----------
    X_scaled : ndarray  Standardised feature matrix

    Returns
    -------
    labels : ndarray of cluster assignments (0 … k-1)
    """
    _header(f"STEP 4 - k-MEANS CLUSTERING  (k={KMEANS_K})")

    t0  = time.perf_counter()
    km  = KMeans(n_clusters=KMEANS_K, random_state=RANDOM_STATE,
                 n_init=10, max_iter=300)
    km.fit(X_scaled)
    labels  = km.labels_
    elapsed = time.perf_counter() - t0

    print(f"  Iterations to converge : {km.n_iter_}")
    print(f"  Final inertia          : {km.inertia_:.4f}")
    print(f"  Wall-clock time        : {elapsed:.4f} s")

    unique, counts = np.unique(labels, return_counts=True)
    print("\n  Cluster sizes:")
    for cl, cnt in zip(unique, counts):
        print(f"    Cluster {cl} : {cnt} cities")

    _cluster_metrics(X_scaled, labels, "k-Means")

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 : HIERARCHICAL (AGGLOMERATIVE) CLUSTERING + DENDROGRAM
# ─────────────────────────────────────────────────────────────────────────────

def run_hierarchical(X_scaled, city_names) -> np.ndarray:
    """
    Apply Agglomerative Clustering with Ward linkage and plot the
    dendrogram to plots/dendrogram.png.

    Parameters
    ----------
    X_scaled   : ndarray  Standardised feature matrix
    city_names : list     City name strings (for dendrogram labels)

    Returns
    -------
    labels : ndarray of cluster assignments (0 … KMEANS_K-1)
    """
    _header(f"STEP 5 - HIERARCHICAL CLUSTERING  "
            f"(Agglomerative | Ward Linkage | k={KMEANS_K})")

    t0 = time.perf_counter()

    # scipy linkage for full dendrogram
    Z = linkage(X_scaled, method="ward")

    # sklearn for flat cluster labels
    hc = AgglomerativeClustering(
        n_clusters=KMEANS_K,
        linkage="ward",
    )
    labels  = hc.fit_predict(X_scaled)
    elapsed = time.perf_counter() - t0

    print(f"  Linkage method  : Ward  (minimises within-cluster variance)")
    print(f"  n_clusters      : {KMEANS_K}")
    print(f"  Wall-clock time : {elapsed:.4f} s")

    unique, counts = np.unique(labels, return_counts=True)
    print("\n  Cluster sizes:")
    for cl, cnt in zip(unique, counts):
        print(f"    Cluster {cl} : {cnt} cities")

    _cluster_metrics(X_scaled, labels, "Hierarchical")

    # ── Dendrogram ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(
        Z,
        labels          = city_names,
        leaf_rotation   = 45,
        leaf_font_size  = 9,
        color_threshold = Z[-(KMEANS_K - 1), 2],   # colour at chosen cut
        ax              = ax,
    )
    ax.axhline(y=Z[-(KMEANS_K - 1), 2], color="#FF5722",
               linestyle="--", linewidth=1.5,
               label=f"Cut for {KMEANS_K} clusters")
    ax.set_title("Hierarchical Clustering Dendrogram  (Ward Linkage)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("City", fontsize=11)
    ax.set_ylabel("Linkage Distance", fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()

    out = os.path.join(PLOTS_DIR, "dendrogram.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [OK] Dendrogram saved -> {out}")

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 : DBSCAN CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def run_dbscan(X_scaled) -> np.ndarray:
    """
    Apply DBSCAN density-based clustering.

    Noise points are labelled -1 by sklearn.

    Parameters
    ----------
    X_scaled : ndarray  Standardised feature matrix

    Returns
    -------
    labels : ndarray (including -1 for noise)
    """
    _header(f"STEP 6 - DBSCAN CLUSTERING  "
            f"(eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_PTS})")

    t0 = time.perf_counter()
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_PTS, metric="euclidean")
    labels  = db.fit_predict(X_scaled)
    elapsed = time.perf_counter() - t0

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()

    print(f"  eps             : {DBSCAN_EPS}")
    print(f"  min_samples     : {DBSCAN_MIN_PTS}")
    print(f"  Clusters found  : {n_clusters}")
    print(f"  Noise points    : {n_noise}  (labelled -1)")
    print(f"  Wall-clock time : {elapsed:.4f} s")

    unique, counts = np.unique(labels, return_counts=True)
    print("\n  Cluster / noise breakdown:")
    for cl, cnt in zip(unique, counts):
        tag = "NOISE" if cl == -1 else f"Cluster {cl}"
        print(f"    {tag:12s}: {cnt} cities")

    _cluster_metrics(X_scaled, labels, "DBSCAN")

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 : 2D GEOGRAPHIC CLUSTER MAPS
# ─────────────────────────────────────────────────────────────────────────────

def plot_2d_maps(df: pd.DataFrame,
                 km_labels, hc_labels, db_labels) -> None:
    """
    Plot side-by-side 2D scatter maps (Longitude × Latitude) coloured
    by cluster assignment for all three algorithms.
    Noise points in DBSCAN are marked with 'x'.

    Saved to plots/cluster_maps_2d.png.
    """
    _header("STEP 7 - 2D GEOGRAPHIC CLUSTER MAPS")

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(
        "Urban Thermal Vulnerability Clusters  (Geographic View)",
        fontsize=15, fontweight="bold", y=1.01
    )

    configs = [
        ("k-Means",       km_labels),
        ("Hierarchical",  hc_labels),
        ("DBSCAN",        db_labels),
    ]

    lon = df["Longitude"].values
    lat = df["Latitude"].values
    temp = df["avg_temperature"].values

    for ax, (name, labels) in zip(axes, configs):
        unique_labels = sorted(set(labels))
        cmap = plt.get_cmap("tab10")

        for cl in unique_labels:
            mask = labels == cl
            if cl == -1:
                # Noise points
                ax.scatter(lon[mask], lat[mask],
                           c="black", marker="x", s=90,
                           linewidths=2, label="Noise", zorder=5)
            else:
                color = cmap(cl / max(1, len([l for l in unique_labels if l >= 0]) - 1))
                sc = ax.scatter(lon[mask], lat[mask],
                                c=[color]*mask.sum(),
                                s=120, edgecolors="white",
                                linewidths=0.8,
                                label=f"Cluster {cl}",
                                zorder=4)
                # City name annotations
                for i, (lo, la, city) in enumerate(
                    zip(lon[mask], lat[mask], df["City"].values[mask])
                ):
                    ax.annotate(city, (lo, la),
                                fontsize=5.5, ha="center", va="bottom",
                                xytext=(0, 4), textcoords="offset points",
                                color="dimgray")

        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Longitude", fontsize=10)
        ax.set_ylabel("Latitude",  fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(linestyle="--", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "cluster_maps_2d.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 : 3D SCATTER PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_3d_scatter(df: pd.DataFrame,
                    km_labels, hc_labels, db_labels) -> None:
    """
    Plot 3D scatter (Latitude, Longitude, avg_temperature) coloured by
    cluster for all three algorithms.

    Saved to plots/cluster_3d.png.
    """
    _header("STEP 8 - 3D CLUSTER SCATTER  (Lat × Lon × Avg Temperature)")

    fig = plt.figure(figsize=(20, 7))
    fig.suptitle(
        "3D Thermal Vulnerability Clusters  (Lat × Lon × Avg Temperature)",
        fontsize=14, fontweight="bold"
    )

    configs = [
        ("k-Means",       km_labels, 131),
        ("Hierarchical",  hc_labels, 132),
        ("DBSCAN",        db_labels, 133),
    ]

    lon  = df["Longitude"].values
    lat  = df["Latitude"].values
    temp = df["avg_temperature"].values

    for name, labels, subplot_id in configs:
        ax = fig.add_subplot(subplot_id, projection="3d")
        unique_labels = sorted(set(labels))
        cmap = plt.get_cmap("tab10")

        for cl in unique_labels:
            mask = labels == cl
            if cl == -1:
                ax.scatter(lon[mask], lat[mask], temp[mask],
                           c="black", marker="x", s=60,
                           linewidths=2, label="Noise")
            else:
                color = cmap(cl / max(1, len([l for l in unique_labels if l >= 0]) - 1))
                ax.scatter(lon[mask], lat[mask], temp[mask],
                           c=[color] * mask.sum(),
                           s=60, edgecolors="white",
                           linewidths=0.5,
                           label=f"Cluster {cl}",
                           alpha=0.85)

        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Lon",  fontsize=8, labelpad=4)
        ax.set_ylabel("Lat",  fontsize=8, labelpad=4)
        ax.set_zlabel("Avg T (K)", fontsize=8, labelpad=4)
        ax.legend(fontsize=7, loc="upper left")
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "cluster_3d.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 : CLUSTER PROFILING TABLE
# ─────────────────────────────────────────────────────────────────────────────

def profile_clusters(df: pd.DataFrame, km_labels) -> None:
    """
    Print mean feature values per k-Means cluster to characterise
    each thermal vulnerability region.
    """
    _header("STEP 9 - CLUSTER PROFILES  (k-Means  —  Mean Feature Values per Cluster)")

    df_copy = df.copy()
    df_copy["Cluster"] = km_labels
    profile = (
        df_copy
        .groupby("Cluster")[["avg_temperature", "avg_humidity",
                              "avg_pressure",   "avg_wind_speed",
                              "Latitude",        "Longitude"]]
        .mean()
        .round(3)
    )
    print(profile.to_string())

    print("\n  Cities per cluster:")
    for cl in sorted(df_copy["Cluster"].unique()):
        cities = df_copy[df_copy["Cluster"] == cl]["City"].tolist()
        print(f"    Cluster {cl}: {', '.join(cities)}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 : COMPARATIVE ANALYSIS — DBSCAN vs k-MEANS
# ─────────────────────────────────────────────────────────────────────────────

def print_comparative_analysis(df: pd.DataFrame,
                                km_labels, db_labels) -> None:
    """
    Print a structured textual analysis comparing how DBSCAN handles
    outliers vs k-Means.
    """
    _header("STEP 10 - COMPARATIVE ANALYSIS : DBSCAN vs k-MEANS")

    n_km_clusters = len(set(km_labels))
    n_db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise       = (db_labels == -1).sum()
    noise_cities  = df["City"].values[db_labels == -1].tolist()

    sep = "-" * 68
    print(f"""
  {sep}
  DBSCAN vs k-Means : Outlier Handling in Urban Thermal Clustering
  {sep}

  METRIC                    k-Means           DBSCAN
  {sep}
  Algorithm type            Centroid-based    Density-based
  Requires k upfront        Yes (k={KMEANS_K})       No
  Clusters found            {n_km_clusters:<18}{n_db_clusters}
  Outliers / noise          0 (none)          {n_noise}
  Handles noise natively    No                Yes (label = -1)
  Shape assumption          Convex / spherical Arbitrary
  {sep}

  Key Observations
  ----------------
  1. k-Means FORCES every city into a cluster.
     Even geographically isolated cities (e.g. Israeli cities in the
     Mediterranean) are assigned to the nearest centroid, potentially
     distorting cluster means and misrepresenting vulnerability zones.

  2. DBSCAN identifies low-density regions as NOISE (label = -1).
     Cities labelled as noise are genuine outliers — locations whose
     climate profile does not match any dense regional group.
     Noise cities in this dataset: {', '.join(noise_cities) if noise_cities else 'None'}

  3. For heatwave vulnerability mapping:
     - k-Means is useful for producing a fixed number of risk zones
       that align with administrative/policy boundaries.
     - DBSCAN is preferable for anomaly detection — flagging cities
       with unusual climate signatures that may need tailored
       heatwave response plans.

  4. Silhouette & Davies-Bouldin scores (printed above) quantify
     cluster quality; a higher silhouette + lower DB index indicates
     better-separated, more internally coherent risk zones.
  {sep}
""")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 11 : SAVE CLUSTER ASSIGNMENTS
# ─────────────────────────────────────────────────────────────────────────────

def save_cluster_assignments(df: pd.DataFrame,
                              km_labels, hc_labels, db_labels) -> None:
    """
    Persist city-level cluster assignments from all three algorithms
    to a single CSV.
    """
    _header("STEP 11 - SAVING CLUSTER ASSIGNMENTS")

    out = df[["City", "Country", "Latitude", "Longitude",
              "avg_temperature", "avg_humidity",
              "avg_pressure", "avg_wind_speed"]].copy()
    out["kmeans_cluster"]      = km_labels
    out["hierarchical_cluster"] = hc_labels
    out["dbscan_cluster"]      = db_labels   # -1 = noise

    out.to_csv(CLUSTER_OUT_CSV, index=False)
    print(f"  [OK] Saved -> {CLUSTER_OUT_CSV}")
    print(f"  Columns: {list(out.columns)}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    overall_start = time.perf_counter()

    # 1. Build per-city feature matrix
    df = build_city_features()

    # 2. Scale
    X_scaled, scaler = scale_features(df)

    city_names = df["City"].tolist()

    # 3. Elbow curve
    plot_elbow(X_scaled)

    # 4. k-Means
    km_labels = run_kmeans(X_scaled)

    # 5. Hierarchical + dendrogram
    hc_labels = run_hierarchical(X_scaled, city_names)

    # 6. DBSCAN
    db_labels = run_dbscan(X_scaled)

    # 7. 2D geographic maps
    plot_2d_maps(df, km_labels, hc_labels, db_labels)

    # 8. 3D scatter
    plot_3d_scatter(df, km_labels, hc_labels, db_labels)

    # 9. Cluster profiles
    profile_clusters(df, km_labels)

    # 10. Comparative analysis
    print_comparative_analysis(df, km_labels, db_labels)

    # 11. Save
    save_cluster_assignments(df, km_labels, hc_labels, db_labels)

    total = time.perf_counter() - overall_start
    _header("CLUSTERING COMPLETE")
    print(f"  Total pipeline time : {total:.2f} s")
    print(f"  Plots saved to      : {PLOTS_DIR}")
    print(f"  Output CSV          : {CLUSTER_OUT_CSV}")


if __name__ == "__main__":
    main()
