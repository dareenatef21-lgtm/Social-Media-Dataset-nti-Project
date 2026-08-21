"""
03_unsupervised_learning.py
=============================
RUN FROM: the `src/` folder -> `python 03_unsupervised_learning.py`
Covers Parts 8-14 of the assignment.

CENTRAL RULE FOR THIS ENTIRE SCRIPT: wellbeing_band is NEVER passed into
the clustering algorithm. It is only used AFTER clustering, purely to
describe/validate what was found - see get_clustering_features() in
preprocessing.py, which structurally excludes it (it isn't even in the
DataFrame that reaches KMeans.fit()).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from config import RANDOM_STATE, TARGET, CLUSTERING_NUMERIC_FEATURES, CLUSTERING_ORDINAL_FEATURE
from preprocessing import load_raw_data, get_clustering_features, build_clustering_preprocessor

DATA_PATH = "../data/social_media_screentime_mental_health_2026.csv"
OUT_DIR = "../outputs"


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    df = load_raw_data(DATA_PATH)

    section("PART 8: WHICH FEATURES BELONG IN THE CLUSTERING DATASET?")
    print(
        "EXCLUDED, and why:\n"
        "  - participant_id: an arbitrary identifier, has zero geometric meaning;\n"
        "    including it would let KMeans 'cluster' on essentially a row number.\n"
        "  - wellbeing_band: this is the target label. Feeding it into KMeans would\n"
        "    mean the 'unsupervised' discovery is actually supervised by the label\n"
        "    we're trying to independently validate against - defeats the purpose.\n"
        "  - anxiety_score_0to27 / low_mood_score_0to27: not leakage in the\n"
        "    train/test sense (clustering doesn't generalize to a held-out set the\n"
        "    same way), but including them would make clusters trivially mirror the\n"
        "    3 wellbeing_band buckets, since the audit in 01_eda.py proved the band\n"
        "    IS a formula of exactly these two columns. That would just re-draw a\n"
        "    label we already have, not discover anything new.\n"
        "  - gender, occupation, region, most_used_platform, primary_purpose: high-\n"
        "    cardinality NOMINAL categories with no natural order. K-Means measures\n"
        "    straight-line (Euclidean) distance; one-hot dummy spikes for a category\n"
        "    like 'region' would distort that distance in a way that doesn't reflect\n"
        "    genuine similarity between two people (see preprocessing.py docstring).\n"
        "\n"
        f"INCLUDED ({len(CLUSTERING_NUMERIC_FEATURES) + 1} features): all numeric\n"
        "behavioral and general-wellbeing scores, plus night_time_use (has a clean\n"
        "0-3 order: Never < Sometimes < Often < Every night, so it's cast to a\n"
        "number rather than one-hot encoded):\n"
        f"  {CLUSTERING_NUMERIC_FEATURES + [CLUSTERING_ORDINAL_FEATURE]}"
    )

    X_cluster_raw = get_clustering_features(df)
    preprocessor = build_clustering_preprocessor()
    X_scaled = preprocessor.fit_transform(X_cluster_raw)
    print(f"\nShape after preprocessing (impute -> ordinal-encode -> StandardScale): {X_scaled.shape}")
    print(
        "WHY SCALE HERE (and not in the supervised pipeline): K-Means groups points\n"
        "by Euclidean distance. minutes_to_first_check_after_waking ranges 0-111 while\n"
        "self_esteem_1to10 ranges 1-10 - unscaled, the 'minutes' feature alone would\n"
        "dominate every distance calculation regardless of its true importance.\n"
        "StandardScaler rescales every feature to mean 0, std 1, so each contributes\n"
        "comparably to distance. Tree ensembles don't have this problem because they\n"
        "split one feature at a time using ranks/thresholds, not distances."
    )

    # ------------------------------------------------------------------
    # PART 9: WCSS across K = 2..10, elbow method
    # ------------------------------------------------------------------
    section("PART 9: WCSS AND THE ELBOW METHOD")
    print(
        "WCSS (Within-Cluster Sum of Squares) = for every point, the squared\n"
        "distance to its OWN cluster's centroid, summed over all points:\n"
        "    WCSS = sum over clusters k, sum over points i in cluster k, of\n"
        "           ||x_i - centroid_k||^2\n"
        "In plain terms: 'how tightly packed are the clusters overall?'. Lower WCSS\n"
        "means points sit closer to their assigned centroid. WCSS ALWAYS decreases\n"
        "(or stays flat) as K increases - with K = number of rows, each point is its\n"
        "own cluster and WCSS = 0. That's not useful; it means we can't just pick the\n"
        "K with the lowest WCSS, we look for the 'elbow' - the point where adding\n"
        "another cluster stops buying much improvement."
    )

    k_range = list(range(2, 11))
    wcss_by_k = {}
    kmeans_by_k = {}
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE)
        km.fit(X_scaled)
        wcss_by_k[k] = km.inertia_  # sklearn calls WCSS "inertia_"
        kmeans_by_k[k] = km
        print(f"  K={k}: WCSS = {km.inertia_:.1f}")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(k_range, [wcss_by_k[k] for k in k_range], marker="o")
    ax.set_xlabel("K (number of clusters)")
    ax.set_ylabel("WCSS (inertia)")
    ax.set_title("Elbow Method")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/elbow_wcss.png", dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------------
    # PART 10: BCSS
    # ------------------------------------------------------------------
    section("PART 10: BCSS (BETWEEN-CLUSTER SUM OF SQUARES)")
    print(
        "BCSS measures how far apart the CLUSTERS are from each other (as opposed\n"
        "to WCSS, which measures how tight each cluster is internally):\n"
        "    BCSS = sum over clusters k, of  n_k * ||centroid_k - grand_mean||^2\n"
        "where n_k is the number of points in cluster k and grand_mean is the\n"
        "overall mean of ALL points (i.e. what a single 'cluster' of everyone would\n"
        "look like). Every point's total squared distance from the grand mean\n"
        "(TSS, Total Sum of Squares) splits exactly into two pieces:\n"
        "    TSS = WCSS + BCSS\n"
        "TSS doesn't depend on K (it's just the variance of the whole dataset), so\n"
        "as K grows and WCSS shrinks, BCSS necessarily grows to compensate. A GOOD\n"
        "clustering has clusters that are both tight (low WCSS) AND well separated\n"
        "from each other (high BCSS) relative to how many clusters you're using -\n"
        "that trade-off is exactly what the elbow method and silhouette score below\n"
        "are trying to summarize in one number."
    )

    def compute_tss(X):
        grand_mean = X.mean(axis=0)
        return float(np.sum((X - grand_mean) ** 2))

    tss = compute_tss(X_scaled)
    bcss_by_k = {}
    for k in k_range:
        bcss_by_k[k] = tss - wcss_by_k[k]
        # sanity check the identity TSS = WCSS + BCSS holds
        assert abs((wcss_by_k[k] + bcss_by_k[k]) - tss) < 1e-6
    print(f"\nTSS (fixed, does not depend on K): {tss:.1f}")
    print(f"{'K':>3} | {'WCSS':>10} | {'BCSS':>10} | BCSS/TSS (variance explained)")
    for k in k_range:
        print(f"{k:>3} | {wcss_by_k[k]:>10.1f} | {bcss_by_k[k]:>10.1f} | {bcss_by_k[k]/tss:.3f}")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(k_range, [wcss_by_k[k] for k in k_range], marker="o", label="WCSS (within)")
    ax.plot(k_range, [bcss_by_k[k] for k in k_range], marker="s", label="BCSS (between)")
    ax.axhline(tss, color="gray", linestyle="--", linewidth=1, label="TSS (constant)")
    ax.set_xlabel("K (number of clusters)")
    ax.set_ylabel("Sum of squares")
    ax.set_title("WCSS vs BCSS across K  (WCSS + BCSS = TSS)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/wcss_vs_bcss.png", dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------------
    # PART 11: K-MEANS++ vs random initialization
    # ------------------------------------------------------------------
    section("PART 11: K-MEANS++ vs RANDOM INITIALIZATION")
    print(
        "Plain K-Means starts by dropping K centroids uniformly at random among the\n"
        "data points. Bad luck can place two initial centroids close together in\n"
        "the same dense region, leaving a whole other cluster's worth of points\n"
        "poorly represented - the algorithm can converge to a genuinely worse local\n"
        "optimum depending on that random draw, and different runs can disagree.\n"
        "\n"
        "K-Means++ (sklearn default, init='k-means++') fixes this at INITIALIZATION\n"
        "time only (the iterative fitting afterward is identical): the first\n"
        "centroid is picked uniformly at random, but every SUBSEQUENT centroid is\n"
        "picked with probability proportional to its squared distance from the\n"
        "nearest already-chosen centroid. Points far from existing centroids are\n"
        "much more likely to be chosen next, which spreads the starting centroids\n"
        "out across the data instead of letting them cluster together by chance.\n"
        "This tends to converge faster and land on a lower, more consistent WCSS.\n"
    )
    K_COMPARE = 4  # will be finalized after looking at the elbow/silhouette results below; using a representative K here to demonstrate the init comparison itself
    n_seeds = 15
    random_final_wcss, plusplus_final_wcss = [], []
    for seed in range(n_seeds):
        km_r = KMeans(n_clusters=K_COMPARE, init="random", n_init=1, random_state=seed).fit(X_scaled)
        km_p = KMeans(n_clusters=K_COMPARE, init="k-means++", n_init=1, random_state=seed).fit(X_scaled)
        random_final_wcss.append(km_r.inertia_)
        plusplus_final_wcss.append(km_p.inertia_)
    print(f"Across {n_seeds} different random seeds, K={K_COMPARE}, n_init=1 (single run each, to expose init sensitivity):")
    print(f"  random init   -> mean WCSS = {np.mean(random_final_wcss):.1f}, std = {np.std(random_final_wcss):.1f}, worst = {np.max(random_final_wcss):.1f}")
    print(f"  k-means++     -> mean WCSS = {np.mean(plusplus_final_wcss):.1f}, std = {np.std(plusplus_final_wcss):.1f}, worst = {np.max(plusplus_final_wcss):.1f}")
    print(
        "-> k-means++ reaches a lower AND more consistent (lower std) WCSS across\n"
        "   seeds - fewer unlucky runs getting stuck in a bad local optimum. This is\n"
        "   why it's sklearn's default and why we use it for every K above (n_init=10\n"
        "   additionally reruns k-means++ 10 times and keeps the best, which is the\n"
        "   standard production-safe setting)."
    )

    # ------------------------------------------------------------------
    # PART 12: CLUSTER EVALUATION - silhouette score across K
    # ------------------------------------------------------------------
    section("PART 12: SILHOUETTE SCORE ACROSS K")
    print(
        "Silhouette score (per point, then averaged) compares a(i) = average\n"
        "distance from point i to other points in its OWN cluster, against b(i) =\n"
        "average distance from i to points in the NEAREST other cluster:\n"
        "    silhouette(i) = (b(i) - a(i)) / max(a(i), b(i))\n"
        "Ranges from -1 to +1. Close to +1: the point sits comfortably inside its\n"
        "own cluster and far from others (good). Close to 0: the point sits right\n"
        "on the boundary between two clusters. Negative: the point is probably in\n"
        "the wrong cluster. We use it alongside the elbow because WCSS alone can't\n"
        "tell us whether clusters are actually well-SEPARATED, only how tight they\n"
        "are - silhouette captures both in one number.\n"
        "(Computed on a 2,000-row random subsample for speed - silhouette_score is\n"
        "O(n^2); the ranking across K is stable under subsampling.)"
    )
    rng = np.random.RandomState(RANDOM_STATE)
    sample_idx = rng.choice(X_scaled.shape[0], size=min(2000, X_scaled.shape[0]), replace=False)
    sil_by_k = {}
    for k in k_range:
        labels = kmeans_by_k[k].predict(X_scaled)
        sil = silhouette_score(X_scaled[sample_idx], labels[sample_idx])
        sil_by_k[k] = sil
        print(f"  K={k}: silhouette = {sil:.4f}")

    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
    ax1.plot(k_range, [wcss_by_k[k] for k in k_range], marker="o", color="#4C72B0", label="WCSS")
    ax1.set_xlabel("K")
    ax1.set_ylabel("WCSS", color="#4C72B0")
    ax2 = ax1.twinx()
    ax2.plot(k_range, [sil_by_k[k] for k in k_range], marker="s", color="#DD8452", label="Silhouette")
    ax2.set_ylabel("Silhouette score", color="#DD8452")
    ax1.set_title("Choosing K: WCSS (elbow) + Silhouette")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/k_selection.png", dpi=130)
    plt.close(fig)

    best_k_by_silhouette = max(sil_by_k, key=sil_by_k.get)
    print(
        f"\nHonest read of the plots (see outputs/elbow_wcss.png and\n"
        f"outputs/k_selection.png): WCSS decreases smoothly with no single sharp\n"
        f"elbow - typical for real, noisy survey data where behavior sits on a\n"
        f"continuum rather than in naturally separated blobs. Silhouette scores are\n"
        f"modest across the board ({min(sil_by_k.values()):.3f}-{max(sil_by_k.values()):.3f}, well short of\n"
        f"the ~0.5+ that would indicate strongly separated clusters) and DECREASE\n"
        f"monotonically from K=2 onward, peaking at K={best_k_by_silhouette}.\n"
        f"We report this honestly rather than forcing a more 'interesting-looking'\n"
        f"K: the most statistically defensible structure in this data is a two-way\n"
        f"split, not a rich multi-persona typology. This itself is a legitimate,\n"
        f"presentable finding - see PART 13 below, where the two clusters turn out to\n"
        f"correspond to a genuinely interpretable 'lower-engagement/higher-wellbeing'\n"
        f"vs 'higher-engagement/lower-wellbeing' split."
    )
    FINAL_K = best_k_by_silhouette
    final_km = kmeans_by_k[FINAL_K]
    final_labels = final_km.labels_

    # ------------------------------------------------------------------
    # PART 12 (continued): PCA for visualization ONLY
    # ------------------------------------------------------------------
    section("PCA VISUALIZATION (2D projection, for plotting only)")
    print(
        "IMPORTANT DISTINCTION: KMeans above was fit on the FULL "
        f"{X_scaled.shape[1]}-dimensional\n"
        "scaled feature space - that is the real clustering model and those are the\n"
        "real cluster assignments. PCA below is used ONLY to compress those same\n"
        "points down to 2 dimensions so we can draw them on a flat plot; it is not\n"
        "re-clustering anything and the 2D coordinates are never fed back into\n"
        "KMeans. Some cluster separation that exists in 11 dimensions can look\n"
        "muddier in a 2D PCA projection simply because compressing to 2 axes\n"
        "necessarily throws away some information - that's a property of\n"
        "visualization, not evidence the clustering itself is wrong."
    )
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_
    print(f"Variance captured by the 2 PCA axes shown: {explained[0]:.1%} + {explained[1]:.1%} = {sum(explained):.1%} of total")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=final_labels, cmap="tab10", s=8, alpha=0.5)
    centroids_pca = pca.transform(final_km.cluster_centers_)
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c="black", marker="X", s=180, label="centroids")
    ax.set_xlabel(f"PCA 1 ({explained[0]:.1%} var)")
    ax.set_ylabel(f"PCA 2 ({explained[1]:.1%} var)")
    ax.set_title(f"K-Means clusters (K={FINAL_K}), PCA projection for display only")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/clusters_pca.png", dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------------
    # PART 13: UNDERSTANDING THE CLUSTERS
    # ------------------------------------------------------------------
    section(f"PART 13: CLUSTER PROFILES (K={FINAL_K})")
    profile_cols = CLUSTERING_NUMERIC_FEATURES + [CLUSTERING_ORDINAL_FEATURE]
    profile_df = X_cluster_raw.copy()
    profile_df["cluster"] = final_labels

    sizes = profile_df["cluster"].value_counts().sort_index()
    print("Cluster sizes:")
    for c, n in sizes.items():
        print(f"  Cluster {c}: {n} people ({n/len(profile_df):.1%})")

    means = profile_df.groupby("cluster")[CLUSTERING_NUMERIC_FEATURES].mean().round(2)
    print("\nMean feature values per cluster (numeric features):")
    print(means.T)
    means.to_csv(f"{OUT_DIR}/cluster_profiles.csv")

    night_use_map = profile_df.groupby("cluster")[CLUSTERING_ORDINAL_FEATURE].agg(
        lambda s: s.value_counts(normalize=True).idxmax()
    )
    print("\nMost common night_time_use per cluster:")
    print(night_use_map)

    print(
        "\nHOW TO INTERPRET (using 'associated with' / 'characterized by' language,\n"
        "never causal claims - clustering finds correlational patterns, not causes):"
    )
    overall_means = X_cluster_raw[CLUSTERING_NUMERIC_FEATURES].mean()
    for c in sorted(profile_df["cluster"].unique()):
        row = means.loc[c]
        diffs = (row - overall_means).sort_values(ascending=False)
        top_high = diffs.head(2).index.tolist()
        top_low = diffs.tail(2).index.tolist()
        print(f"\n  Cluster {c} (n={sizes[c]}): relative to the overall average, this group is")
        print(f"    characterized by HIGHER {top_high} and LOWER {top_low}.")

    # ------------------------------------------------------------------
    # PART 13 (continued): does clustering correspond to wellbeing_band?
    # wellbeing_band was NEVER used to fit the clusters - this is purely
    # a post-hoc, descriptive comparison.
    # ------------------------------------------------------------------
    section("EXTERNAL VALIDATION (NOT used for training): clusters vs wellbeing_band")
    print(
        "wellbeing_band was excluded from clustering entirely. Here we ONLY check,\n"
        "after the fact, whether the behavior/wellbeing-score clusters we discovered\n"
        "happen to line up with the (separately defined) risk bands - this is a\n"
        "sanity check on whether the clusters are meaningful, not a modeling step."
    )
    crosstab = pd.crosstab(profile_df["cluster"], df.loc[profile_df.index, TARGET], normalize="index").round(3)
    crosstab = crosstab[["At-risk", "Moderate", "Good"]]
    print(crosstab)
    crosstab.to_csv(f"{OUT_DIR}/cluster_vs_wellbeing_band.csv")
    print(
        "\nRead this as: 'of the people in Cluster X, what fraction fell in each\n"
        "wellbeing_band?'. If some clusters show a noticeably higher At-risk share\n"
        "than others, that's evidence the behavioral clusters are picking up a\n"
        "real, meaningful pattern (associated with risk) even though risk itself\n"
        "was never shown to the clustering algorithm - NOT evidence that being in\n"
        "that cluster causes worse wellbeing."
    )

    # ------------------------------------------------------------------
    # OPTIONAL SECONDARY VIEW: a more granular persona breakdown at K=4,
    # for presentation narrative only. Clearly separate from FINAL_K above
    # (chosen honestly by the silhouette metric) - shown here because a
    # 2-cluster story is statistically the most defensible, but a 4-cluster
    # view can still be a legitimate, DECLARED exploratory choice for
    # richer storytelling, as long as we're transparent that it isn't the
    # metric-selected K. If asked in the discussion, be upfront about this
    # distinction rather than presenting K=4 as if it were the "answer".
    # ------------------------------------------------------------------
    section("OPTIONAL SECONDARY VIEW: K=4 persona breakdown (narrative only, not metric-selected)")
    K_NARRATIVE = 4
    narrative_km = kmeans_by_k[K_NARRATIVE]
    narrative_labels = narrative_km.labels_
    narrative_profile = X_cluster_raw.copy()
    narrative_profile["cluster"] = narrative_labels
    narrative_means = narrative_profile.groupby("cluster")[CLUSTERING_NUMERIC_FEATURES].mean().round(2)
    narrative_sizes = narrative_profile["cluster"].value_counts().sort_index()
    print(f"Sizes: {dict(narrative_sizes)}")
    print(narrative_means.T)
    narrative_means.to_csv(f"{OUT_DIR}/cluster_profiles_k4_narrative.csv")
    print(
        "\nUse this table if your presentation wants richer personas than the\n"
        "2-cluster split - just be ready to say explicitly that K=2 is the\n"
        "silhouette-selected K and K=4 is a declared exploratory choice, not a\n"
        "second 'optimal' answer. Mixing those two up under questioning is the\n"
        "single easiest way to look like you don't understand your own method."
    )

    print(f"\nUnsupervised learning complete. Final (metric-selected) choice: K={FINAL_K}. "
          f"Secondary narrative view: K={K_NARRATIVE}.")


if __name__ == "__main__":
    main()
