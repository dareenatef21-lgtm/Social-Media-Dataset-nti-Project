"""
config.py
=========
Single source of truth for which columns are used, how, and why.

WHY THIS FILE EXISTS
---------------------
Every script in this project (EDA, ensemble training, clustering, the final
pipeline, and the Streamlit app) imports these lists instead of re-typing
column names. This guarantees that:
  1. Training and deployment can never silently drift apart (the #1 cause
     of "works in the notebook, breaks in production" bugs).
  2. If you change a decision (e.g. add a feature), you change it in ONE
     place and every script picks it up automatically.

DATASET AUDIT SUMMARY (see src/01_eda.py for the full evidence)
------------------------------------------------------------------
- 7,000 rows, 25 columns, 0 fully-duplicate rows.
- Missing values: gender (1.0%), avg_sleep_hours (1.5%). Missingness rate
  does not shift the target distribution in either column -> safe to treat
  as random and impute simply.
- `wellbeing_band` (target) is NOT an independent outcome: a depth-3
  decision tree using ONLY anxiety_score_0to27 and low_mood_score_0to27
  reaches 100% test accuracy. The label is a deterministic severity-band
  formula built from those two columns (thresholds ~9.5 and ~16.5,
  mirroring standard clinical screening cutoffs). Using them as predictive
  features would be target leakage -> EXCLUDED from the supervised
  feature set.
- The same two columns are ALSO excluded from clustering. Not because of
  leakage (clustering has no train/test to leak across) but because
  including them would make the discovered clusters trivially mirror the
  three wellbeing_band buckets, which defeats the point of unsupervised
  discovery (finding NEW structure, not re-deriving a known formula).
- `participant_id` is a unique identifier with no predictive meaning ->
  excluded everywhere.
"""

TARGET = "wellbeing_band"
TARGET_CLASSES_ORDER = ["At-risk", "Moderate", "Good"]  # severity order, for readable plots

ID_COLUMN = "participant_id"

# Columns that directly define the target (see audit above). Excluded from
# BOTH the supervised feature set (leakage) and the clustering feature set
# (circularity).
LEAKAGE_COLUMNS = ["anxiety_score_0to27", "low_mood_score_0to27"]

# ---------------------------------------------------------------------------
# SUPERVISED LEARNING (ensemble models) feature groups
# ---------------------------------------------------------------------------

# Continuous / count numeric features. No missing values except avg_sleep_hours.
NUMERIC_FEATURES = [
    "age",
    "platforms_used_count",
    "daily_screen_hours",
    "daily_notifications",
    "minutes_to_first_check_after_waking",
    "avg_sleep_hours",
    "life_satisfaction_1to10",
    "loneliness_1to10",
    "self_esteem_1to10",
    "fomo_1to10",
    "social_comparison_1to10",
    "physical_activity_days_per_week",
]

# Categorical features that have a genuine, meaningful ORDER. Encoded with
# OrdinalEncoder (0,1,2,...) rather than one-hot, so the model can use
# "more/less" information instead of treating each level as unrelated.
ORDINAL_FEATURES = {
    "night_time_use": ["Never", "Sometimes", "Often", "Every night"],
    "attempted_digital_detox": ["No", "Yes, failed", "Yes, succeeded"],
    "seeks_mental_health_support": ["No", "Considering it", "Yes"],
    "uses_screen_time_limits": ["No", "Yes"],
}

# Categorical features with NO natural order -> one-hot encoded.
# gender has missing values, handled with its own imputer (see preprocessing.py).
NOMINAL_FEATURES = [
    "gender",
    "occupation",
    "region",
    "most_used_platform",
    "primary_purpose",
]

ALL_SUPERVISED_FEATURES = (
    NUMERIC_FEATURES + list(ORDINAL_FEATURES.keys()) + NOMINAL_FEATURES
)

# ---------------------------------------------------------------------------
# UNSUPERVISED LEARNING (clustering) feature group
# ---------------------------------------------------------------------------
# Deliberately NUMERIC-ONLY (plus one ordinal cast to an integer scale).
# WHY: K-Means measures Euclidean distance. One-hot dummy columns for
# high-cardinality nominal fields (region, platform, occupation - up to 8
# categories each) would each contribute a 0/1 spike that distorts distance
# in ways that don't correspond to real "similarity". Tree ensembles don't
# have this problem (they split one feature at a time), which is why the
# supervised preprocessing above is allowed to one-hot freely. Clustering
# preprocessing is intentionally different, for this reason.
CLUSTERING_NUMERIC_FEATURES = [
    "daily_screen_hours",
    "daily_notifications",
    "minutes_to_first_check_after_waking",
    "avg_sleep_hours",
    "platforms_used_count",
    "physical_activity_days_per_week",
    "life_satisfaction_1to10",
    "loneliness_1to10",
    "self_esteem_1to10",
    "fomo_1to10",
    "social_comparison_1to10",
]
# night_time_use has a real 0-3 order and is cheap to fold in as a number.
CLUSTERING_ORDINAL_FEATURE = "night_time_use"
CLUSTERING_ORDINAL_ORDER = ["Never", "Sometimes", "Often", "Every night"]

RANDOM_STATE = 42  # fixed seed used everywhere in this project, see README for why
