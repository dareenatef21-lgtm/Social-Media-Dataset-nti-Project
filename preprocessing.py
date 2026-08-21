"""
preprocessing.py
=================
Builds the preprocessing objects used by every other script.

WHY Pipeline + ColumnTransformer (instead of manually transforming the
DataFrame with pandas)?
------------------------------------------------------------------------
1. LEAKAGE PREVENTION: A Pipeline's `.fit()` is only ever called on the
   training fold. When used inside cross_val_score / GridSearchCV, sklearn
   automatically refits the imputer/encoder/scaler on each training fold
   and only *applies* (never refits) them to the held-out fold. If you
   instead ran e.g. `SimpleImputer().fit_transform(whole_dataframe)` before
   splitting, the median/mode used for imputation would have "seen" the
   test rows -> preprocessing leakage. This is why we split BEFORE fitting
   anything (see 02_ensemble_learning.py).
2. REPRODUCIBILITY: the exact same object that was fit on training data is
   reused (not rebuilt) at inference time, so training and serving can
   never drift apart.
3. DEPLOYMENT: a Pipeline that ends in a classifier can be pickled/joblib'd
   as ONE artifact. The Streamlit app never needs to know that gender gets
   one-hot encoded or that avg_sleep_hours gets median-imputed - it just
   calls `.predict()` on a raw DataFrame.

WHY SOME COLUMNS ARE SCALED AND OTHERS AREN'T
------------------------------------------------------------------------
- Supervised models here (Bagging, Random Forest, AdaBoost) are all
  TREE-BASED. Trees split on "is feature X <= threshold?" one feature at a
  time, so multiplying a feature by 1000 or by 0.001 does not change where
  the optimal split falls. Tree ensembles are scale-invariant -> no
  StandardScaler needed in the supervised pipeline. (If we later added a
  distance-based or gradient-based model, e.g. Logistic Regression or SVM,
  we WOULD need to scale.)
- K-Means (clustering) is DISTANCE-based. A feature measured in "minutes"
  (0-111) would dominate Euclidean distance over a feature measured in
  "1-10" purely because of its numeric range, not because it is actually
  more important. So the clustering pipeline scales every feature with
  StandardScaler (mean 0, std 1) before clustering.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from config import (
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    NOMINAL_FEATURES,
    CLUSTERING_NUMERIC_FEATURES,
    CLUSTERING_ORDINAL_FEATURE,
    CLUSTERING_ORDINAL_ORDER,
    ID_COLUMN,
    LEAKAGE_COLUMNS,
    TARGET,
)


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """Load the CSV exactly as provided. No cleaning happens here on
    purpose - cleaning/imputation belongs inside the Pipeline so it is
    always fit only on training data, never on the full dataset up front.
    """
    df = pd.read_csv(csv_path)
    return df


def get_supervised_xy(df: pd.DataFrame):
    """Split into the legitimate feature matrix X and target y.

    Drops participant_id (identifier, not predictive) and the two
    leakage columns (anxiety_score_0to27, low_mood_score_0to27) - see
    config.py docstring for the audit that justifies this.
    """
    feature_cols = NUMERIC_FEATURES + list(ORDINAL_FEATURES.keys()) + NOMINAL_FEATURES
    X = df[feature_cols].copy()
    y = df[TARGET].copy()
    return X, y


def build_supervised_preprocessor() -> ColumnTransformer:
    """ColumnTransformer for the ensemble classifiers.

    - numeric: median-impute (robust to the skew in daily_notifications,
      no scaling - trees don't need it, see module docstring).
    - ordinal: explicit category order from config.ORDINAL_FEATURES so
      "Never < Sometimes < Often < Every night" is preserved as 0,1,2,3.
    - gender: its own branch. ~1% of rows are missing gender. We impute
      with the constant 'Unknown' rather than the most frequent category
      (Male), because a true missing response is a different signal than
      someone who actively selected the most common answer - collapsing
      the two would quietly bias the encoding.
    - other nominal (occupation, region, most_used_platform,
      primary_purpose): most_frequent imputation (defensive - training
      data has 0 missing here, but this makes the pipeline robust if a
      real deployment ever receives a blank field) then one-hot encoding.
      handle_unknown='ignore' means a brand-new category value at
      inference time (e.g. a platform that didn't exist in training) is
      encoded as all-zeros instead of crashing the app.
    """
    numeric_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
    ])

    ordinal_cols = list(ORDINAL_FEATURES.keys())
    ordinal_categories = [ORDINAL_FEATURES[c] for c in ordinal_cols]
    ordinal_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(categories=ordinal_categories)),
    ])

    gender_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])

    other_nominal_cols = [c for c in NOMINAL_FEATURES if c != "gender"]
    nominal_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, NUMERIC_FEATURES),
            ("ordinal", ordinal_pipe, ordinal_cols),
            ("gender", gender_pipe, ["gender"]),
            ("nominal", nominal_pipe, other_nominal_cols),
        ],
        remainder="drop",  # anything not listed (e.g. leakage cols, id) is dropped even if present
    )
    return preprocessor


def get_clustering_features(df: pd.DataFrame) -> pd.DataFrame:
    """Builds the clustering feature frame.

    Deliberately excludes: wellbeing_band (target - clustering must never
    see it, or "unsupervised" stops being true), anxiety_score_0to27 and
    low_mood_score_0to27 (would make clusters circularly mirror the
    target), participant_id (identifier), and every high-cardinality
    nominal column (gender, occupation, region, platform, purpose) for the
    distance-metric reason explained in preprocessing.py's module
    docstring.
    """
    out = df[CLUSTERING_NUMERIC_FEATURES].copy()
    out[CLUSTERING_ORDINAL_FEATURE] = df[CLUSTERING_ORDINAL_FEATURE]
    return out


def build_clustering_preprocessor() -> ColumnTransformer:
    """Preprocessing for K-Means: median-impute, ordinal-encode
    night_time_use, then StandardScale everything (required - K-Means
    uses Euclidean distance, see module docstring).
    """
    numeric_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
    ])
    ordinal_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(categories=[CLUSTERING_ORDINAL_ORDER])),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, CLUSTERING_NUMERIC_FEATURES),
            ("ordinal", ordinal_pipe, [CLUSTERING_ORDINAL_FEATURE]),
        ]
    )
    # Scale AFTER imputing/encoding, applied to the full transformed matrix.
    full_pipe = Pipeline(steps=[
        ("pre", pre),
        ("scale", StandardScaler()),
    ])
    return full_pipe


def get_feature_names_out(preprocessor: ColumnTransformer):
    """Recovers human-readable feature names after the ColumnTransformer
    has been fit, so Random Forest feature_importances_ can be labeled
    correctly instead of showing 'feature_37'. Requires scikit-learn
    >= 1.0 (ColumnTransformer.get_feature_names_out).
    """
    return preprocessor.get_feature_names_out()
