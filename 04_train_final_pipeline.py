"""
04_train_final_pipeline.py
============================
RUN FROM: the `src/` folder -> `python 04_train_final_pipeline.py`
Covers Part 15 of the assignment: builds and saves the ONE artifact the
Streamlit app depends on.

MODEL CHOICE: Random Forest (tuned)
-------------------------------------
From the Part 7 comparison table (see outputs/ensemble_comparison_with_tuned.csv):

    model                        accuracy  macro_precision  macro_recall  macro_f1  weighted_f1
    Decision Tree (baseline)       0.442        0.446          0.509       0.437       0.433
    Bagging                        0.550        0.516          0.416       0.420       0.508
    Random Forest                  0.547        0.526          0.483       0.498       0.541
    AdaBoost                       0.554        0.552          0.437       0.451       0.523
    Random Forest (tuned)          0.534        0.501          0.486       0.492       0.532

We do NOT pick the highest accuracy (that's AdaBoost/Bagging, both of
which achieve it by mostly predicting the majority "Moderate" class and
doing poorly on At-risk recall - see their confusion matrices). We pick
by MACRO-F1, since catching the minority At-risk class matters as much as
getting Moderate right. Untuned Random Forest (0.498) and tuned Random
Forest (0.492) are within noise of each other on this particular test
split (1,750 rows - a few dozen predictions is enough to move macro-F1 by
this much). We deploy the TUNED configuration because its hyperparameters
were chosen via proper 5-fold cross-validation (not by peeking at the test
set), which is the methodologically correct selection process, even
though the margin over the untuned default is not dramatic. Trade-offs
considered:
  - Interpretability: Random Forest still supports feature_importances_
    (see outputs/rf_feature_importance.png) - a single tree would be more
    interpretable but ensembles that fix its high variance are worth the
    small interpretability cost here.
  - Overfitting: min_samples_leaf/min_samples_split constraints found by
    tuning guard against individual trees memorizing rows.
  - Inference speed: 300 trees is a few milliseconds per prediction -
    fine for an interactive UI, no practical downside.
  - AdaBoost was rejected despite similar accuracy because its macro-F1
    (0.451) and At-risk recall are meaningfully weaker, and we found it
    unstable to imbalance-handling changes during development (see the
    comment in 02_ensemble_learning.py) - a red flag for a model going
    into a live tool.

Hyperparameters below come directly from the RandomizedSearchCV run in
02_ensemble_learning.py's Part 21 (best_params_, reproducible because
random_state=42 is fixed throughout the whole project). We hardcode them
here rather than re-running the 20-iteration search every time we want to
rebuild the deployment artifact - re-run 02_ensemble_learning.py yourself
at any point to re-derive them from scratch and confirm they match.
"""

import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from config import RANDOM_STATE
from preprocessing import load_raw_data, get_supervised_xy, build_supervised_preprocessor

DATA_PATH = "../data/social_media_screentime_mental_health_2026.csv"
MODEL_PATH = "../models/wellbeing_pipeline.joblib"

BEST_PARAMS = dict(
    n_estimators=300,
    min_samples_split=5,
    min_samples_leaf=1,
    max_features=0.5,
    max_depth=12,
)


def main():
    df = load_raw_data(DATA_PATH)
    X, y = get_supervised_xy(df)

    final_pipeline = Pipeline(steps=[
        ("preprocess", build_supervised_preprocessor()),
        ("model", RandomForestClassifier(
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            **BEST_PARAMS,
        )),
    ])

    # FINAL REFIT ON 100% OF THE DATA.
    # WHY this differs from 02_ensemble_learning.py (which fits on a 75%
    # train split and scores on the untouched 25%): that split's ONLY job
    # was to produce an honest, unbiased estimate of real-world
    # performance (the numbers quoted above). Once that evaluation is
    # done and reported, there's no more use for holding data back - more
    # training data almost always makes the final model at least as good,
    # so the artifact we actually ship is refit on every row we have.
    # This is standard practice, not a leakage shortcut: the reported
    # metrics were computed BEFORE this refit, on data this final model
    # will not be re-tested against.
    final_pipeline.fit(X, y)
    print(f"Final pipeline fit on all {len(X)} rows.")

    joblib.dump(final_pipeline, MODEL_PATH)
    print(f"Saved to {MODEL_PATH}")
    print(
        "\nWHAT IS SAVED: the ENTIRE Pipeline object - both the ColumnTransformer\n"
        "(imputers + ordinal encoder + one-hot encoder, each already fit with the\n"
        "medians/modes/categories learned from this data) AND the fitted Random\n"
        "Forest. This is one Python object, one file.\n\n"
        "WHY save the whole pipeline instead of just the model: if we only saved\n"
        "the RandomForestClassifier, the Streamlit app would need to manually\n"
        "re-implement every imputation/encoding decision by hand, in a second\n"
        "place, and any tiny mismatch (wrong median, different category order)\n"
        "would silently corrupt predictions. Saving the full Pipeline means the\n"
        "exact same fitted transformers run at inference time as ran during\n"
        "training - by construction, not by careful copy-pasting.\n\n"
        "HOW IT'S LOADED AND USED (see app.py):\n"
        "    pipeline = joblib.load('models/wellbeing_pipeline.joblib')\n"
        "    pipeline.predict(new_user_dataframe)          # -> class label\n"
        "    pipeline.predict_proba(new_user_dataframe)     # -> per-class probabilities\n"
        "`new_user_dataframe` must be a single-row (or multi-row) DataFrame with\n"
        "exactly the raw column names in config.ALL_SUPERVISED_FEATURES - the\n"
        "pipeline handles every transformation internally from there."
    )

    # Sanity check: reload from disk and confirm it predicts without error,
    # exactly the way the Streamlit app will use it.
    reloaded = joblib.load(MODEL_PATH)
    sample = X.iloc[[0]]
    pred = reloaded.predict(sample)
    proba = reloaded.predict_proba(sample)
    print(f"\nSanity check - reloaded pipeline predicts on row 0: {pred[0]}, "
          f"probabilities: {dict(zip(reloaded.classes_, proba[0].round(3)))}")


if __name__ == "__main__":
    main()
