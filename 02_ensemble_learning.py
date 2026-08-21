"""
02_ensemble_learning.py
========================
RUN FROM: the `src/` folder -> `python 02_ensemble_learning.py`

Covers Parts 3-7 and 21 of the assignment: Bagging, Random Forest,
AdaBoost, a fair multiclass comparison between them, and light
hyperparameter tuning done the leakage-safe way (the Pipeline is the
thing that gets cross-validated, not just the classifier).

WHY A PLAIN DECISION TREE IS INCLUDED TOO
------------------------------------------
It's not one of your 4 required models, but every ensemble here is BUILT
FROM decision trees. Training one plain tree first gives a concrete
"before" picture: how much does bagging/boosting actually help versus a
single tree on THIS dataset? Without that baseline, "Random Forest got
0.56 macro-F1" is a number with no meaning.

METRICS
-------
Because wellbeing_band has 3 imbalanced classes (Moderate 53.8%, Good
32.3%, At-risk 13.9%), plain accuracy is misleading - a model that always
predicts "Moderate" scores ~53.8% accuracy while being useless. We report:
  - accuracy (for reference only)
  - precision/recall/F1 with macro averaging (unweighted mean across the
    3 classes - treats At-risk as equally important as Moderate, which
    matches the real goal: catching at-risk people matters even though
    they're the minority class)
  - weighted F1 (accounts for class size - included for completeness)
  - full classification_report (per-class breakdown)
  - confusion matrix
"""

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, f1_score,
)

from config import RANDOM_STATE, TARGET_CLASSES_ORDER
from preprocessing import load_raw_data, get_supervised_xy, build_supervised_preprocessor, get_feature_names_out

DATA_PATH = "../data/social_media_screentime_mental_health_2026.csv"
OUT_DIR = "../outputs"


def evaluate(name, pipeline, X_test, y_test, results_list):
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, preds, average="macro", zero_division=0
    )
    f1_weighted = f1_score(y_test, preds, average="weighted")

    print(f"\n{'-'*60}\n{name}\n{'-'*60}")
    print(f"Accuracy:        {acc:.4f}")
    print(f"Macro Precision: {prec_macro:.4f}")
    print(f"Macro Recall:    {rec_macro:.4f}")
    print(f"Macro F1:        {f1_macro:.4f}")
    print(f"Weighted F1:     {f1_weighted:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, preds, zero_division=0))
    cm = confusion_matrix(y_test, preds, labels=TARGET_CLASSES_ORDER)
    print("Confusion matrix (rows=actual, cols=predicted), order =", TARGET_CLASSES_ORDER)
    print(cm)

    results_list.append({
        "model": name, "accuracy": acc, "macro_precision": prec_macro,
        "macro_recall": rec_macro, "macro_f1": f1_macro, "weighted_f1": f1_weighted,
    })
    return cm


def plot_confusion(cm, name, filename):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(TARGET_CLASSES_ORDER)))
    ax.set_yticks(range(len(TARGET_CLASSES_ORDER)))
    ax.set_xticklabels(TARGET_CLASSES_ORDER)
    ax.set_yticklabels(TARGET_CLASSES_ORDER)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(name)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{filename}", dpi=130)
    plt.close(fig)


def main():
    df = load_raw_data(DATA_PATH)
    X, y = get_supervised_xy(df)

    # Split BEFORE any fitting. The preprocessor inside each Pipeline below
    # is only ever .fit() on X_train - it never sees X_test until .transform()
    # at prediction time. This is what "no preprocessing leakage" means in
    # practice, not just in theory.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train rows: {len(X_train)}   Test rows: {len(X_test)}")
    print("random_state=42 is used everywhere in this project so that anyone "
          "re-running this script gets IDENTICAL splits, bootstrap samples, and "
          "results - required for the 'reproducible' requirement.")

    results = []
    cms = {}

    # ------------------------------------------------------------------
    # BASELINE: single Decision Tree
    # ------------------------------------------------------------------
    tree_pipe = Pipeline(steps=[
        ("preprocess", build_supervised_preprocessor()),
        ("model", DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    tree_pipe.fit(X_train, y_train)
    cms["Decision Tree (baseline)"] = evaluate("Decision Tree (baseline, not an ensemble)", tree_pipe, X_test, y_test, results)

    # ------------------------------------------------------------------
    # A. BAGGING
    # ------------------------------------------------------------------
    # Bagging = Bootstrap AGGregatING. Train many copies of the SAME base
    # learner (here, a decision tree) on different bootstrap samples
    # (random samples of the training rows, drawn WITH replacement, each
    # the same size as the original training set) and average their votes.
    # Each tree sees a slightly different subset of rows, so each tree
    # makes different mistakes; averaging cancels out a lot of that noise.
    # This targets VARIANCE: a single unpruned tree memorizes its training
    # rows (low bias, high variance / overfits). Bagging keeps the low
    # bias but drives variance down because the errors of different trees
    # are only weakly correlated.
    bagging_pipe = Pipeline(steps=[
        ("preprocess", build_supervised_preprocessor()),
        ("model", BaggingClassifier(
            estimator=DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            n_estimators=200,      # more trees = lower variance, with diminishing returns; 200 is a solid, still-fast default
            max_samples=1.0,       # each bootstrap sample is the same size as the training set (the standard bagging definition)
            max_features=1.0,      # each tree sees ALL features (this is what separates plain Bagging from Random Forest)
            bootstrap=True,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])
    t0 = time.time()
    bagging_pipe.fit(X_train, y_train)
    bagging_time = time.time() - t0
    cms["Bagging"] = evaluate("A. Bagging (200 trees, all features)", bagging_pipe, X_test, y_test, results)

    # ------------------------------------------------------------------
    # B. RANDOM FOREST
    # ------------------------------------------------------------------
    # Random Forest = Bagging + one extra trick: at EACH SPLIT inside each
    # tree, only a random subset of features is even considered (default
    # sqrt(n_features) for classification). This decorrelates the trees
    # further than plain bagging does - without it, if one feature
    # (e.g. daily_screen_hours) is clearly the strongest predictor, almost
    # every bagged tree would pick it for its very first split, making the
    # trees more similar to each other than we'd like. Forcing each split
    # to ignore most features most of the time means different trees pick
    # up on different secondary patterns, which is what actually improves
    # the ensemble average.
    rf_pipe = Pipeline(steps=[
        ("preprocess", build_supervised_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,           # let trees grow fully; the ensembling controls overfitting, not tree depth
            min_samples_leaf=3,
            max_features="sqrt",      # the "extra randomness" step described above
            class_weight="balanced",  # up-weights the minority At-risk class during training
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])
    t0 = time.time()
    rf_pipe.fit(X_train, y_train)
    rf_time = time.time() - t0
    cms["Random Forest"] = evaluate("B. Random Forest (300 trees)", rf_pipe, X_test, y_test, results)

    # ------------------------------------------------------------------
    # D. ADABOOST (covering Part C "Boosting" conceptually + Part D)
    # ------------------------------------------------------------------
    # AdaBoost = ADAptive BOOSTing. Unlike bagging (trees trained
    # independently, in parallel, on random row subsets), boosting trains
    # trees SEQUENTIALLY. Every training row starts with equal weight.
    # After each weak learner (here: a depth-1 "decision stump") is
    # trained, rows it got WRONG have their weight increased and rows it
    # got RIGHT have their weight decreased, so the next stump is forced
    # to focus on the previously-hard cases. Final prediction is a
    # weighted vote across all stumps, where more accurate stumps get more
    # say. This targets BIAS: a single decision stump is far too simple
    # (high bias, underfits) - boosting incrementally patches its
    # blind spots.
    # IMPORTANT scikit-learn version note: AdaBoostClassifier's
    # `algorithm` parameter (SAMME vs SAMME.R) is deprecated as of
    # scikit-learn 1.4 and REMOVED in 1.6+ - SAMME is now the only
    # behavior. We do not pass `algorithm=` at all, which is the
    # forward-compatible choice for any recent scikit-learn version.
    #
    # IMPORTANT DESIGN CHOICE, found by testing (not assumed): unlike the
    # tree, Bagging, and Random Forest models above, the base stump here
    # does NOT use class_weight='balanced'. When it does, AdaBoost's
    # accuracy collapses (Moderate recall drops to 0.0) because two
    # separate imbalance-correction mechanisms fight each other: AdaBoost
    # already up-weights misclassified rows every round (its core
    # mechanism), and a stump that is ALSO told "treat At-risk rows as
    # ~4x more important" over-corrects so hard in the early rounds that
    # the ensemble permanently abandons the majority class. This is a real
    # failure mode we found while building this, not a theoretical
    # concern - see the "why might AdaBoost perform worse" defense
    # question in STUDY_GUIDE.md for how to explain it live.
    ada_pipe = Pipeline(steps=[
        ("preprocess", build_supervised_preprocessor()),
        ("model", AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE),  # note: no class_weight here, see comment above
            n_estimators=200,
            learning_rate=0.5,   # shrinks each stump's contribution; smaller values need more estimators but generalize better
            random_state=RANDOM_STATE,
        )),
    ])
    t0 = time.time()
    ada_pipe.fit(X_train, y_train)
    ada_time = time.time() - t0
    cms["AdaBoost"] = evaluate("D. AdaBoost (200 stumps)", ada_pipe, X_test, y_test, results)

    # ------------------------------------------------------------------
    # PART 7: COMPARISON TABLE
    # ------------------------------------------------------------------
    for r, t in zip(results[1:], [bagging_time, rf_time, ada_time]):
        r["train_time_sec"] = round(t, 2)
    results[0]["train_time_sec"] = None  # baseline tree, not timed for comparison

    comp_df = pd.DataFrame(results).set_index("model").round(4)
    print("\n" + "=" * 70)
    print("PART 7: MODEL COMPARISON TABLE")
    print("=" * 70)
    print(comp_df)
    comp_df.to_csv(f"{OUT_DIR}/ensemble_comparison.csv")

    for name, cm in cms.items():
        safe = name.split(" ")[0].lower().replace("(", "").replace(")", "")
        plot_confusion(cm, name, f"confusion_{safe}.png")

    # ------------------------------------------------------------------
    # Random Forest feature importance (Part 5)
    # ------------------------------------------------------------------
    fitted_pre = rf_pipe.named_steps["preprocess"]
    feature_names = get_feature_names_out(fitted_pre)
    importances = rf_pipe.named_steps["model"].feature_importances_
    imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    print("\n" + "=" * 70)
    print("RANDOM FOREST FEATURE IMPORTANCE (top 15)")
    print("=" * 70)
    print(imp_series.head(15))
    print(
        "\nHOW TO READ THIS: each value is, roughly, how much that feature\n"
        "reduced impurity (Gini) averaged across all 300 trees and all splits\n"
        "that used it, then normalized so all importances sum to 1. It answers\n"
        "'how useful was this feature for making splits', NOT 'does this feature\n"
        "cause the outcome' - importance can be inflated for features that are\n"
        "merely correlated with a truly causal one."
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    imp_series.head(15)[::-1].plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_xlabel("Relative importance")
    ax.set_title("Random Forest - top 15 feature importances")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/rf_feature_importance.png", dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------------
    # PART 21: LIGHT HYPERPARAMETER TUNING (Random Forest, since it's the
    # front-runner) via RandomizedSearchCV over the WHOLE PIPELINE, so
    # preprocessing is refit correctly inside every CV fold.
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 21: HYPERPARAMETER TUNING (RandomizedSearchCV, Random Forest)")
    print("=" * 70)
    print(
        "We tune the PIPELINE object (preprocess + model), not just the\n"
        "classifier. If we tuned a classifier on already-transformed data,\n"
        "cross-validation would still leak: the encoder/imputer would have been\n"
        "fit once on ALL training data before the CV folds were even created.\n"
        "Passing the full Pipeline to RandomizedSearchCV means sklearn refits\n"
        "preprocessing from scratch inside every fold.\n"
    )
    param_dist = {
        "model__n_estimators": [100, 200, 300, 400],
        "model__max_depth": [None, 8, 12, 16, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 3, 5],
        "model__max_features": ["sqrt", "log2", 0.5],
    }
    print(
        "WHY these 5 parameters (and not more): \n"
        " - n_estimators: more trees generally helps until it plateaus; mainly a\n"
        "   compute-vs-marginal-benefit tradeoff, not an overfitting risk by itself.\n"
        " - max_depth: controls how complex each tree is allowed to get; too deep\n"
        "   -> individual trees overfit (though the ensemble buffers this).\n"
        " - min_samples_split / min_samples_leaf: force each leaf to represent a\n"
        "   minimum number of real people, preventing trees from carving out\n"
        "   leaves for single data points (a classic overfitting pattern).\n"
        " - max_features: controls how much randomness is injected between trees\n"
        "   (the key knob that makes Random Forest different from plain Bagging).\n"
        "We deliberately do NOT search things like criterion or bootstrap ratio -\n"
        "expanding the search space rarely pays off as much as tuning these five,\n"
        "and a huge grid would just slow this down without much benefit."
    )

    search = RandomizedSearchCV(
        estimator=Pipeline(steps=[
            ("preprocess", build_supervised_preprocessor()),
            ("model", RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)),
        ]),
        param_distributions=param_dist,
        n_iter=20,               # 20 random combinations - manageable, not an exhaustive grid
        scoring="f1_macro",      # optimize for the metric that matters given class imbalance
        cv=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"\nBest CV macro-F1: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    tuned_pipe = search.best_estimator_
    cms["Random Forest (tuned)"] = evaluate(
        "Random Forest - TUNED", tuned_pipe, X_test, y_test, results
    )
    tuned_row = pd.DataFrame(results).set_index("model").round(4)
    tuned_row.to_csv(f"{OUT_DIR}/ensemble_comparison_with_tuned.csv")

    print(
        "\nNOTE: hyperparameters were selected using ONLY cross-validated\n"
        "performance on the TRAINING split (search.fit(X_train, y_train)).\n"
        "X_test was not touched until the single evaluate() call above - so the\n"
        "test score you see is a fair, unbiased estimate of real-world\n"
        "performance, not a number that was implicitly optimized against."
    )

    return comp_df, tuned_row


if __name__ == "__main__":
    main()
