"""
01_eda.py
=========
RUN FROM: the `src/` folder -> `python 01_eda.py`
PURPOSE: Full exploratory audit of the dataset (Part 1 of the assignment),
including the target-leakage investigation that determines which columns
are safe to use later. Nothing here trains a "real" model - the small
trees below exist ONLY to test whether a set of columns can predict the
target suspiciously well (i.e. as a leakage detector).

This script only reads data and prints/saves findings. It never fits
anything that gets reused elsewhere - that all happens in
02_ensemble_learning.py using the shared preprocessing.py pipeline.
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from config import TARGET, ID_COLUMN, LEAKAGE_COLUMNS, RANDOM_STATE
from preprocessing import load_raw_data

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

DATA_PATH = "../data/social_media_screentime_mental_health_2026.csv"


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    df = load_raw_data(DATA_PATH)

    section("1. SHAPE")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    section("2. DTYPES")
    print(df.dtypes)

    section("3. MISSING VALUES")
    miss = df.isna().sum()
    miss_pct = (miss / len(df) * 100).round(2)
    miss_table = pd.DataFrame({"missing_count": miss, "missing_pct": miss_pct})
    print(miss_table[miss_table["missing_count"] > 0])
    print("\nAll other columns: 0 missing.")

    section("4. DUPLICATE ROWS")
    print(f"Fully duplicate rows: {df.duplicated().sum()}")
    print(f"Duplicate participant_id values: {df[ID_COLUMN].duplicated().sum()}")

    section("5. TARGET DISTRIBUTION (wellbeing_band)")
    print(df[TARGET].value_counts())
    print()
    print((df[TARGET].value_counts(normalize=True) * 100).round(1).astype(str) + "%")
    print(
        "\n-> Imbalanced: majority class 'Moderate' alone accounts for ~53.8%.\n"
        "   A model that always predicts 'Moderate' would score ~53.8% accuracy\n"
        "   while catching ZERO At-risk cases. This is why later scripts report\n"
        "   macro-F1 and per-class recall, not just accuracy."
    )

    section("6. DESCRIPTIVE STATISTICS (numeric columns)")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    print(df[numeric_cols].describe().T)

    section("7. UNIQUE VALUES (categorical columns)")
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    cat_cols = [c for c in cat_cols if c not in (ID_COLUMN, TARGET)]
    for c in cat_cols:
        print(f"\n--- {c} ({df[c].nunique()} unique) ---")
        print(df[c].value_counts(dropna=False))

    section("8. MISSINGNESS vs TARGET (is missing data random?)")
    for col in ["gender", "avg_sleep_hours"]:
        sub = df[df[col].isna()][TARGET].value_counts(normalize=True).round(3)
        print(f"\nwellbeing_band distribution WHERE {col} is missing:")
        print(sub)
    print("\noverall wellbeing_band distribution:")
    print(df[TARGET].value_counts(normalize=True).round(3))
    print(
        "\n-> The distribution among missing rows is close enough to the overall\n"
        "   distribution (no column's missingness isolates one class) that we treat\n"
        "   these as safe to impute with simple strategies (median / a constant\n"
        "   'Unknown' category) rather than needing a more complex MICE-style approach."
    )

    section("9. TARGET LEAKAGE AUDIT")
    print(
        "Question: could anxiety_score_0to27, low_mood_score_0to27, or the other\n"
        "psychological self-report scores make this classification problem\n"
        "artificially easy, or worse, BE the definition of the target?\n"
    )
    print("Test A: predict wellbeing_band using ONLY the two suspect columns.")
    X_leak = df[LEAKAGE_COLUMNS]
    y = df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(
        X_leak, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    leak_tree = DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE)
    leak_tree.fit(Xtr, ytr)
    leak_acc = accuracy_score(yte, leak_tree.predict(Xte))
    print(f"Held-out test accuracy using ONLY {LEAKAGE_COLUMNS}: {leak_acc:.4f}")

    print("\nThe exact rule a shallow tree recovers (fit on full data, depth=3):")
    full_tree = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
    full_tree.fit(X_leak, y)
    print(f"  training accuracy of this rule: {full_tree.score(X_leak, y):.4f}")
    print(export_text(full_tree, feature_names=LEAKAGE_COLUMNS))

    print(
        "CONCLUSION: wellbeing_band is a deterministic function of anxiety_score_0to27\n"
        "and low_mood_score_0to27 (severity-band cutoffs, similar to clinical screening\n"
        "tools like GAD-7/PHQ-style banding). These two columns literally ENCODE the\n"
        "label. Including them as predictive features would not be 'learning' anything -\n"
        "the model would just be decoding the label formula. DECISION: excluded from\n"
        "the supervised feature set (see config.py -> LEAKAGE_COLUMNS)."
    )

    print("\nTest B: the other 5 psychological scores (life_satisfaction, loneliness,")
    print("self_esteem, fomo, social_comparison) - are these ALSO part of the formula,")
    print("or do they carry independent (if weaker) signal?")
    other_psych = [
        "life_satisfaction_1to10", "loneliness_1to10", "self_esteem_1to10",
        "fomo_1to10", "social_comparison_1to10",
    ]
    X_other = df[other_psych]
    Xtr2, Xte2, ytr2, yte2 = train_test_split(
        X_other, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    other_tree = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
    other_tree.fit(Xtr2, ytr2)
    other_acc = accuracy_score(yte2, other_tree.predict(Xte2))
    print(f"Held-out accuracy using ONLY the other 5 psych scores: {other_acc:.4f}")
    print(
        "-> Far from perfect (baseline majority-class accuracy is ~0.538), so these\n"
        "   five are NOT part of the label formula. They carry real but modest signal,\n"
        "   the same way daily_screen_hours or avg_sleep_hours do. DECISION: keep them\n"
        "   as legitimate predictive features (they describe general wellbeing\n"
        "   dimensions distinct from the anxiety/mood clinical scores that define the\n"
        "   label itself)."
    )

    section("10. SANITY CHECK: are behavior columns realistically related to anxiety?")
    corr_cols = [
        "daily_screen_hours", "daily_notifications", "avg_sleep_hours",
        "minutes_to_first_check_after_waking", "physical_activity_days_per_week",
    ]
    corr = df[corr_cols + ["anxiety_score_0to27"]].corr()["anxiety_score_0to27"].sort_values(ascending=False)
    print(corr)
    print(
        "-> Moderate, believable correlations (e.g. daily_screen_hours ~0.52,\n"
        "   avg_sleep_hours ~ -0.26), not near-zero and not near-1. This is consistent\n"
        "   with real screen-time/mental-health research and confirms the dataset was\n"
        "   built with a genuine underlying relationship, not just random labels -\n"
        "   which is why the legitimate (leakage-free) prediction task in\n"
        "   02_ensemble_learning.py is hard but not hopeless."
    )

    section("11. COLUMN ROLE SUMMARY")
    print(
        "Identifier (dropped everywhere):        participant_id\n"
        "Target:                                  wellbeing_band\n"
        "Excluded - target leakage:               anxiety_score_0to27, low_mood_score_0to27\n"
        "Numeric predictive features:             age, platforms_used_count,\n"
        "                                          daily_screen_hours, daily_notifications,\n"
        "                                          minutes_to_first_check_after_waking,\n"
        "                                          avg_sleep_hours, life_satisfaction_1to10,\n"
        "                                          loneliness_1to10, self_esteem_1to10,\n"
        "                                          fomo_1to10, social_comparison_1to10,\n"
        "                                          physical_activity_days_per_week\n"
        "Ordinal categorical features:            night_time_use, attempted_digital_detox,\n"
        "                                          seeks_mental_health_support,\n"
        "                                          uses_screen_time_limits\n"
        "Nominal categorical features:            gender, occupation, region,\n"
        "                                          most_used_platform, primary_purpose\n"
    )

    print("\nEDA complete. See src/config.py for how these decisions are encoded in code.")


if __name__ == "__main__":
    main()
