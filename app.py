
import pandas as pd
import streamlit as st
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "wellbeing_pipeline.joblib")

st.set_page_config(page_title="Social Media & Wellbeing Predictor", page_icon="📱", layout="centered")


@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


pipeline = load_pipeline()

BAND_COLOR = {"Good": "#2E7D32", "Moderate": "#F9A825", "At-risk": "#C62828"}
BAND_ORDER = ["At-risk", "Moderate", "Good"]

st.title("📱 Social Media Habits & Wellbeing Predictor")
st.caption(
    "NTI Machine Learning internship project - Ensemble Learning + Deployment. "
    "Trained on a survey dataset (n=7,000) using a Random Forest classifier "
    "inside a scikit-learn Pipeline."
)
st.info(
    "This is a student project prototype trained on survey data for a learning "
    "exercise - it is **not** a clinical or diagnostic tool. If you're genuinely "
    "concerned about your mental wellbeing, please talk to a doctor, counselor, "
    "or a trusted person in your life.",
    icon="ℹ️",
)

st.divider()
st.subheader("Tell us about your social media habits")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**About you**")
        age = st.number_input("Age", min_value=13, max_value=100, value=25, step=1)
        gender = st.selectbox("Gender", ["Male", "Female", "Non-binary", "Prefer not to say"])
        occupation = st.selectbox(
            "Occupation",
            ["Student", "Full-time employed", "Part-time employed", "Self-employed", "Unemployed", "Retired"],
        )
        region = st.selectbox(
            "Region", ["Africa", "Asia", "Europe", "Latin America", "North America", "Oceania"]
        )

        st.markdown("**General wellbeing**")
        life_satisfaction = st.slider("Life satisfaction (1 = very low, 10 = very high)", 1, 10, 6)
        loneliness = st.slider("Loneliness (1 = never lonely, 10 = constantly lonely)", 1, 10, 5)
        self_esteem = st.slider("Self-esteem (1 = very low, 10 = very high)", 1, 10, 6)
        physical_activity = st.slider("Physical activity - days per week", 0, 7, 3)

    with col2:
        st.markdown("**Your social media usage**")
        most_used_platform = st.selectbox(
            "Most-used platform",
            ["TikTok", "Instagram", "YouTube", "Facebook", "Snapchat", "X/Twitter", "Reddit", "LinkedIn"],
        )
        primary_purpose = st.selectbox(
            "Primary purpose of use",
            ["Entertainment", "Connection with friends", "News/information",
             "Passing time/boredom", "Work/career", "Content creation"],
        )
        platforms_used_count = st.slider("Number of different platforms used regularly", 1, 8, 3)
        daily_screen_hours = st.slider("Average daily screen time (hours)", 0.0, 16.0, 3.3, step=0.1)
        daily_notifications = st.number_input("Daily notifications received", min_value=0, max_value=500, value=60, step=1)
        minutes_to_first_check = st.number_input(
            "Minutes after waking before first checking your phone", min_value=0, max_value=180, value=20, step=1
        )
        night_time_use = st.selectbox("How often do you use social media at night in bed?", ["Never", "Sometimes", "Often", "Every night"])
        avg_sleep_hours = st.slider("Average sleep per night (hours)", 3.0, 12.0, 7.0, step=0.1)

        st.markdown("**More wellbeing scores**")
        fomo = st.slider("FOMO - fear of missing out (1 = none, 10 = very high)", 1, 10, 5)
        social_comparison = st.slider("Social comparison tendency (1 = none, 10 = very high)", 1, 10, 5)

    st.markdown("**Habits & support-seeking**")
    col3, col4, col5 = st.columns(3)
    with col3:
        uses_screen_time_limits = st.radio("Do you use screen-time limit features?", ["No", "Yes"], horizontal=True)
    with col4:
        attempted_digital_detox = st.selectbox(
            "Have you attempted a digital detox?", ["No", "Yes, failed", "Yes, succeeded"]
        )
    with col5:
        seeks_mental_health_support = st.selectbox(
            "Do you seek mental health support?", ["No", "Considering it", "Yes"]
        )

    submitted = st.form_submit_button("Predict my wellbeing band", use_container_width=True)

if submitted:
    # Build a single-row DataFrame with EXACTLY the raw column names the
    # pipeline was trained on (src/config.py -> ALL_SUPERVISED_FEATURES).
    # No manual encoding here - the pipeline does all of that internally.
    input_row = pd.DataFrame([{
        "age": age,
        "platforms_used_count": platforms_used_count,
        "daily_screen_hours": daily_screen_hours,
        "daily_notifications": daily_notifications,
        "minutes_to_first_check_after_waking": minutes_to_first_check,
        "avg_sleep_hours": avg_sleep_hours,
        "life_satisfaction_1to10": life_satisfaction,
        "loneliness_1to10": loneliness,
        "self_esteem_1to10": self_esteem,
        "fomo_1to10": fomo,
        "social_comparison_1to10": social_comparison,
        "physical_activity_days_per_week": physical_activity,
        "night_time_use": night_time_use,
        "attempted_digital_detox": attempted_digital_detox,
        "seeks_mental_health_support": seeks_mental_health_support,
        "uses_screen_time_limits": uses_screen_time_limits,
        "gender": gender,
        "occupation": occupation,
        "region": region,
        "most_used_platform": most_used_platform,
        "primary_purpose": primary_purpose,
    }])

    prediction = pipeline.predict(input_row)[0]
    probabilities = pipeline.predict_proba(input_row)[0]
    proba_series = pd.Series(probabilities, index=pipeline.classes_).reindex(BAND_ORDER)

    st.divider()
    color = BAND_COLOR.get(prediction, "#333333")
    st.markdown(
        f"<div style='padding:1.2rem;border-radius:0.6rem;background-color:{color}22;"
        f"border:2px solid {color};text-align:center;'>"
        f"<span style='font-size:0.95rem;color:#444;'>Predicted wellbeing band</span><br>"
        f"<span style='font-size:2rem;font-weight:700;color:{color};'>{prediction}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("##### Predicted probabilities")
    st.caption(
        "These are the model's predicted probabilities across the three bands, "
        "not a guaranteed or certain outcome - a real-world prediction from a "
        "model with modest overall accuracy (see the project report)."
    )
    st.bar_chart(proba_series)

    with st.expander("See the exact values sent to the model"):
        st.dataframe(input_row.T.rename(columns={0: "value"}))

    if prediction == "At-risk":
        st.warning(
            "Reminder: this tool is a class project demo, not a diagnosis. If any of "
            "this resonates with how you're actually feeling, consider talking to "
            "someone you trust or a mental health professional.",
            icon="💛",
        )

st.divider()
with st.expander("How this model works (for the project discussion)"):
    st.markdown(
       