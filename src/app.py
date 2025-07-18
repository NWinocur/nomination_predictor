import streamlit as st
import pandas as pd
import pickle
import os

# Load the pipeline
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/pipeline.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Define prediction labels
class_dict = {
    0: "Extrovert",
    1: "Introvert"
}

# Define Streamlit app
st.title("Personality Predictor: Introvert vs. Extrovert")

st.markdown("Answer the questionnaire below. Your responses will be used to predict your personality inclination.")

# User inputs
time_alone = st.number_input("Time spent alone daily (hours)", min_value=0, max_value=11, value=5)
stage_fear = st.radio("Do you have stage fright?", ["Yes", "No"])
social_events = st.slider("How often do you attend social events?", 0, 10, 5)
going_out = st.slider("How often do you go outside?", 0, 7, 3)
drained = st.radio("Do you feel drained after socializing?", ["Yes", "No"])
friends_circle = st.slider("How many close friends do you have?", 0, 15, 5)
post_freq = st.slider("How often do you post on social media?", 0, 10, 5)

# Make prediction
if st.button("Predict"):
    input_df = pd.DataFrame([{
        'Time_spent_Alone': time_alone,
        'Stage_fear': stage_fear,
        'Social_event_attendance': social_events,
        'Going_outside': going_out,
        'Drained_after_socializing': drained,
        'Friends_circle_size': friends_circle,
        'Post_frequency': post_freq
    }])

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    pred_label = class_dict[prediction]

    st.subheader(f"Prediction: {pred_label}")
    st.write("**Confidence Levels:**")
    for idx, prob in enumerate(proba):
        st.write(f"- {class_dict[idx]}: {round(prob * 100, 1)}%")
