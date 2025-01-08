import streamlit as st
import joblib
import pandas as pd
from rf import RandomForest_manual  # Import custom classes
from streamlit_option_menu import option_menu

# Page configuration
st.set_page_config(page_title="Mental Health Prediction App", layout="wide")

# Function to load resources
@st.cache_resource
def load_label_encoders(file_path):
    return joblib.load(file_path)

@st.cache_resource
def load_model(file_path):
    return joblib.load(file_path)

# Load model and label encoders
model_file = "random_forest_manual.joblib"
label_encoders_file = "label_encoders.joblib"
target_encoder = joblib.load("target_encoder.joblib")

model = load_model(model_file)
label_encoders = load_label_encoders(label_encoders_file)

# Navigation menu
selected_page = option_menu(
    "Navigation",
    ["Prediction", "About the App"],
    icons=["clipboard-data", "info-circle"],
    menu_icon="list",
    default_index=0,
    orientation="vertical"
)

if selected_page == "Prediction":
    # Header
    st.title("Mental Health Prediction Based on Music Preferences")

    # Input form
    st.header("Questionnaire")
    with st.form("input_form"):
        age = st.number_input("How old are you?", min_value=1, max_value=100, value=25)

        primary_streaming = st.selectbox("Primary streaming platform:", 
                                         label_encoders["Primary streaming service"].classes_)

        hours_per_day = st.number_input("Hours per day listening to music:", min_value=0.0, max_value=24.0, value=2.0, step=0.5)

        while_working = st.radio("Do you listen to music while working?", label_encoders["While working"].classes_)

        instrumental = st.radio("Do you listen to instrumental music?", label_encoders["Instrumentalist"].classes_)

        composer = st.radio("Do you prefer specific composers?", label_encoders["Composer"].classes_)

        fav_genre = st.selectbox("Favorite genre:", label_encoders["Fav genre"].classes_)

        exploratory = st.radio("Do you enjoy exploring new music genres?", label_encoders["Exploratory"].classes_)

        foreign_language = st.radio("Do you listen to foreign language music?", label_encoders["Foreign languages"].classes_)

        frequency_columns = [
            "Frequency [Classical]", "Frequency [Country]", "Frequency [EDM]",
            "Frequency [Folk]", "Frequency [Gospel]", "Frequency [Hip hop]",
            "Frequency [Jazz]", "Frequency [K pop]", "Frequency [Latin]", "Frequency [Lofi]",
            "Frequency [Metal]", "Frequency [Pop]", "Frequency [R&B]", "Frequency [Rap]",
            "Frequency [Rock]", "Frequency [Video game music]"
        ]

        frequency_responses = {
            col: st.selectbox(f"Frequency for {col.split('[')[1].replace(']', '')}:", 
                              label_encoders[col].classes_)
            for col in frequency_columns
        }


        music_effect = st.radio("How does music affect your mental state?", label_encoders["Music effects"].classes_)

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Build input DataFrame
        data = {
            "Age": age,
            "Primary streaming service": primary_streaming,
            "Hours per day": hours_per_day,
            "While working": while_working,
            "Instrumentalist": instrumental,
            "Composer": composer,
            "Fav genre": fav_genre,
            "Exploratory": exploratory,
            "Foreign languages": foreign_language,
            "Music effects": music_effect,
        }
        data.update(frequency_responses)

        input_df = pd.DataFrame([data])

        # Encode categorical columns
        for col in input_df.columns:
            if col in label_encoders:  # Check if the column has a LabelEncoder
                input_df[col] = label_encoders[col].transform(input_df[col])

        # Ensure column order matches model expectation
        expected_columns = [
            "Age", "Primary streaming service", "Hours per day", "While working",
            "Instrumentalist", "Composer", "Fav genre", "Exploratory", "Foreign languages",
            "Frequency [Classical]", "Frequency [Country]", "Frequency [EDM]",
            "Frequency [Folk]", "Frequency [Gospel]", "Frequency [Hip hop]",
            "Frequency [Jazz]", "Frequency [K pop]", "Frequency [Latin]", "Frequency [Lofi]",
            "Frequency [Metal]", "Frequency [Pop]", "Frequency [R&B]", "Frequency [Rap]",
            "Frequency [Rock]", "Frequency [Video game music]",
            "Music effects"
        ]

        input_df = input_df[expected_columns]

        st.write("Processed Input Data for Model:")
        st.write(input_df)

        # Predict using the model
        prediction = model.predict(input_df.to_numpy())

        # Decode prediction
        decoded_prediction = target_encoder.inverse_transform(prediction)

        # Display result
        st.subheader("Prediction Result:")
        st.write(f"Predicted Mental Health Depression Condition: {decoded_prediction[0]}")

elif selected_page == "About the App":
    st.title("About the Mental Health Prediction App")
    st.write(
        """
        ### Overview
        This application predicts mental health conditions based on users' music preferences and related attributes. 

        ### How it works
        - Users provide details about their music habits, preferences, and other factors.
        - The app processes the inputs and uses a trained Random Forest model to predict mental health conditions.
        - Results are displayed instantly after submitting the form.

        ### Data Privacy
        - The app does not store or share user input data. All processing is done locally on this session.

        ### About the Model
        - The Random Forest model was trained on a dataset of music preferences and mental health conditions.
        - Label encoding is used to preprocess categorical data for compatibility with the model.

        ### Disclaimer
        - This app is for informational purposes only and should not replace professional mental health advice.
        """
    )
