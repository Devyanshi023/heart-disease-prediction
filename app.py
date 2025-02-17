import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

# Train and Save Model (if not already saved)
@st.cache_resource
def train_and_save_model():
    if not os.path.exists("heart_disease_model.pkl"):
        st.warning("Training model... Please wait.")
        heart_data = load_data()
        X = heart_data.drop(columns='target', axis=1)
        Y = heart_data['target']

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

        model = RandomForestClassifier(n_estimators=100, random_state=2)
        model.fit(X_train, Y_train)

        joblib.dump(model, "heart_disease_model.pkl")
        st.success("Model trained and saved successfully!")

# Load Model
def load_model():
    if not os.path.exists("heart_disease_model.pkl"):
        train_and_save_model()
    return joblib.load("heart_disease_model.pkl")

# Streamlit UI
st.title("Heart Disease Prediction Web App")
st.write("Enter your health details below to get a prediction and personalized health tips.")

# User Inputs
age = st.number_input("Age", min_value=10, max_value=100, value=50)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (BP)", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol Level", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of ST Segment", options=[0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia Type", options=[0, 1, 2, 3])

# Prediction and Health Tips
if st.button("Predict"):
    model = load_model()
    
    # Convert categorical inputs to numeric values
    input_data = np.array([[age, sex, cp, trestbps, chol, int(fbs == "Yes"), restecg, thalach, int(exang == "Yes"), oldpeak, slope, ca, thal]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display Result
    if prediction[0] == 0:
        st.success("The person does NOT have heart disease.")
        
        # **General Health Tips**
        st.subheader("💡 Personalized Health Tips:")
        
        # **Blood Pressure Tips**
        if trestbps > 130:
            st.write("🔹 **High Blood Pressure:** Reduce sodium intake, exercise regularly, and stay hydrated.")
        elif trestbps < 90:
            st.write("🔹 **Low Blood Pressure:** Eat more small meals, drink fluids, and avoid sudden position changes.")

        # **Cholesterol Tips**
        if chol > 240:
            st.write("🔹 **High Cholesterol:** Eat fiber-rich foods, avoid saturated fats, and increase omega-3 intake.")
        elif chol < 160:
            st.write("🔹 **Low Cholesterol:** Consider increasing healthy fats like avocados and nuts.")

        # **Heart Rate Tips**
        if thalach < 100:
            st.write("🔹 **Low Heart Rate:** Increase cardio activity and monitor for fatigue symptoms.")
        elif thalach > 180:
            st.write("🔹 **High Heart Rate:** Reduce caffeine, improve sleep, and manage stress.")

        # **Blood Sugar Management**
        if fbs == "Yes":
            st.write("🔹 **High Blood Sugar:** Avoid processed sugars, increase fiber, and monitor glucose levels.")

        # **Exercise & Stress**
        if exang == "Yes":
            st.write("🔹 **Exercise-Induced Angina:** Avoid high-intensity workouts and follow a heart-friendly exercise plan.")
        if oldpeak > 2.0:
            st.write("🔹 **Heart Stress Warning:** Avoid high-stress situations and practice relaxation techniques like meditation.")

        # **General Wellness**
        st.write("✅ **Stay healthy!** Maintain a balanced diet, get enough sleep, and stay active.")

    else:
        st.error("The person HAS heart disease. Please consult a doctor.")

        # **Heart Disease Management Tips**
        st.subheader("⚠️ Health Tips Based on Your Risk Factors:")

        # **Blood Pressure**
        if trestbps > 130:
            st.write("🔹 **Manage High BP:** Reduce salt intake, stay physically active, and manage stress.")
        
        # **Cholesterol**
        if chol > 240:
            st.write("🔹 **Lower Cholesterol:** Eat more fruits, vegetables, and avoid fried foods.")
        
        # **Diabetes & Blood Sugar**
        if fbs == "Yes":
            st.write("🔹 **Control Blood Sugar:** Reduce sugar intake and monitor blood glucose regularly.")

        # **Heart Rate Issues**
        if thalach > 180:
            st.write("🔹 **Monitor High Heart Rate:** Reduce caffeine and avoid high-stress environments.")

        # **Exercise & Lifestyle Adjustments**
        if exang == "Yes":
            st.write("🔹 **Exercise Safely:** Choose light exercises like walking and stretching instead of intense workouts.")

        # **Stress & Mental Health**
        if oldpeak > 2.0:
            st.write("🔹 **Reduce Stress:** Try yoga, meditation, or breathing exercises to relax.")

        # **Overall Guidance**
        st.write("⚠️ **Regular Check-Ups:** Follow up with a doctor for personalized heart care advice.")

# Run with: streamlit run app.py
