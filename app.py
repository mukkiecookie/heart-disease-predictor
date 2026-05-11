import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

def get_binary_file_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'
    return href

st.title("Heart Disease Predictor")
tab1,tab2,tab3 = st.tabs(['Predict','Bulk Predict','Model Information'])

with tab1:
    st.header("Predict Heart Disease")
    st.write("Enter the following details to predict the likelihood of heart disease:")

    age = st.number_input("Age (years)", min_value=1, max_value=150)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
    cholesterol = st.number_input("Serum Cholesterol (mm/dl)", min_value=0)
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["< 120 mg/dl", "> 120 mg/dl"])
    resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0)
    st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])

    # Convert categorical inputs to numeric
    sex = 0 if sex == "Male" else 1
    chest_pain = ["Atypical Angina", "Non-Anginal Pain", "Asymptomatic", "Typical Angina"].index(chest_pain)
    fasting_bs = 1 if fasting_bs == "> 120 mg/dl" else 0
    resting_ecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

    # Create a DataFrame with user inputs
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })

    # Load the models and make predictions
    algonames = ["Decision Tree", "Random Forest", "Support Vector Machine", "Logistic Regression"]
    modelnames = ["DecisionTree.pkl", "RandomForest.pkl", "SVM.pkl", "LogisticR.pkl"]

    predictions = []
    def predict_heart_disease(data):
        for modelname in modelnames:
            model = pickle.load(open(modelname, "rb"))
            prediction = model.predict(data)
            predictions.append(prediction)
        return predictions
    
    # Create a button to trigger the prediction
    if st.button("Predict"):

        st.subheader("Results:")
        st.markdown("--------------------------------")

        results = predict_heart_disease(input_data)

        heart_detected = False

        for i in range(len(results)):
            st.subheader(algonames[i])

            if results[i][0] == 0:
                st.write("No heart disease detected.")
            else:
                st.write("Heart disease detected.")
                heart_detected = True

            st.markdown("--------------------------------")


        # FINAL RESULT IMAGE
        if heart_detected:
            st.error("⚠️ Heart Disease Detected")

            st.image(
                "heart_disease.png",
                caption="Please consult a doctor.",
                use_container_width=True
            )

        else:
            st.success("✅ No Heart Disease Detected")

            st.image(
                "healthy_heart.png",
                caption="Your heart looks healthy!",
                use_container_width=True
            )

with tab2:
    st.header("Bulk Predict Heart Disease")
    st.write("Upload a CSV file with the following columns to predict heart disease for multiple patients:")
    st.info("""
            1. No NaN values allowed.
            2. Total 11 columns required in this order. (Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope).\n
            3. Check the spelling of the column names and values carefully.
            4. Featur values conventions: \n
                - Age: age in years (integer) \n
                - Sex: "Male" or "Female" \n
                - ChestPainType: "Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic" \n
                - RestingBP: resting blood pressure in mm Hg (integer) \n
                - Cholesterol: serum cholesterol in mg/dl (integer) \n
                - FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, else 0] \n
                - RestingECG: "Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy" \n
                - MaxHR: maximum heart rate achieved (integer) \n
                - ExerciseAngina: Exercise-induced angina [1: if exercise induces angina, else 0] \n
                - Oldpeak: ST depression induced by exercise (float) \n
                - ST_Slope: "Upsloping", "Flat", "Downsloping" \n
            """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        model = pickle.load(open("LogisticR.pkl", "rb"))

        # Ensure that the input data has the correct columns and order
        expected_columns = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"]

        if set(expected_columns).issubset(input_data.columns):

            X = input_data[expected_columns]

            predictions = model.predict(X)

            input_data["Prediction LR"] = predictions

            input_data.to_csv("PredictedHeartLR.csv", index=False)

            st.subheader("Predictions:")
            st.write(input_data)

            st.markdown(get_binary_file_downloader_html(input_data), unsafe_allow_html=True)

        else:
            st.error("The uploaded CSV file does not contain the required columns. Please check the column names and try again.")

    else:
        st.warning("Please upload a CSV file to get predictions.")


with tab3:
    import plotly.express as px

    st.header("Model Information")
    st.write("This application uses the following machine learning models to predict heart disease:")
    st.markdown("""
    1. **Decision Tree**: A simple tree-based model that splits the data based on feature values to make predictions.
    2. **Random Forest**: An ensemble of decision trees that improves accuracy by averaging the predictions of multiple trees.
    3. **Support Vector Machine (SVM)**: A model that finds the optimal hyperplane to separate classes in the feature space.
    4. **Logistic Regression**: A linear model that estimates the probability of a binary outcome based on input features.
    """)

    data = {
        "Model": ["Decision Tree", "Random Forest", "Support Vector Machine", "Logistic Regression"],
        "Accuracy": [80.97, 84.23, 84.22, 85.86]
    }

    df = pd.DataFrame(data)
    fig = px.bar(df, x="Model", y="Accuracy", title="Model Accuracy Comparison", text="Accuracy")
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig)

