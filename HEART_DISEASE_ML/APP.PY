import streamlit as st
import pandas as pd
import joblib
model=joblib.load("KNN_heart.pkl")
scaler=joblib.load("scaler.pkl")
expected_col=joblib.load("columns.pkl")
st.title("HEART STROKE PREDICTION BY SHREYA")
st.markdown("provide the following details")
age=st.slider("Age",18,100,40)
sex=st.selectbox("SEX",['M','F'])
chest_pain=st.selectbox("chest pain type",["ATA","NAP","TA","ASY"])
resting_bp=st.number_input("Resting blood pressure",min_value=80,max_value=200,value=120)
cholesterol=st.number_input("Cholesterol",min_value=100,max_value=600,value=200)
fasting_bs=st.selectbox("Fasting blood sugar",["<120","120->120"])
resting_ecg=st.selectbox("Resting electrocardiographic results",["Normal","ST-T wave abnormality","Left ventricular hypertrophy"])
max_hr=st.number_input("Maximum heart rate achieved",min_value=60,max_value=220,value=150)
exercise_angina=st.selectbox("Exercise induced angina",["Y","N"])
oldpeak=st.number_input("Oldpeak",min_value=0.0,max_value=10.0,value=1.0)

st_slope=st.number_input("ST Slope",min_value=0.0,max_value=10.0,value=1.0)
if st.button("Predict"):
    input={
        'age':age,
        'sex':sex,
        'chest_pain':chest_pain,
        'resting_bp':resting_bp,
        'cholesterol':cholesterol,
        'fasting_bs':fasting_bs,
        'resting_ecg':resting_ecg,
        'max_hr':max_hr,
        'exercise_angina':exercise_angina,
        'oldpeak':oldpeak,
        'st_slope':st_slope

    }
    input_df=pd.DataFrame([input])
    input_df=pd.get_dummies(input_df)
    for col in expected_col:
        if col not in input_df.columns:
            input_df[col]=0
    input_df=input_df[expected_col]
    input_df=scaler.transform(input_df)
    prediction=model.predict(input_df)
    if prediction[0]==0:
        st.success("The patient is not at risk of heart disease.")
    else:
        st.error("The patient is at risk of heart disease.")