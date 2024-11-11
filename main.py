import uvicorn
from data import HealthData
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd

# Create the app object
app = FastAPI()
pickle_in = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_in)

@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To Krish Youtube Channel': f'{name}'}


# Expose the prediction functionality
@app.post('/predict')
def predict_banknote(data: HealthData):
    age = data.age
    height = data.height_cm
    weight = data.weight_kg
    waist = data.waist_cm
    eyesight_left = data.eyesight_left
    eyesight_right = data.eyesight_right
    hearing_left = data.hearing_left
    hearing_right = data.hearing_right
    systolic = data.systolic
    relaxation = data.relaxation
    fasting_blood_sugar = data.fasting_blood_sugar
    cholesterol = data.cholesterol
    triglyceride = data.triglyceride
    HDL = data.HDL
    LDL = data.LDL
    hemoglobin = data.hemoglobin
    urine_protein = data.urine_protein
    serum_creatinine = data.serum_creatinine
    AST = data.AST
    ALT = data.ALT
    Gtp = data.Gtp
    dental_caries = data.dental_caries
    # Create a DataFrame from the extracted values
    input_data = pd.DataFrame([{
        'age': age,
        'height(cm)': height,  # Corrected column name
        'weight(kg)': weight,  # Corrected column name
        'waist(cm)': waist,    # Corrected column name
        'eyesight(left)': eyesight_left,  # Corrected column name
        'eyesight(right)': eyesight_right,  # Corrected column name
        'hearing(left)': hearing_left,  # Corrected column name
        'hearing(right)': hearing_right,  # Corrected column name
        'systolic': systolic,
        'relaxation': relaxation,
        'fasting blood sugar': fasting_blood_sugar,  # Corrected column name
        'Cholesterol': cholesterol,  # Corrected column name
        'triglyceride': triglyceride,
        'HDL': HDL,
        'LDL': LDL,
        'hemoglobin': hemoglobin,
        'Urine protein': urine_protein,  # Corrected column name
        'serum creatinine': serum_creatinine,  # Corrected column name
        'AST': AST,
        'ALT': ALT,
        'Gtp': Gtp,
        'dental caries': dental_caries  # Corrected column name
    }])


    # Now you can safely pass this DataFrame to the classifier
    prediction = classifier.predict(input_data)
    if prediction[0] > 0.5:
        prediction = "SMOKER"
    else:
        prediction = "NON-SMOKER"
    return {
        'prediction': prediction
    }