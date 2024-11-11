from pydantic import BaseModel

# Class which describes the data measurements for health features
class HealthData(BaseModel):
    age: int
    height_cm: int
    weight_kg: int
    waist_cm: float
    eyesight_left: float
    eyesight_right: float
    hearing_left: int
    hearing_right: int
    systolic: int
    relaxation: int
    fasting_blood_sugar: int
    cholesterol: int
    triglyceride: int
    HDL: int
    LDL: int
    hemoglobin: float
    urine_protein: int
    serum_creatinine: float
    AST: int
    ALT: int
    Gtp: int
    dental_caries: int
