import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/Train.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import os
for dirname, _, filenames in os.walk('/Test.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Step 1: Load the data
# Replace 'data.csv' with your dataset's filename
train_df = pd.read_csv('Train.csv')
test_df = pd.read_csv('Test.csv')

print(train_df.head())

print(train_df.describe())

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(train_df.select_dtypes("number").corr());

plt.show()

# Define features and target variable
X = train_df.drop(['id', 'smoking'], axis=1)
y = train_df['smoking']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0001, random_state=42)

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, roc_curve

# Step 3: Create preprocessing pipelines
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=20, n_jobs=-1, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

#  Evaluate the model
# Predict probabilities
y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

# Calculate AUC-ROC
auc = roc_auc_score(y_test, y_pred_prob)
print(f'AUC-ROC: {auc:.2f}')

# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Once you are satisfied with the model performance, you can make predictions on the test set
test_features = test_df.drop('id', axis=1)
test_predictions = pipeline.predict_proba(test_features)[:, 1]

# submission = pd.DataFrame({'id': test_df['id'], 'smoking': test_predictions})
# submission.to_csv('C:/Users/Parv Sidana/Desktop/project/submission.csv', index=False)

import pickle
pickle_out = open("classifier.pkl","wb")
pickle.dump(pipeline, pickle_out)
pickle_out.close()

# Define the feature names (these should match the columns used for training)
feature_names = ['age', 'height', 'weight', 'blood_pressure', 'cholesterol', 'smoking_habit', 'exercise_frequency', 'diabetes', 'heart_rate', 'glucose', 'bmi', 'alcohol', 'exercise_duration', 'steps_per_day', 'body_temperature', 'hydration', 'sleep_hours', 'stress_level', 'diet_quality', 'diet_duration', 'age_group', 'smoking_cessation']

# Define the column names that match the features used during training
feature_names = ['age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)', 'eyesight(right)', 
                 'hearing(left)', 'hearing(right)', 'systolic', 'relaxation', 'fasting blood sugar', 
                 'Cholesterol', 'triglyceride', 'HDL', 'LDL', 'hemoglobin', 'Urine protein', 'serum creatinine', 
                 'AST', 'ALT', 'Gtp', 'dental caries']

# Input data as a list (ensure the input is in the same order as the feature_names)
input_data = [[55,165,60,81.0,0.5,0.6,1,1,135,87,94,172,300,40,75,16.5,1,1.0,22,25,27,0]]

# Convert the input data to a DataFrame with the correct column names
input_df = pd.DataFrame(input_data, columns=feature_names)

# Now you can pass the DataFrame to predict
print(pipeline.predict(input_df))