from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

def feature_engineering(X):
    X = X.copy()
    X['AvgCharges'] = X['TotalCharges'] / (X['tenure'] + 1)
    X['AvgCharges'] = X['AvgCharges'].replace([np.inf, -np.inf], 0)
    X['AvgCharges'] = X['AvgCharges'].fillna(0)
    X['TotalCharges_log'] = np.log1p(X['TotalCharges'].fillna(0))
    return X

model = joblib.load("Customer_Churn_Prevention.pkl")

EXPECTED_COLS = [
    'gender','SeniorCitizen','Partner','Dependents','tenure','MonthlyCharges','TotalCharges',
    'PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
    'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()

        numeric_fields = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
        for field in numeric_fields:
            data[field] = float(data.get(field, 0))

        for col in EXPECTED_COLS:
            if col not in data:
                data[col] = 'No'

        df = pd.DataFrame([data])

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        result = "Churn" if prediction == 1 else "No Churn"

        return render_template('index.html', prediction_text=f"Result: {result} (Probability: {probability:.2f})")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
