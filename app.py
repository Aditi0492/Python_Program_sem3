from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model_data = joblib.load('C:/Users/athar/Desktop/Capstonef/ev_flask_app/random_forest_model.pkl')
model = model_data['model']
columns = model_data['columns']
encoders = model_data['encoders']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Step 1: Gather form data
        data = {
            'battery_current': float(request.form['battery_current']),
            'extra_load_kg': float(request.form['extra_load_kg']),
            'vehicle_model': request.form['vehicle_model'],
            'c_rating': float(request.form['c_rating']),
            'charge_cycles_weekly': float(request.form['charge_cycles_weekly']),
            'age_of_ev_years': float(request.form['age_of_ev_years']),
            'driving_conditions': request.form['driving_conditions'],
            'charging_type': request.form['charging_type'],
            'outside_temp_c': float(request.form['outside_temp_c']),
            'auxiliary_load_percent': float(request.form['auxiliary_load_percent']),
            'reduced_range_km': float(request.form['reduced_range_km'])
        }

        # Step 2: Create DataFrame
        df = pd.DataFrame([data])

        # Step 3: Apply Label Encoders (as used in training)
        for col in encoders:
            if col in df.columns:
                df[col] = encoders[col].transform(df[col])

        # Step 4: Reorder to match model input columns
        df = df[columns]

        # Step 5: Predict
        prediction = model.predict(df)[0]
        result = "Battery Breakdown Likely" if prediction == 1 else "Battery Healthy"
        return render_template('index.html', prediction_text=f"Prediction: {result}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")





if __name__ == '__main__':
    app.run(debug=True)
