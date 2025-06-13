# render_template loads HTML files (like index.html, predict.html, result.html).
# request handles data coming from the HTML form.
from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

# This creates the Flask application instance.
app = Flask(__name__)              

# Load model
MODEL_PATH = os.path.join("model", "best_pcb_model.joblib")
model = joblib.load(MODEL_PATH)

# Feature setup
NUMERIC_FIELDS = [
    'HCl (%)', 'H2SO4 (%)', 'CuSO4 (%)', 'pH Value', 'Temperature (°C)',
    'Etching Time (s)', 'Capacitance (pF)', 'Insulation Resistance (MΩ)',
    'Impedance Match (Ω)', 'Dielectric Breakdown (kV)', 'Board Thickness (mm)',
    'Board Width (mm)', 'Board Length (mm)', 'Hole Diameter Deviation (mm)',
    'Pad-to-Hole Alignment (mm)'
]

CATEGORICAL_FIELDS = [
    'Silkscreen Clarity', 'Solderability Test',
    'Dimension Accuracy', 'Frequency Test Result', 'Defect Type'
]

NUMERIC_RANGES = {
    'HCl (%)': (2.5, 4.5),
    'H2SO4 (%)': (2.0, 4.0),
    'CuSO4 (%)': (0.8, 1.6),
    'pH Value': (4.0, 6.0),
    'Temperature (°C)': (40, 50),
    'Etching Time (s)': (80, 140),
    'Capacitance (pF)': (20, 40),
    'Insulation Resistance (MΩ)': (10, 100),
    'Impedance Match (Ω)': (45, 55),
    'Dielectric Breakdown (kV)': (2.0, 5.0),
    'Board Thickness (mm)': (1.0, 1.6),
    'Board Width (mm)': (120, 180),
    'Board Length (mm)': (180, 220),
    'Hole Diameter Deviation (mm)': (0.0, 0.2),
    'Pad-to-Hole Alignment (mm)': (0.0, 0.2)
}


# Logic rules to override ML output
# If Silkscreen is Blurred, or Solderability Test fails, etc. → it’s faulty, no matter what the ML model says.
# Also checks if numeric inputs are within expected ranges.
def check_rules(row):
    if (
        row['Silkscreen Clarity'] == 'Blurred' or
        row['Solderability Test'] == 'Fail' or
        row['Dimension Accuracy'] == 'Inaccurate' or
        row['Frequency Test Result'] == 'Fail' or
        row['Defect Type'] != 'None'
    ):
        return True

    if not (
        2.5 <= row['HCl (%)'] <= 4.5 and
        2.0 <= row['H2SO4 (%)'] <= 4.0 and
        0.8 <= row['CuSO4 (%)'] <= 1.6 and
        4.0 <= row['pH Value'] <= 6.0 and
        40 <= row['Temperature (°C)'] <= 50 and
        80 <= row['Etching Time (s)'] <= 140 and
        20 <= row['Capacitance (pF)'] <= 40 and
        row['Insulation Resistance (MΩ)'] > 30 and
        45 <= row['Impedance Match (Ω)'] <= 55 and
        row['Dielectric Breakdown (kV)'] > 2.5 and
        1.0 <= row['Board Thickness (mm)'] <= 1.6 and
        120 <= row['Board Width (mm)'] <= 180 and
        180 <= row['Board Length (mm)'] <= 220 and
        row['Hole Diameter Deviation (mm)'] <= 0.1 and
        row['Pad-to-Hole Alignment (mm)'] <= 0.1
    ):
        return True

    return False


# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            form_data = request.form
            input_data = {}

            # Collect numeric inputs
            for field in NUMERIC_FIELDS:
                input_data[field] = float(form_data.get(field, 0))

            # Collect categorical inputs
            for field in CATEGORICAL_FIELDS:
                input_data[field] = form_data.get(field, 'None')

            # Convert to DataFrame
            df = pd.DataFrame([input_data])

            # Apply rules
            rule_failed = check_rules(df.iloc[0])

            # Drop defect type before prediction
            model_input = df.drop(columns=["Defect Type"])
            model_prediction = model.predict(model_input)[0]

            # Override if rule fails
            final_result = 0 if rule_failed else model_prediction
            prediction_label = "✅ OK PCB" if final_result == 1 else "❌ Faulty PCB"

            return render_template("result.html", prediction=prediction_label)

        except Exception as e:
            return render_template("result.html", prediction=f"❗ Error: {e}")

    # GET method - show form
    return render_template(
        "predict.html",
        numeric_fields=NUMERIC_FIELDS,
        numeric_ranges=NUMERIC_RANGES,
        categorical_fields=CATEGORICAL_FIELDS
    )

@app.route("/result")
def result():
    return render_template("result.html", prediction="Use the form to predict.")


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
