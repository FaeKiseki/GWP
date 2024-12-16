from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os

# Initialize Flask app
app = Flask(__name__)

# Paths for the model and scaler
model_path = '/Users/macbookpro/Documents/GWP/Models/best_model.pkl'
scaler_path = '/Users/macbookpro/Documents/GWP/Models/scaler.pkl'

# Load the trained model and scaler
if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    raise FileNotFoundError("Model or scaler file not found. Ensure the paths are correct and files exist.")

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define the feature columns (must match training data order)
feature_columns = [
    "team", "targeted_productivity", "smv", "wip", "over_time",
    "incentive", "idle_time", "idle_men", "no_of_style_change",
    "no_of_workers", "quarter_Quarter2", "quarter_Quarter3", 
    "quarter_Quarter4", "quarter_Quarter5", "department_finishing ",
    "department_sweing", "day_Saturday", "day_Sunday", "day_Thursday",
    "day_Tuesday", "day_Wednesday", "month", "day_of_week"
]

@app.route('/')
def home():
    """Render the homepage with the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        # Collect input data from the form
        input_data = {}
        for col in feature_columns:
            input_value = request.form.get(col, default=0)
            input_data[col] = float(input_value) if input_value else 0.0

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Scale the input data
        scaled_input = scaler.transform(input_df)

        # Make predictions
        prediction = model.predict(scaled_input)[0]
        return jsonify({"prediction": round(prediction, 4)})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
