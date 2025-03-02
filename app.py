from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the models
model_paths = {
    "Random Forest": "models/random_forest.pkl",
    "Gradient Boosting": "models/gradient_boosting.pkl",
    "XGBoost": "models/xgboost.pkl"
}

models = {}
for model_name, model_path in model_paths.items():
    try:
        with open(model_path, 'rb') as file:
            models[model_name] = pickle.load(file)
    except FileNotFoundError:
        print(f"Warning: Model file '{model_path}' not found.")
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")

# Load the scaler
scaler_path = "models/scaler.pkl"
scaler = None
try:
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    print(f"Warning: Scaler file '{scaler_path}' not found.")
except Exception as e:
    print(f"Error loading scaler: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form/<model_type>')
def form(model_type):
    if model_type not in models:
        return "Model not found", 404
    return render_template('form.html', model_type=model_type)

@app.route('/submit_form', methods=['POST'])
def submit_form():
    model_type = request.form.get('model_type')
    if model_type not in models:
        return "Model not found", 404

    model = models[model_type]

    # Collect form data
    try:
        data = [
            float(request.form['distance_to_solar_noon']),
            float(request.form['temperature']),
            float(request.form['wind_direction']),
            float(request.form['wind_speed']),
            float(request.form['sky_cover']),
            float(request.form['visibility']),
            float(request.form['humidity']),
            float(request.form['average_wind_speed']),
            float(request.form['average_pressure'])
        ]
    except ValueError:
        return "Invalid input data", 400

    # Standardize the data
    if scaler:
        data = scaler.transform([data])

    # Perform prediction
    power_generated = model.predict(data)[0]

    return render_template('form.html', power_generated=power_generated, model_type=model_type)

if __name__ == '__main__':
    app.run(debug=True)
