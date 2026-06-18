import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import datetime
import warnings

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
import yaml

warnings.filterwarnings("ignore")

app = Flask(__name__)


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


config = load_config()
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
model_name = config["mlflow"]["registry_model_name"]

try:
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
    MODEL_LOADED = True
    print(f"[INFO] Model '{model_name}' loaded from Production")
except Exception as e:
    print(f"[WARN] Could not load model: {e}")
    MODEL_LOADED = False
    model = None

try:
    proc_dir = config["data"]["processed_dir"]
    _X = pd.read_csv(os.path.join(proc_dir, "brfss_X_test.csv"), nrows=1)
    from src.feature_engineering import engineer_brfss_features
    _X = engineer_brfss_features(_X)
    FEATURE_NAMES = list(_X.columns)
    N_FEATURES = len(FEATURE_NAMES)
except Exception:
    FEATURE_NAMES = []
    N_FEATURES = 0


@app.route("/")
def home():
    return jsonify({
        "service": "Heart Disease Prediction API",
        "model": model_name,
        "status": "running",
        "endpoints": ["/health", "/model-info", "/predict",
                      "/predict_batch", "/predict-form"]
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "timestamp": datetime.datetime.now().isoformat()
    })


@app.route("/model-info")
def model_info():
    return jsonify({
        "model_name": model_name,
        "stage": "Production",
        "n_features": N_FEATURES,
        "status": "loaded" if MODEL_LOADED else "not loaded",
        "timestamp": datetime.datetime.now().isoformat()
    })


@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()
    if "features" not in data:
        return jsonify({"error": "Missing 'features' key"}), 400

    features = data["features"]
    df = pd.DataFrame([features], columns=FEATURE_NAMES[:len(features)])

    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0
    df = df[FEATURE_NAMES]

    pred = model.predict(df)
    prediction = int(pred[0])

    return jsonify({
        "prediction": prediction,
        "label": "Heart Disease" if prediction == 1 else "No Heart Disease",
        "timestamp": datetime.datetime.now().isoformat()
    })


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()
    if "batch" not in data:
        return jsonify({"error": "Missing 'batch' key"}), 400

    results = []
    for item in data["batch"]:
        df = pd.DataFrame([item], columns=FEATURE_NAMES[:len(item)])
        for col in FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0
        df = df[FEATURE_NAMES]
        pred = int(model.predict(df)[0])
        results.append({
            "prediction": pred,
            "label": "Heart Disease" if pred == 1 else "No Heart Disease"
        })

    return jsonify({
        "results": results,
        "total": len(results),
        "disease_count": sum(1 for r in results if r["prediction"] == 1),
        "timestamp": datetime.datetime.now().isoformat()
    })


FORM_HTML = """
<!DOCTYPE html>
<html><head><title>Heart Disease Predictor</title>
<style>
  body { font-family: Arial, sans-serif; max-width: 700px; margin: 50px auto;
         background: #f5f5f5; }
  .card { background: white; padding: 30px; border-radius: 10px;
          box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
  h1 { color: #d32f2f; }
  label { display: block; margin: 10px 0 5px; font-weight: bold; }
  input, select { width: 100%%; padding: 8px; border: 1px solid #ddd;
                  border-radius: 5px; box-sizing: border-box; }
  button { background: #d32f2f; color: white; padding: 12px 30px;
           border: none; border-radius: 5px; cursor: pointer;
           font-size: 16px; margin-top: 15px; }
  button:hover { background: #b71c1c; }
  .result { margin-top: 20px; padding: 15px; border-radius: 5px;
            font-size: 18px; font-weight: bold; }
  .disease { background: #ffcdd2; color: #c62828; }
  .healthy { background: #c8e6c9; color: #2e7d32; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
</style></head>
<body><div class="card">
<h1>Heart Disease Predictor</h1>
<p>Enter patient data to predict heart disease risk.</p>
<form method="POST" action="/predict-form">
<div class="grid">
  <div><label>Age</label><input name="Age" type="number" value="55"></div>
  <div><label>Sex (0=F, 1=M)</label><input name="Sex" type="number" value="1" min="0" max="1"></div>
  <div><label>BMI</label><input name="BMI" type="number" step="0.1" value="28.5"></div>
  <div><label>Smoking (0/1)</label><input name="Smoking" type="number" value="0" min="0" max="1"></div>
  <div><label>Physical Health (days)</label><input name="PhysicalHealth" type="number" value="5"></div>
  <div><label>Mental Health (days)</label><input name="MentalHealth" type="number" value="3"></div>
  <div><label>Sleep Time (hrs)</label><input name="SleepTime" type="number" step="0.5" value="7"></div>
  <div><label>Diabetic (0/1/2)</label><input name="Diabetic" type="number" value="0" min="0" max="2"></div>
  <div><label>Stroke (0/1)</label><input name="Stroke" type="number" value="0" min="0" max="1"></div>
  <div><label>Kidney Disease (0/1)</label><input name="KidneyDisease" type="number" value="0" min="0" max="1"></div>
  <div><label>General Health (0-4)</label><input name="GenHealth" type="number" value="2" min="0" max="4"></div>
  <div><label>Diff Walking (0/1)</label><input name="DiffWalking" type="number" value="0" min="0" max="1"></div>
</div>
<button type="submit">Predict</button>
</form>
{% if result is not none %}
<div class="result {{ 'disease' if result == 1 else 'healthy' }}">
  Prediction: {{ "Heart Disease Detected" if result == 1 else "No Heart Disease" }}
</div>
{% endif %}
</div></body></html>
"""


@app.route("/predict-form", methods=["GET", "POST"])
def predict_form():
    result = None
    if request.method == "POST" and MODEL_LOADED:
        form_data = {}
        for key in request.form:
            try:
                form_data[key] = float(request.form[key])
            except ValueError:
                form_data[key] = 0

        df = pd.DataFrame([form_data])
        from src.feature_engineering import engineer_brfss_features
        df = engineer_brfss_features(df)

        for col in FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0
        df = df[FEATURE_NAMES]

        pred = model.predict(df)
        result = int(pred[0])

    return render_template_string(FORM_HTML, result=result)


if __name__ == "__main__":
    port = config["serving"]["port"]
    print(f"\n[INFO] Starting Heart Disease Prediction API on port {port}")
    print(f"[INFO] Web UI: http://localhost:{port}/predict-form")
    print(f"[INFO] API docs: http://localhost:{port}/")
    app.run(host="0.0.0.0", port=port, debug=False)
