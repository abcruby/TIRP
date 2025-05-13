from flask import Flask, render_template, request, redirect, url_for
import os
import joblib
import mysql.connector
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score

# Initialize Flask app and configure upload folder
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploaded_models'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Connect to MySQL
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="malware_db"
    )

# Upload new model
@app.route("/upload_model", methods=["GET", "POST"])
def upload_model():
    if request.method == "POST":
        model_name = request.form["model_name"]
        model_file = request.files["model_file"]

        if not model_file:
            return "No file selected"

        filename = os.path.join(app.config['UPLOAD_FOLDER'], model_file.filename)
        model_file.save(filename)

        # Load the model and evaluate accuracy
        model = joblib.load(filename)
        X_test = np.load("X_test.npy")
        y_test = np.load("y_test.npy")
        y_pred = model.predict(X_test)
        accuracy = round(float(accuracy_score(y_test, y_pred) * 100), 2)

        # Insert into MySQL
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO models (name, accuracy, status, uploaded_at)
            VALUES (%s, %s, %s, %s)
        """, (model_name, accuracy, "Inactive", datetime.now()))
        conn.commit()
        cursor.close()
        conn.close()

        return redirect(url_for("model_list"))

    return render_template("upload_model.html")

# Model list display
@app.route("/models")
def model_list():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM models ORDER BY uploaded_at DESC")
    models = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template("model_list.html", models=models)

# Activate a model
@app.route("/activate/<int:model_id>")
def activate_model(model_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE models SET status='Inactive'")
    cursor.execute("UPDATE models SET status='Active' WHERE id=%s", (model_id,))
    conn.commit()
    cursor.close()
    conn.close()
    return redirect(url_for("model_list"))

# Deactivate a model
@app.route("/deactivate/<int:model_id>")
def deactivate_model(model_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE models SET status='Inactive' WHERE id=%s", (model_id,))
    conn.commit()
    cursor.close()
    conn.close()
    return redirect(url_for("model_list"))

# Delete a model
@app.route("/delete/<int:model_id>")
def delete_model(model_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM models WHERE id=%s", (model_id,))
    conn.commit()
    cursor.close()
    conn.close()
    return redirect(url_for("model_list"))

# List datasets
@app.route("/datasets")
def list_datasets():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM datasets ORDER BY uploaded_at DESC")
    datasets = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template("datasets.html", datasets=datasets)

# Detect attack types, and return results
@app.route("/predict_dataset", methods=["POST"])
def predict_dataset():
    file = request.files["dataset"]
    df = pd.read_csv(file)

    model = joblib.load("models/rf_model.pkl")
    pca = joblib.load("models/pca.pkl")
    scaler = joblib.load("models/scaler.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")

    # Drop non-numeric or unnecessary columns
    df = df.drop(columns=["attack_type", "protocol", "service", "state"], errors="ignore")
    # Apply preprocessing: scaling + PCA
    scaled = scaler.transform(df)
    X_pca = pca.transform(scaled)
    # Predict labels
    y_pred = model.predict(X_pca)
    predicted_labels = label_encoder.inverse_transform(y_pred)

    # Format results with anomaly indicator
    result = []
    for i, label in enumerate(predicted_labels):
        is_anomaly = "Yes" if label != "Normal" else "No"
        result.append({
            "record": i + 1,
            "is_anomaly": is_anomaly,
            "attack_type": label
        })

    # Count for pie chart visualization
    total_normal = sum(1 for r in result if r["is_anomaly"] == "No")
    total_malicious = len(result) - total_normal

    # Breakdown of malicious types
    malicious_types = {}
    for r in result:
        if r["is_anomaly"] == "Yes":
            malicious_types[r["attack_type"]] = malicious_types.get(r["attack_type"], 0) + 1

    pie_data = [
        {"label": "Normal", "count": total_normal},
        {
            "label": "Malicious",
            "count": total_malicious,
            "details": [{"type": k, "count": v} for k, v in malicious_types.items()]
        }
    ]

    return render_template("detect.html", result=result, pie_data=pie_data)

# Display empty detection page
@app.route("/detect", methods=["GET"])
def detect():

    return render_template("detect.html")

# Start the Flask development server
if __name__ == "__main__":
    app.run(debug=True)
