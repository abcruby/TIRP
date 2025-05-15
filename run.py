from flask import Flask, render_template, request, redirect, url_for, send_file, session, flash
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from models import db, User, Result
from admin import admin_bp

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/authentication'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
with app.app_context():
    db.create_all()

app.register_blueprint(admin_bp)

# === Load Models and Transformers ===
rf_model = joblib.load("models/rf_model.pkl")
pca = joblib.load("models/pca.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
lstm_model = load_model("models/lstm_seq_model.h5")

with open("models/feature_names.txt") as f:
    expected_features = f.read().splitlines()

latest_result = None
latest_next_label = None
latest_label_probs = None

@app.route('/')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = User.query.get(session['user_id'])
    return render_template("index.html", user=user)

@app.route('/', methods=['POST'])
def upload_predict():
    global latest_result, latest_next_label, latest_label_probs

    if 'user_id' not in session:
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    file = request.files['file']
    if not file:
        flash("No file uploaded.")
        return redirect(url_for("home"))

    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    df = df.drop(columns=['protocol', 'service', 'state', 'attack_type', 'Prediction'], errors='ignore')
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Ensure all expected features are present
    missing = [col for col in expected_features if col not in df.columns]
    if missing:
        flash(f"Missing required features: {missing}")
        return redirect(url_for("home"))
    
    df = df[expected_features]

    # Preprocessing
    X_scaled = scaler.transform(df)
    X_pca = pca.transform(X_scaled)
    y_pred = rf_model.predict(X_pca)
    predicted_labels = label_encoder.inverse_transform(y_pred)
    df['Prediction'] = predicted_labels

    if "duration" not in df.columns:
        df["duration"] = np.random.uniform(0.1, 10.0, size=len(df)).round(1)

    # Sequence Forecasting using LSTM
    if len(y_pred) >= 10:
        seq_input = y_pred[-10:].reshape(1, -1)
        predictions = lstm_model.predict(seq_input)[0]
        top_indices = np.argsort(predictions)[::-1][:5]
        top_labels = label_encoder.inverse_transform(top_indices)
        latest_next_label = top_labels[0]
        latest_label_probs = [(label, f"{predictions[i]*100:.2f}%") for label, i in zip(top_labels, top_indices)]
    else:
        latest_next_label = "N/A"
        latest_label_probs = []

    df.to_csv("predicted_results.csv", index=False)
    latest_result = df

    result = Result(
        filename=file.filename,
        predictions=df.to_csv(index=False),
        user_id=user.id,
        timestamp=datetime.utcnow(),
        next_label=latest_next_label,
        label_probs=str(latest_label_probs)
    )
    db.session.add(result)
    db.session.commit()

    return redirect(url_for('results'))

@app.route('/results')
def results():
    global latest_result, latest_next_label, latest_label_probs
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = User.query.get(session['user_id'])

    if latest_result is None:
        return redirect(url_for('home'))

    df = latest_result.copy()
    total = len(df)
    benign = (df["Prediction"] == "Normal").sum()
    malicious = total - benign

    return render_template("result.html",
                           total=total,
                           benign=benign,
                           malicious=malicious,
                           attacks=df.to_dict(orient='records'),
                           next_label=latest_next_label,
                           label_probs=latest_label_probs,
                           user=user)

@app.route('/download')
def download():
    return send_file("predicted_results.csv", as_attachment=True)

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == 'POST':
        username = request.form["username"]
        password = request.form["password"]
        firstname = request.form["firstname"]
        lastname = request.form["lastname"]

        if User.query.filter_by(username=username).first():
            flash("Username already exists.")
            return redirect(url_for("signup"))

        hashed_pw = generate_password_hash(password)
        user = User(username=username, password=hashed_pw, firstname=firstname, lastname=lastname)
        db.session.add(user)
        db.session.commit()
        flash("Signup successful. Please log in.")
        return redirect(url_for("login"))

    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            session["user_id"] = user.id
            return redirect(url_for("admin.admin_dashboard") if user.usertype == "admin" else url_for("home"))
        flash("Invalid credentials.")
    return render_template("login.html")

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully.")
    return redirect(url_for("login"))

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = User.query.get(session['user_id'])
    return render_template("profile.html", user=user)

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = User.query.get(session['user_id'])
    results = Result.query.filter_by(user_id=user.id).order_by(Result.timestamp.desc()).all()
    return render_template("history.html", results=results, user=user)

@app.route('/result/<int:result_id>')
def view_result(result_id):
    user = User.query.get(session['user_id'])

    if 'user_id' not in session:
        return redirect(url_for('login'))
    result = Result.query.get_or_404(result_id)
    if result.user_id != session['user_id']:
        flash("Unauthorized access.")
        return redirect(url_for('home'))

    from io import StringIO
    df = pd.read_csv(StringIO(result.predictions))
    total = len(df)
    benign = (df["Prediction"] == "Normal").sum()
    malicious = total - benign

    try:
        label_probs = eval(result.label_probs)
    except:
        label_probs = []

    return render_template("result.html",
                           total=total,
                           benign=benign,
                           malicious=malicious,
                           attacks=df.to_dict(orient='records'),
                           next_label=result.next_label,
                           label_probs=label_probs,
                           user=user)
@app.route('/change_password', methods=['GET', 'POST'])
def change_password():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])

    if request.method == 'POST':
        current_pw = request.form['current_password']
        new_pw = request.form['new_password']
        confirm_pw = request.form['confirm_password']

        if not check_password_hash(user.password, current_pw):
            flash('❌ Current password is incorrect.')
            return redirect(url_for('change_password'))

        if new_pw != confirm_pw:
            flash('❌ New passwords do not match.')
            return redirect(url_for('change_password'))

        user.password = generate_password_hash(new_pw)
        db.session.commit()
        flash('✅ Password updated successfully!')
        return redirect(url_for('profile'))

    return render_template('change_password.html', user=user)
if __name__ == "__main__":
    app.run(debug=True)
