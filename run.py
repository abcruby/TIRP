from flask import Flask, render_template, request, redirect, url_for, send_file, session, flash
import pandas as pd
import numpy as np
import os
import joblib
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from models import db, User, Result

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.secret_key = 'supersecretkey'

# MySQL connection using root and blank password
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/authentication'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

with app.app_context():
    db.create_all()

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        firstname = request.form['firstname']
        lastname = request.form['lastname']

        if User.query.filter_by(username=username).first():
            flash('Username already exists.')
            return redirect(url_for('signup'))

        hashed_pw = generate_password_hash(password)
        new_user = User(username=username, password=hashed_pw, firstname=firstname, lastname=lastname)
        db.session.add(new_user)
        db.session.commit()
        flash('Signup successful! Please login.')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        next_page = request.args.get('next') or url_for('index')

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            return redirect(next_page)
        else:
            flash('Invalid credentials.')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.')
    return redirect(url_for('login'))

# Load models
rf_model = joblib.load("models/rf_model.pkl")
pca = joblib.load("models/pca.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
seq_model = joblib.load("models/rf_seq_model_binary.pkl")

# Store latest result
latest_result = None
latest_next_label = None
latest_label_probs = None

@app.route("/", methods=["GET", "POST"])
def index():
    global latest_result, latest_next_label, latest_label_probs

    if 'user_id' not in session:
        return redirect(url_for('login', next=request.path))

    if request.method == "POST":
        file = request.files["file"]
        if not file:
            return "No file uploaded"

        df = pd.read_csv(file)
        df = df.drop(columns=['protocol', 'service', 'state', 'attack_type'], errors='ignore')

        # Scale + PCA
        X_scaled = scaler.transform(df)
        X_pca = pca.transform(X_scaled)

        # Predict attack type
        y_pred = rf_model.predict(X_pca)
        labels = label_encoder.inverse_transform(y_pred)
        df["Prediction"] = labels

        # Add duration if missing
        if "duration" not in df.columns:
            df["duration"] = np.random.uniform(0.1, 10.0, size=len(df)).round(1)

        # Binary label: 0 (Normal), 1 (Malicious)
        normal_index = label_encoder.transform(["Normal"])[0]
        binary_labels = np.where(y_pred == normal_index, 0, 1)

        # Predict next step if enough data
        if len(binary_labels) >= 5:
            seq_input = binary_labels[-5:].reshape(1, -1)
            next_probs = seq_model.predict_proba(seq_input)[0]
            next_top_indices = np.argsort(next_probs)[::-1][:5]
            next_labels = label_encoder.inverse_transform(next_top_indices)
            latest_next_label = label_encoder.inverse_transform([seq_model.predict(seq_input)[0]])[0]
            latest_label_probs = [(label, f"{next_probs[i]*100:.2f}%") for i, label in zip(next_top_indices, next_labels)]
        else:
            latest_next_label = "N/A (Too few records)"
            latest_label_probs = []

        df.to_csv("predicted_results.csv", index=False)
        latest_result = df

        # Save result to DB
        result_record = Result(
            filename=file.filename,
            predictions=df.to_csv(index=False),
            user_id=session['user_id']
        )
        db.session.add(result_record)
        db.session.commit()

        return redirect(url_for("results"))

    return render_template("index.html")

@app.route("/results")
def results():
    global latest_result, latest_next_label, latest_label_probs

    if latest_result is not None:
        df = latest_result.copy()

        # Summary counts
        total = len(df)
        malicious = (df["Prediction"] != "Normal").sum()
        benign = (df["Prediction"] == "Normal").sum()

        return render_template("result.html",
                               total=total,
                               malicious=malicious,
                               benign=benign,
                               attacks=df.to_dict(orient='records'),
                               next_label=latest_next_label,
                               label_probs=latest_label_probs)
    else:
        return redirect(url_for("index"))

@app.route("/download")
def download():
    return send_file("predicted_results.csv", as_attachment=True)

@app.route("/history")
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_results = Result.query.filter_by(user_id=session['user_id']).order_by(Result.timestamp.desc()).all()
    return render_template("history.html", results=user_results)

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

    return render_template('change_password.html')

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    return render_template('profile.html', user=user)


@app.route("/result/<int:result_id>")
def view_result(result_id):
    result = Result.query.filter_by(id=result_id, user_id=session['user_id']).first_or_404()
    from io import StringIO
    df = pd.read_csv(StringIO(result.predictions))

    total = len(df)
    malicious = (df["Prediction"] != "Normal").sum()
    benign = (df["Prediction"] == "Normal").sum()

    return render_template("result.html",
                           total=total,
                           malicious=malicious,
                           benign=benign,
                           attacks=df.to_dict(orient='records'),
                           next_label="N/A",
                           label_probs=[])  # You can improve by storing those too

if __name__ == "__main__":
    app.run(debug=True)
