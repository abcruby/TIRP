from flask import Flask, render_template, request, redirect, url_for
import os
import joblib
import mysql.connector
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score
from sqlalchemy import text, func
from models import db, User, Result

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/authentication'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploaded_models'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db.init_app(app)

# ========== Admin Dashboard ==========
@app.route("/admin")
def admin_dashboard():
    with app.app_context():
        # Total users
        total_users = User.query.count()

        # Active users = users who have at least 1 prediction result
        active_users = db.session.query(Result.user_id).distinct().count()

        # New users in last 7 days
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        new_users = User.query.filter(User.created_at >= seven_days_ago).count()

        # Total test runs
        test_runs = Result.query.count()

        # Malware detected (if Prediction column in saved CSV includes malicious label)
        malware_keywords = ["Backdoor", "DoS", "Shellcode", "Worm", "Reconnaissance"]
        malware_detected = db.session.query(Result).filter(
            db.or_(*[Result.predictions.like(f"%{k}%") for k in malware_keywords])
        ).count()

        # Top 5 malware types from the last 50 results
        recent_results = db.session.query(Result.predictions).order_by(Result.timestamp.desc()).limit(50).all()
        malware_counter = {}

        for row in recent_results:
            for mtype in malware_keywords:
                if mtype in row[0]:
                    malware_counter[mtype] = malware_counter.get(mtype, 0) + 1

        top_malware_list = sorted(malware_counter.items(), key=lambda x: x[1], reverse=True)[:5]

        return render_template("admin.html",
                               total_users=total_users,
                               active_users=active_users,
                               new_users=new_users,
                               test_runs=test_runs,
                               malware_detected=malware_detected,
                               top_malware=top_malware_list)

if __name__ == "__main__":
    app.run(debug=True)
