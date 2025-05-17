from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash
import os
import subprocess
from datetime import datetime, timedelta
from models import db, User, Result
from models import ModelTrainingLog
from sqlalchemy import or_
import pandas as pd
from werkzeug.utils import secure_filename

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

@admin_bp.route("/")
def admin_dashboard():
    latest_log = ModelTrainingLog.query.order_by(ModelTrainingLog.timestamp.desc()).first()
    total_users = User.query.count()
    active_users = db.session.query(Result.user_id).distinct().count()
    new_users = User.query.filter(User.created_at >= datetime.utcnow() - timedelta(days=7)).count()
    test_runs = Result.query.count()

    # Malware keyword detection
    malware_keywords = ["Backdoor", "DoS", "Shellcode", "Worm", "Reconnaissance"]
    malware_detected = db.session.query(Result).filter(
        or_(*[Result.predictions.like(f"%{k}%") for k in malware_keywords])
    ).count()

    # Analyze recent results for top malware
    recent_results = db.session.query(Result.predictions).order_by(Result.timestamp.desc()).limit(50).all()
    malware_counter = {}
    for row in recent_results:
        for mtype in malware_keywords:
            if mtype in row[0]:
                malware_counter[mtype] = malware_counter.get(mtype, 0) + 1

    top_malware_list = sorted(malware_counter.items(), key=lambda x: x[1], reverse=True)[:5]

    logs_by_dataset = db.session.query(
        ModelTrainingLog.filename,
        db.func.avg(ModelTrainingLog.accuracy).label('avg_accuracy'),
        db.func.count(ModelTrainingLog.id).label('runs')
    ).group_by(ModelTrainingLog.filename).all()
    logs = ModelTrainingLog.query.order_by(ModelTrainingLog.timestamp.asc()).all()
    dataset_labels = [log.filename for log in logs if log.accuracy is not None]
    accuracy_values = [log.accuracy for log in logs if log.accuracy is not None]
    return render_template("admin.html",
                           total_users=total_users,
                           active_users=active_users,
                           new_users=new_users,
                           test_runs=test_runs,
                           malware_detected=malware_detected,
                           top_malware=top_malware_list,
                           logs_by_dataset=logs_by_dataset,
                           model_accuracy=latest_log.accuracy if latest_log else None,
                           precision=latest_log.precision if latest_log else None,
                           recall=latest_log.recall if latest_log else None,
                           f1_score=latest_log.f1_score if latest_log else None,
                           accuracy_values=accuracy_values,
                           dataset_labels=dataset_labels)

@admin_bp.route("/users")
def manage_users():
    if not session.get("user_id"):
        return redirect(url_for("login"))

    current_user = User.query.get(session["user_id"])
    if current_user.usertype != "admin":
        flash("Unauthorized access.")
        return redirect(url_for("index"))

    users = User.query.all()
    return render_template("admin_users.html", users=users)
@admin_bp.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    return render_template('admin_profile.html', user=user)

@admin_bp.route("/users/add", methods=["GET", "POST"])
def add_user():
    if not session.get("user_id"):
        return redirect(url_for("login"))

    current_user = User.query.get(session["user_id"])
    if current_user.usertype != "admin":
        flash("Unauthorized access.")
        return redirect(url_for("index"))

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        firstname = request.form["firstname"]
        lastname = request.form["lastname"]
        usertype = request.form.get("usertype", "user")

        if User.query.filter_by(username=username).first():
            flash("Username already exists.")
            return redirect(url_for("admin.add_user"))

        hashed_pw = generate_password_hash(password)
        new_user = User(
            username=username,
            password=hashed_pw,
            firstname=firstname,
            lastname=lastname,
            usertype=usertype
        )
        db.session.add(new_user)
        db.session.commit()
        flash("User added successfully.")
        return redirect(url_for("admin.manage_users"))
    

    return render_template("admin_add_user.html")
@admin_bp.route('/upload-dataset', methods=['GET', 'POST'])
def upload_dataset():
    if request.method == 'POST':
        file = request.files.get('dataset')
        if not file:
            flash("No file selected.", "danger")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        upload_path = os.path.join("datasets", filename)
        os.makedirs("datasets", exist_ok=True)
        file.save(upload_path)

# Save the filename for training
        with open("datasets/last_filename.txt", "w") as f:
            f.write(filename)


        # Trigger training script
        try:
            subprocess.run(["python3", "train.py"], check=True)
            flash("✅ Model retrained successfully.", "success")
        except subprocess.CalledProcessError:
            flash("❌ Error occurred during training.", "danger")

        return redirect(url_for("admin.upload_dataset"))
    

    return render_template("upload_dataset.html")