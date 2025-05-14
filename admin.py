from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash
from datetime import datetime, timedelta
from models import db, User, Result

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

@admin_bp.route("/")
def admin_dashboard():
    total_users = User.query.count()
    active_users = db.session.query(Result.user_id).distinct().count()
    new_users = User.query.filter(User.created_at >= datetime.utcnow() - timedelta(days=7)).count()
    test_runs = Result.query.count()

    malware_keywords = ["Backdoor", "DoS", "Shellcode", "Worm", "Reconnaissance"]
    malware_detected = db.session.query(Result).filter(
        db.or_(*[Result.predictions.like(f"%{k}%") for k in malware_keywords])
    ).count()

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
