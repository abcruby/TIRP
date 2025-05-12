from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)
    firstname = db.Column(db.String(150))
    lastname = db.Column(db.String(150))
    usertype = db.Column(db.String(150))
    results = db.relationship('Result', backref='user', lazy=True)

class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255))
    predictions = db.Column(db.Text)  # store as CSV string or JSON
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
