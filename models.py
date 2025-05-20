from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()
class ModelTrainingLog(db.Model):
    __tablename__ = 'model_training_logs'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    accuracy = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    def __repr__(self):
        return f"<TrainingLog {self.dataset_name}>"
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)
    firstname = db.Column(db.String(150))
    lastname = db.Column(db.String(150))
    usertype = db.Column(db.String(150))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    results = db.relationship('Result', backref='user', lazy=True)
    def __repr__(self):
        return f'<User {self.username}>'
class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    predictions = db.Column(db.Text(length=(2**32)-1), nullable=False) 
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    next_label = db.Column(db.String(100), nullable=True)
    label_probs = db.Column(db.Text, nullable=True)  # JSON-formatted string of probabilities

    def is_malware_detected(self):
        return any(attack != "Normal" for attack in self.get_attack_labels())

    def get_attack_labels(self):
        import pandas as pd
        from io import StringIO

        try:
            df = pd.read_csv(StringIO(self.predictions))
            return df["Prediction"].dropna().unique().tolist()
        except Exception:
            return []

    def __repr__(self):
        return f"<Result {self.filename} by user {self.user_id}>"