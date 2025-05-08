# cccccc
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load dataset
df = pd.read_csv("train_dataset.csv")

# Encode attack_type to numerical label
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['attack_type'])

# Drop non-numeric categorical columns before scaling
df = df.drop(columns=['attack_type', 'protocol', 'service', 'state'], errors='ignore')

# Feature scaling
scaler = MinMaxScaler()
features = df.drop(columns=['label'])
X_scaled = scaler.fit_transform(features)
y = df['label'].values

# PCA transformation
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

# Train Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Behavior Sequence Prediction: Binary (0 = Normal, 1 = Malicious)
normal_index = label_encoder.transform(["Normal"])[0] if "Normal" in label_encoder.classes_ else 0
binary_labels = np.where(y == normal_index, 0, 1)

# Build sequences of length 5 to predict 6th
seq_len = 5
sequences = []
next_flags = []

for i in range(len(binary_labels) - seq_len):
    sequences.append(binary_labels[i:i+seq_len])
    next_flags.append(binary_labels[i + seq_len])

X_seq = np.array(sequences)
y_seq = np.array(next_flags)

X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

rf_seq = RandomForestClassifier(n_estimators=100, random_state=42)
rf_seq.fit(X_seq_train, y_seq_train)

import os
os.makedirs("models", exist_ok=True)

joblib.dump(rf, "models/rf_model.pkl")
joblib.dump(pca, "models/pca.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")
joblib.dump(rf_seq, "models/rf_seq_model_binary.pkl")
