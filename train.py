import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# ========== Load dataset ==========
df = pd.read_csv("train_dataset.csv")
df.columns = df.columns.str.strip()  # Clean headers

# ========== Detect label column ==========
possible_label_columns = ["attack_type", "Label", "label", "Attack Label", "Attack", "Target"]
label_col = next((col for col in df.columns if col in possible_label_columns), None)
if not label_col:
    raise ValueError("❌ No label column found. Tried: " + ", ".join(possible_label_columns))

label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df[label_col])
df.drop(columns=[label_col], inplace=True)

# ========== Drop unnecessary columns ==========
df.drop(columns=['protocol', 'service', 'state'], errors='ignore', inplace=True)

# ========== Clean and prepare data ==========
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

X = df.drop(columns=['label'])
y = df['label']

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# PCA to reduce dimension
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# ========== Train-test split ==========
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

# ========== Train Random Forest ==========
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# ========== Evaluate ==========
y_pred = rf.predict(X_test)
print("\n✅ Random Forest Model Performance:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ========== Train LSTM for next label prediction ==========
seq_len = 5
sequences = []
next_labels = []

y_full = y.values
for i in range(len(y_full) - seq_len):
    sequences.append(y_full[i:i + seq_len])
    next_labels.append(y_full[i + seq_len])

X_seq = np.array(sequences)
y_seq = np.array(next_labels)

num_classes = len(np.unique(y_seq))
y_seq_cat = to_categorical(y_seq, num_classes=num_classes)

# Reshape for LSTM
X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))

X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(X_seq, y_seq_cat, test_size=0.2, random_state=42)

lstm_model = Sequential([
    LSTM(64, input_shape=(seq_len, 1)),
    Dense(num_classes, activation='softmax')
])
lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_seq_train, y_seq_train, epochs=10, batch_size=32, verbose=1)

# ========== Save Models ==========
os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/rf_model.pkl")
joblib.dump(pca, "models/pca.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")
lstm_model.save("models/lstm_forecast_model.h5")

# ========== Save Feature Importance ==========
importances = rf.feature_importances_
pca_feature_names = [f"PC{i+1}" for i in range(len(importances))]

feature_df = pd.DataFrame({
    "Feature": pca_feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

os.makedirs("static/charts", exist_ok=True)
feature_df.to_csv("static/charts/feature_importance.csv", index=False)

print("✅ All models trained and saved.")
