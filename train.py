import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# ==================== Load and Preprocess Dataset ====================

df = pd.read_csv("train_dataset.csv")
df.columns = df.columns.str.strip()

possible_labels = ["Label", "label", "attack_type", "Attack"]
found_label = next((col for col in df.columns if col.strip() in possible_labels), None)

if not found_label:
    raise ValueError(f"❌ No known label column found. Available columns: {df.columns.tolist()}")

df.rename(columns={found_label: "attack_type"}, inplace=True)


# Encode labels
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["attack_type"])

# Drop non-numeric or irrelevant columns
df.drop(columns=['attack_type', 'protocol', 'service', 'state'], inplace=True, errors='ignore')
df = df.replace([np.inf, -np.inf], np.nan).dropna()

features = df.drop(columns=['label'])
y = df.loc[features.index, 'label'].values

# Save feature names used during training
os.makedirs("models", exist_ok=True)
with open("models/feature_names.txt", "w") as f:
    f.write("\n".join(features.columns))

# Scale and PCA
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(features)

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# ==================== Train Random Forest ====================

rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

print("✅ Random Forest Model Performance:")
print(confusion_matrix(y_test, rf.predict(X_test)))
print(classification_report(y_test, rf.predict(X_test)))

# Save RF model and preprocessing objects
joblib.dump(rf, "models/rf_model.pkl")
joblib.dump(pca, "models/pca.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")

# ==================== Train LSTM for Sequence Forecasting ====================

seq_len = 10
sequences = []
next_labels = []

for i in range(len(y) - seq_len):
    sequences.append(y[i:i + seq_len])
    next_labels.append(y[i + seq_len])

X_seq = np.array(sequences)
y_seq = to_categorical(next_labels, num_classes=len(label_encoder.classes_))

X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

lstm_model = Sequential()
lstm_model.add(Embedding(input_dim=len(label_encoder.classes_), output_dim=64, input_length=seq_len))
lstm_model.add(LSTM(64))
lstm_model.add(Dense(len(label_encoder.classes_), activation='softmax'))

lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.fit(X_seq_train, y_seq_train, epochs=10, batch_size=32, validation_split=0.2)

lstm_model.save("models/lstm_seq_model.h5")

# ==================== Feature Importance ====================

importance_df = pd.DataFrame({
    "Feature": [f"PC{i+1}" for i in range(X_pca.shape[1])],
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

os.makedirs("static/charts", exist_ok=True)
importance_df.to_csv("static/charts/feature_importance.csv", index=False)

print("✅ All models and charts saved.")
