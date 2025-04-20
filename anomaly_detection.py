import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
cpu_devices = tf.config.list_physical_devices('CPU')
tf.config.set_visible_devices(cpu_devices)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest


# Simulate sensor data for a pump in a digester
np.random.seed(42)
n_samples = 10000

# Features (sensor readings)
vibration = np.random.normal(loc=0.5, scale=0.2, size=n_samples)
temperature = np.random.uniform(low=30, high=80, size=n_samples)
pressure = np.random.uniform(low=10, high=50, size=n_samples)
runtime_hours = np.random.exponential(scale=1000, size=n_samples)

# Inject anomalies (e.g., high vibration)
anomaly_mask = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
vibration_anomaly = np.where(
    anomaly_mask == 1,
    np.random.normal(loc=2.0, scale=0.5, size=n_samples),  # High vibration = anomaly
    vibration
)
temperature_anomaly = np.where(
    anomaly_mask == 1,
    np.random.uniform(low=60, high=80, size=n_samples),    # Sudden temp spike
    temperature
)
# Create DataFrame
df = pd.DataFrame({
    "vibration": vibration,
    "temperature": temperature,
    "pressure": pressure,
    "runtime_hours": runtime_hours,
    "anomaly": anomaly_mask  # Binary label
})



# Split data
X = df[["vibration", "temperature", "pressure", "runtime_hours"]]
y = df["anomaly"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train_normal = X_train[y_train == 0]
# Standardize features
scaler = StandardScaler()
X_train_normal_scaled = scaler.fit_transform(X_train_normal)
X_test = scaler.transform(X_test)


# Build autoencoder
input_dim = X_train_normal_scaled.shape[1]
encoding_dim = 2  # Bottleneck layer

input_layer = Input(shape=(input_dim,))
encoder = Dense(64, activation="relu")(input_layer)
encoder = Dense(32, activation="relu")(encoder)
encoder = Dense(16, activation="relu")(encoder)  # Bottleneck
decoder = Dense(32, activation="relu")(encoder)
decoder = Dense(64, activation="relu")(decoder)
decoder = Dense(input_dim, activation="sigmoid")(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

autoencoder.compile(optimizer="adam", loss="mae")
autoencoder.fit(
    X_train_normal_scaled,
    X_train_normal_scaled,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)

# Predict reconstruction error
reconstructions_train = autoencoder.predict(scaler.transform(X_train))
mse_train = np.mean(np.power(scaler.transform(X_train) - reconstructions_train, 2), axis=1)
# Add reconstruction error as a new feature
X_train_with_re = np.hstack([X_train, mse_train.reshape(-1, 1)])

X_train_combined = X_train_with_re
y_train_combined = y_train


# Split data
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_with_re, y_train,
    test_size=0.2,
    stratify=y_train,  # Preserve 80-20 split
    random_state=42
)

print("Original training data class distribution:")
print(pd.Series(y_train_combined).value_counts(normalize=True))
y_train_binary = (y_train_final > 0).astype(int)  # "Degraded" and "Imminent Failure" = 1
y_val_binary = (y_val > 0).astype(int)
# Standardize features (critical for XGBoost with MSE features)
scaler_final = StandardScaler()
X_train_final_scaled = scaler_final.fit_transform(X_train_final)
X_val_scaled = scaler_final.transform(X_val)
# Train on scaled training data (no labels needed)
model = IsolationForest(
    n_estimators=100,  # Number of trees (default=100)
    max_samples=256,   # Samples per tree (default=256)
    contamination=0.2, # Expected anomaly ratio (20% in your case)
    random_state=42
)
model.fit(X_train_final_scaled)  # Use scaled features from earlier

# Predictions: -1 = anomaly, 1 = normal
y_pred = model.predict(X_val_scaled)
y_pred = np.where(y_pred == -1, 1, 0)  # Convert to 0=Normal, 1=Anomaly
# Evaluate
print(classification_report(y_val, y_pred, target_names=["Normal", "Anomaly"]))
