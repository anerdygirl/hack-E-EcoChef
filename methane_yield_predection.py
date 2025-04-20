import pandas as pd
import numpy as np
import tensorflow as tf
cpu_devices = tf.config.list_physical_devices('CPU')
tf.config.set_visible_devices(cpu_devices)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load data
df = pd.read_csv("data.csv")

# Example features and labels
X_reg = df[["temperature", "pH", "moisture_content", "food_waste_ratio"]]
y_reg = df["methane_yield"].values

X_cls = df[["temperature", "pH", "moisture_content"]]
y_cls = df["food_waste_ratio"].values  # For classification (encoded as integers)

preprocessor_reg = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["temperature", "pH", "moisture_content"]),
        ("cat", OneHotEncoder(), ["food_waste_ratio"])
    ]
)

X_reg = preprocessor_reg.fit_transform(X_reg)

scaler_cls = StandardScaler()
X_cls = scaler_cls.fit_transform(X_cls)
# Split data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)
# Define model
model_reg = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_reg.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(1, activation='linear')
])

# Compile model
model_reg.compile(
    optimizer='adam',
    loss='mse',  # Mean Squared Error for regression
    metrics=['mae']
)

# Train model
history_reg = model_reg.fit(
    X_train_reg, y_train_reg,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    verbose=1
)
# One-hot encode labels
y_train_cls = to_categorical(y_train_cls)
y_test_cls = to_categorical(y_test_cls)
# Classification model
model_cls = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_cls.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(y_train_cls.shape[1], activation='softmax')
])

model_cls.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
# Train
history_cls = model_cls.fit(
    X_train_cls, y_train_cls,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    verbose=1
)

# Regression evaluation
mse, mae = model_reg.evaluate(X_test_reg, y_test_reg)
print(f"Regression MSE: {mse:.4f}, MAE: {mae:.4f}")

# Classification evaluation
loss, accuracy = model_cls.evaluate(X_test_cls, y_test_cls)
print(f"Classification Accuracy: {accuracy:.4f}")


# Predict methane yield for new sample (regression)
# Regression prediction
new_sample_reg = pd.DataFrame(
    [[37.5, 7.2, 88.0, 0]], 
    columns=["temperature", "pH", "moisture_content", "food_waste_ratio"]
)
new_sample_reg_preprocessed = preprocessor_reg.transform(new_sample_reg)
predicted_yield = model_reg.predict(new_sample_reg_preprocessed)
print(f"Predicted Methane Yield: {predicted_yield[0][0]:.4f} m³/kg VS")

# Classification prediction
test_samples = [
    [35.0, 7.0, 85.0],   # Lower temp, neutral pH, moderate moisture
    [40.0, 7.5, 90.0],   # Higher temp, optimal pH, high moisture
    [30.0, 6.8, 82.0]    # Edge case: low temp, low pH, low moisture
]
for sample in test_samples:
    df_sample = pd.DataFrame([sample], columns=["temperature", "pH", "moisture_content"])
    X_sample = scaler_cls.transform(df_sample)
    prediction = model_cls.predict(X_sample)
    print(f"Input: {sample} → Predicted Ratio: {np.argmax(prediction)}")

#new_sample_cls_preprocessed = scaler_cls.transform(new_sample_cls)
#predicted_ratio = model_cls.predict(new_sample_cls_preprocessed)
#print(f"Predicted Optimal Ratio: {np.argmax(predicted_ratio)}")