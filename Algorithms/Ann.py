import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from constants import DATA_DIRECTORY
import tensorflow as tf
import keras
from keras import layers

#region Load Data and prepare Dataset
csv_files = [f for f in os.listdir(DATA_DIRECTORY) if f.endswith('.csv')]

def create_dataset(X, y, time_steps=5):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

TIME_STEPS = 5

for file in csv_files:
    file_path = os.path.join(DATA_DIRECTORY, file)
    print(f"\n-- Processing file: {file}")

    try:
        df = pd.read_csv(file_path, parse_dates=['Timestamp'])
        df.sort_values("Timestamp", inplace=True)
        df.set_index("Timestamp", inplace=True)
    except Exception as e:
        print(f"Error reading {file}: {e}")
        continue

    target_col = "Volume_with_Variable_Leakage"
    data = df[target_col].values

    # Train-test split (70% train, 30% test)
    split_index = int(0.7 * len(data))
    train, test = data[:split_index], data[split_index:]

    # Scale the data
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.reshape(-1, 1))
    test_scaled = scaler.transform(test.reshape(-1, 1))

    # Create sliding window dataset
    X_train, y_train = create_dataset(train_scaled, train_scaled, TIME_STEPS)
    X_test, y_test = create_dataset(test_scaled, test_scaled, TIME_STEPS)

    print(f"Train shape: {X_train.shape}, {y_train.shape}")
    print(f"Test shape: {X_test.shape}, {y_test.shape}")
#regionend

#region Define and train the ANN model
# Build ANN model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(TIME_STEPS,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation="linear")  # Output layer
])

model.compile(optimizer='adam', loss='mse', metrics=["mae"])

# Early stopping to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)
# Reshape X to 2D before training (samples, TIME_STEPS)
X_train_flat = X_train.reshape(X_train.shape[0], TIME_STEPS)
X_test_flat = X_test.reshape(X_test.shape[0], TIME_STEPS)
# Train the model
history = model.fit(
    X_train_flat, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test_flat, y_test),
    callbacks=[early_stopping],
    verbose=1
)



# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
#regionend
# Predict
train_pred = model.predict(X_train_flat).reshape(-1, 1)
test_pred = model.predict(X_test_flat).reshape(-1, 1)




# Inverse scale the predictions
train_pred_inv = scaler.inverse_transform(train_pred)
y_train_inv = scaler.inverse_transform(y_train)
test_pred_inv = scaler.inverse_transform(test_pred)
y_test_inv = scaler.inverse_transform(y_test)

# Print shapes to verify
print("y_train_inv shape:", y_train_inv.shape)
print("train_pred_inv shape:", train_pred_inv.shape)
print("y_test_inv shape:", y_test_inv.shape)
print("test_pred_inv shape:", test_pred_inv.shape)

# Calculate Mean Squared Error (MSE)
train_mse = mean_squared_error(y_train_inv, train_pred_inv)
test_mse = mean_squared_error(y_test_inv, test_pred_inv)

print(f"Train MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")

#regionend

#region Visualization of predictions
# Plot training predictions
plt.figure(figsize=(12, 5))
plt.plot(df.index[TIME_STEPS:split_index], y_train_inv, label='Train Actual', color='green')
plt.plot(df.index[TIME_STEPS:split_index], train_pred_inv, label='Train Predicted', color='red')
plt.title('Training Predictions')
plt.xlabel('Date')
plt.ylabel(target_col)
plt.legend()
plt.show()

# Plot testing predictions
start_test_idx = split_index + TIME_STEPS
end_test_idx = start_test_idx + len(test_pred_inv)

plt.figure(figsize=(12, 5))
plt.plot(df.index[start_test_idx:end_test_idx], y_test_inv, label='Test Actual', color='green')
plt.plot(df.index[start_test_idx:end_test_idx], test_pred_inv, label='Test Predicted', color='red')
plt.title('Testing Predictions')
plt.xlabel('Date')
plt.ylabel(target_col)
plt.legend()
plt.show()
#regionend

#region Future forecasting
# Future Forecasting
future_steps = 100
last_sequence = X_test[-1].reshape(TIME_STEPS)  # Flattened for ANN
future_forecast = []

for _ in range(future_steps):
    next_pred = model.predict(last_sequence.reshape(1, TIME_STEPS))  # 2D input
    future_forecast.append(next_pred[0, 0])
    last_sequence = np.roll(last_sequence, -1)
    last_sequence[-1] = next_pred

# Inverse transform future forecast
future_forecast_inv = scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1))

# Generate future timestamps
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date, periods=future_steps + 1, freq='D')[1:]

# Plot future forecasting
plt.figure(figsize=(12, 5))
plt.plot(df.index, df[target_col], label='Historical Data', color='blue')
plt.plot(future_dates, future_forecast_inv, label='Future Forecast', color='orange')
plt.title('Future Forecasting')
plt.xlabel('Date')
plt.ylabel(target_col)
plt.legend()
plt.show()
#regionend