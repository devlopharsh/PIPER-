import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, reciprocal
from constants import DATA_DIRECTORY

# An array containing all the csv files
csv_files = [f for f in os.listdir(DATA_DIRECTORY) if f.endswith('.csv')]

#region Load Data & Create Time-series Dataset
# Iterate over each file
for file in csv_files:
    file_path = os.path.join(DATA_DIRECTORY, file)
    print(f"\n-- Processing file: {file}")

    try:
        df = pd.read_csv(file_path, parse_dates=['Timestamp'])
        df.sort_values("Timestamp", inplace=True)
        df.set_index("Timestamp", inplace=True)
    except Exception as e:
        print(f"-- Error reading {file}: {e}")
        continue  # Skip to the next file if error occurs

    # Choose the target column
    target_col = "Volume_with_Variable_Leakage"
    data = df[target_col].values

    # 70% Train, 30% Test Split
    split_index = int(0.7 * len(data))
    train, test = data[:split_index], data[split_index:]

    # Scale the Data
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.reshape(-1, 1))
    test_scaled = scaler.transform(test.reshape(-1, 1))

    # Create Time-Series Dataset
    def create_dataset(X, y, time_steps=5):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            Xs.append(X[i:i + time_steps])
            ys.append(y[i + time_steps])
        return np.array(Xs), np.array(ys)

    TIME_STEPS = 5
    X_train, y_train = create_dataset(train_scaled, train_scaled, TIME_STEPS)
    X_test, y_test = create_dataset(test_scaled, test_scaled, TIME_STEPS)

    print(f"-- {file}: X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
    print(f"-- {file}: X_test shape:  {X_test.shape} | y_test shape:  {y_test.shape}")
#endregion

#region RandomizedSearchCV (Hyperparameter Tuning)
    base_svr = SVR(kernel='rbf')

    param_distributions = {
        'C': uniform(loc=1, scale=1000),
        'gamma': reciprocal(0.0001, 1),
        'epsilon': uniform(loc=0.001, scale=0.1)
    }

    random_search = RandomizedSearchCV(
        estimator=base_svr,
        param_distributions=param_distributions,
        n_iter=30,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train.reshape(X_train.shape[0], -1), y_train.ravel())

    print("\n---RandomizedSearchCV Results ---")
    print(f"-- MSE (RandomSearch): {-random_search.best_score_:.2f}")

    # Best model from random search
    best_model = random_search.best_estimator_

    # Evaluate the  Model
    def evaluate_model(model: SVR, X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray, scaler: MinMaxScaler):
        train_pred = model.predict(X_tr.reshape(X_tr.shape[0], -1))
        test_pred = model.predict(X_te.reshape(X_te.shape[0], -1))

        train_pred_inv = scaler.inverse_transform(train_pred.reshape(-1, 1))
        y_train_inv = scaler.inverse_transform(y_tr.reshape(-1, 1))
        test_pred_inv = scaler.inverse_transform(test_pred.reshape(-1, 1))
        y_test_inv = scaler.inverse_transform(y_te.reshape(-1, 1))

        train_mse = mean_squared_error(y_train_inv, train_pred_inv)
        test_mse = mean_squared_error(y_test_inv, test_pred_inv)

        return train_mse, test_mse, train_pred_inv, test_pred_inv, y_train_inv, y_test_inv

    train_mse, test_mse, train_pred_inv, test_pred_inv, y_train_inv, y_test_inv = \
        evaluate_model(best_model, X_train, y_train, X_test, y_test, scaler)

    print(f"-- {file}: Train MSE={train_mse:.2f}, Test MSE={test_mse:.2f}")
#endregion

#region Plot the Best Model's Predictions
    print(f"-- Showing plots for: {file}")

    plt.figure(figsize=(14, 5))
    plt.plot(df.index[TIME_STEPS:split_index], y_train_inv, label='Train Actual', color='green')
    plt.plot(df.index[TIME_STEPS:split_index], train_pred_inv, label='Train Predicted', color='red')
    plt.title(f"Training Predictions for {file}")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.legend()
    plt.show()

    plt.figure(figsize=(14, 5))
    start_test_idx = split_index + TIME_STEPS
    end_test_idx = split_index + TIME_STEPS + len(test_pred_inv)
    plt.plot(df.index[start_test_idx:end_test_idx], y_test_inv, label='Test Actual', color='green')
    plt.plot(df.index[start_test_idx:end_test_idx], test_pred_inv, label='Test Predicted', color='red')
    plt.title(f"Testing Predictions for {file}")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.legend()
    plt.show()
#endregion

#region Future Forecasting
    future_steps = 100
    last_sequence = X_test[-1]

    future_forecast = []
    for _ in range(future_steps):
        next_pred = best_model.predict(last_sequence.reshape(1, -1))
        future_forecast.append(next_pred[0])
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1] = next_pred

    future_forecast_inv = scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1))

    # Generate future timestamps
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date, periods=future_steps+1, freq='D')[1:]

#endregion

#region Plot for future forecasting
    print(f"-- Showing future forecast for: {file}")

    plt.figure(figsize=(14, 5))
    plt.plot(df.index, df[target_col], label='Historical Data', color='green')
    plt.plot(future_dates, future_forecast_inv, label='Future Forecast', color='blue')
    plt.xlabel('Date')
    plt.ylabel(target_col)
    plt.title(f'{target_col} Future Forecast for {file}')
    plt.legend()
    plt.show()
#endregion

