# PIPER LSTM Experiment Configuration

**Generated:** 2026-11-14 00:12:45

## Model Architecture

- **Model Type:** LSTM (PyTorch)
- **Input Dimension:** 32
- **Hidden Units:** 96
- **Number of Layers:** 2
- **Bidirectional:** False
- **Dropout:** 0.36

## Input Features

- pressure
- flow_rate
- temperature
- valve_status
- pump_state
- pump_speed
- compressor_state
- energy_consumption
- alarm_triggered
- Hour
- DayOfWeek
- Month
- pressure_roll6_mean
- pressure_roll6_std
- pressure_roll24_mean
- pressure_roll24_std
- flow_rate_roll6_mean
- flow_rate_roll6_std
- flow_rate_roll24_mean
- flow_rate_roll24_std
- temperature_roll6_mean
- temperature_roll6_std
- temperature_roll24_mean
- temperature_roll24_std
- energy_consumption_roll6_mean
- energy_consumption_roll6_std
- energy_consumption_roll24_mean
- energy_consumption_roll24_std
- pressure_delta
- flow_rate_delta
- temperature_delta
- energy_consumption_delta

## Sequence Configuration

- **Sequence Length (window):** 24
- **Stride:** 1

## Optimization

- **Optimizer:** Adam
- **Learning Rate:** 0.001
- **Weight Decay:** 1e-05
- **Batch Size:** 256
- **Max Epochs:** 10
- **Early Stopping Patience:** 4
- **Loss Function:** BCEWithLogitsLoss with pos_weight

## Data Split

- **Test Fraction:** 0.3
- **Split Method:** Chronological per segment (head=train, tail=test)
- **CV Folds:** 10 (forward-chaining)
- **Threshold Selection:** Median F1-optimal threshold from CV

## Preprocessing

- **Scaler:** StandardScaler (fitted on train only)
- **Time Features:** Hour, DayOfWeek, Month
- **Rolling Windows:** 6h, 24h (mean, std)
- **Delta Features:** First differences
- **Random Seed:** 42
- **Classification Threshold:** 0.500
