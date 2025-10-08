# Leak Detection in Public Utility Networks using IoT and Machine Learning

The project focuses on detecting water leakages in supply networks using **sensor data** and **machine learning models**.  
Two main tasks are addressed:

1. **Leak Classification** – detect whether a leak exists at a given time.  
2. **Volume Forecasting** – predict future water consumption volumes in cases where data is absent.  


## Repository Structure
- Algorithms/ # ML models (SVM, ANN, LSTM, GRU)
- utils/ # Data preprocessing, feature extraction, helper functions
- LeakageData # Sensor Data
- requirements.txt # Python dependencies

## Features
- **Data Preprocessing**: Aggregation, feature engineering, rolling statistics.  
- **Classification Models**: SVM, Random Forest, Artificial Neural Networks.  
- **Time Series Forecasting**: LSTM, GRU, baseline regressors.  
- **Evaluation Metrics**:  
  - Classification: Accuracy, Precision, Recall, F1, Confusion Matrix.  
- **Deployment-ready**: Models can be exported for **AWS SageMaker** or **Kubernetes**.  

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/ApoErch/Water-Leakage-Detection.git
cd Water-Leakage-Detection
```
### 2. Setup Virtual Environment & Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run ML Algorithms
#### Classifier
```bash
python -m Algorithms.Lstm
python -m Algorithms.Gru
```
