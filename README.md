## Table of Contents

- [Leak Detection in Public Utility Networks using IoT and Machine Learning](#leak-detection-in-public-utility-networks-using-iot-and-machine-learning)
  - [Repository Structure](#repository-structure)
  - [Features](#features)
  - [Getting Started](#getting-started)
    - [1. Clone the Repository](#1-clone-the-repository)
    - [2. Setup Virtual Environment \& Install Dependencies](#2-setup-virtual-environment--install-dependencies)
    - [3. Run ML Algorithms](#3-run-ml-algorithms)
      - [Classifier](#classifier)
- [AWS SageMaker — Deployment Guide (No-Docker Deployment)](#aws-sagemaker--deployment-guide-no-docker-deployment)
  - [Prerequisites](#prerequisites)
  - [Deployment Tools Overview](#deployment-tools-overview)
    - [1. `tools/sm_deploy_endpoint.py`](#1-toolssm_deploy_endpointpy)
    - [2. `tools/sm_invoke_demo.py`](#2-toolssm_invoke_demopy)
      - [What It Does](#what-it-does)


# Leak Detection in Public Utility Networks using IoT and Machine Learning

The project focuses on detecting water leakages in supply networks using **sensor data** and **machine learning models**.  
Two main tasks are addressed:

1. **Leak Classification** – detect whether a leak exists at a given time.  
2. **Serving & Deployment** – packaging trained artifacts and deploying on **AWS SageMaker**.  


## Repository Structure
- `Algorithms/` # ML models (SVM, ANN, LSTM, GRU, RF, XGBoost)
- `LeakageData/` # Sensor Data
- `models/artifacts/` → inference-time bundles (weights, scaler, decision threshold, metadata).
- `server/` → lightweight serving app (FastAPI/uvicorn style) + its **serving-only** requirements.
- `tools/` → scripts to **deploy** and **invoke** a SageMaker endpoint.
- `utils/` → data prep, CV, argument parsing, constants.
- `requirements.txt` # Python dependencies

## Features
- **Data Preprocessing**: Aggregation, feature engineering, rolling statistics, missing data handling.  
- **Classification Models**: SVM, Random Forest, XGBoost, Artificial Neural Networks.  
- **Time Series Forecasting**: LSTM, GRU, baseline regressors.  
- **Evaluation Metrics**:  
  - Classification: Accuracy, Precision, Recall, F1, Confusion Matrix.  
- **Deployment-ready**: Models can be exported for **AWS SageMaker**.  

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

# AWS SageMaker — Deployment Guide (No-Docker Deployment)

- You **do not** build any container. You can upload your `model.tar.gz` to S3 and deploy with SageMaker’s **prebuilt PyTorch** image (Script Mode).
- The steps below package your **trained classifier artifacts** and deploy a real-time HTTPS endpoint on **AWS SageMaker**.

##  Prerequisites

```bash
aws configure  # set region (e.g., eu-central-1) and credentials
```
- Have an IAM Role ARN with SageMaker permissions, e.g.: `arn:aws:iam::<ACCOUNT_ID>:role/SageMakerExecutionRole`
- Package model artifacts
```bash
tar -czf models/model.tar.gz \
-C models/artifacts . \
-C ../<inference directory> inference.py
```

## Deployment Tools Overview

The repository includes two helper scripts in the `tools/` directory for automating **AWS SageMaker deployment** and **endpoint testing**.


### 1. `tools/sm_deploy_endpoint.py`

This script **uploads** your packaged model (`model.tar.gz`) to S3 and **creates a SageMaker endpoint** using AWS’s prebuilt PyTorch serving container.

**Main steps performed:**
1. **Upload artifacts to S3**
   - Takes your `model.tar.gz` (which must include trained model files).
   - Uploads it to a designated S3 bucket/prefix.
2. **Configure the SageMaker model**
   - Uses `PyTorchModel` from the SageMaker SDK.
   - Automatically sets up the environment (e.g., `THR_OVERRIDE` if you want to override the leak detection threshold).
3. **Deploy the endpoint**
   - Can deploy either as:
     - A **provisioned instance** (`ml.m5.large`, etc.) for steady workloads.
     - Or a **serverless endpoint** (on-demand, scales to zero) using `--serverless`.
4. **Prints the created endpoint name** when successful.

**Usage example:**
```bash
python tools/sm_deploy_endpoint.py \
--model_tar models/model.tar.gz \
--s3_bucket my-ml-bucket \
--role_arn arn:aws:iam::<ACCOUNT_ID>:role/SageMakerExecutionRole \
--endpoint_name leak-gru-endpoint \
--region eu-central-1 \
--serverless
```


### 2. `tools/sm_invoke_demo.py`

This script is used to **test and validate** your deployed SageMaker endpoint end-to-end.

After you’ve deployed your leak detection model using `tools/sm_deploy_endpoint.py`, this tool sends a few **sample data windows** (sensor readings) to the endpoint and prints the predictions.

####  What It Does

1. **Loads model metadata**  
   - Reads the file `gru_cls_meta.json` from `models/artifacts/` to extract:
     - The **feature order** used during training.
     - The **window size** (number of timesteps per prediction).
     - The **threshold** used to classify leak vs. no-leak.

2. **Loads frozen test data**  
   - Opens a Parquet test file (e.g., `test_agg.parquet`) that contains sensor readings identical to the format used during training.
   - Ensures feature columns, meter IDs, and timestamps are consistent.

3. **Builds sliding windows**  
   - Converts time-series sensor data into 3D input arrays:  
     `[num_windows, window_length, num_features]`.  
   - Each window represents a short historical sequence of sensor readings for one meter.

4. **Selects a small, balanced sample**  
   - Randomly picks a few windows (using `--n`) with both leak (`1`) and no-leak (`0`) cases.
   - This helps verify that the endpoint is returning meaningful results.

5. **Invokes the SageMaker endpoint**  
   - Uses the AWS **boto3** client (`sagemaker-runtime`) to send a JSON payload containing:
     - the input windows (`windows`)
     - the feature names (`feature_cols`)
     - and an optional threshold override (`thr`)
   - Calls the endpoint you specify via `--endpoint_name`.

6. **Prints results clearly**  
   - Displays:
     - `Meter_ID`
     - `Timestamp_end` (last timestamp in the window)
     - `y_true` (ground truth label)
     - `prob` (predicted leak probability)
     - `pred` (final leak prediction)
   - Calculates the **sample accuracy** for the test windows.


**Usage example:**
```bash
python tools/sm_invoke_demo.py \
--endpoint_name leak-gru-endpoint \
--region eu-central-1 \
--meta_path models/artifacts/gru_cls_meta.json \
--test_path models/artifacts/test_agg.parquet \
--n 4
```