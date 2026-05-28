# Production Churn Prediction ML System

A production-oriented modular machine learning system for customer churn prediction designed with real-world ML engineering principles.

This project focuses on:
- scalable ML pipeline architecture
- clean separation of concerns
- defensive programming
- modular training/evaluation systems
- reproducible ML workflows
- production engineering mindset

# Project Architecture
    text
Ingestion Layer
      ↓
Preprocessing Layer
      ↓
Training Layer
      ↓
Evaluation Layer
      ↓
Inference Ready System

# Current Features

## Data Layer
- Structured ingestion pipeline
- Schema validation
- Data integrity checks

## Preprocessing Layer
- Feature preprocessing pipeline
- Missing value handling
- Data transformation support

## Training Layer
- Modular trainer architecture
- Validation split system
- Logistic Regression baseline
- Config-driven model initialization
- Model serialization

## Evaluation Layer
- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix

## Monitoring
- Centralized logging system
- Training/evaluation logs
- Defensive error handling

# Project Structure

    text
churn-ml-system/
│
├── app/
├── configs/
├── logs/
├── models/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── inference/
│   ├── models/
│   │   ├── trainer.py
│   │   └── evaluate.py
│   │
│   ├── monitoring/
│   ├── pipeline/
│   ├── services/
│   └── utils/
│
├── tests/
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md

# Tech Stack

- Python
- Scikit-learn
- Pandas
- NumPy
- XGBoost
- FastAPI
- Docker

# Production Engineering Concepts Applied

- modular ML architecture
- separation of concerns
- defensive programming
- validation-first design
- artifact serialization
- structured logging
- config-driven workflows
- reproducible training pipelines

# Future Improvements

- Cross-validation system
- Hyperparameter tuning
- Experiment tracking
- MLflow integration
- Drift monitoring
- CI/CD integration
- Model registry
- Deployment pipeline

# How to Run

## Install dependencies
bash
pip install -r requirements.txt


## Run project

bash
python main.py

# Docker

## Build Docker image

bash
docker build -t churn-ml-system .


## Run container

bash
docker run churn-ml-system

# Goal of This Project
The objective of this project is to build a scalable and production-oriented churn prediction system that reflects real-world ML engineering practices, including modular pipeline design, reproducible model training, structured evaluation workflows, logging, validation, and deployment-ready architecture.


# Goal of This Project

The goal of this project is to move beyond notebook-style machine learning and design systems using production-grade ML engineering principles used in real-world industry environments.
