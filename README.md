# Heart Disease Prediction

## Overview

This project implements an end-to-end machine learning pipeline for predicting heart disease using various classification models. The pipeline includes data exploration, preprocessing, model training, and evaluation. The models tested include Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, K-Nearest Neighbors, and XGBoost. The CatBoost model yielded the best performance.

## Dataset

The dataset used for this project is the "Heart Disease Health Indicators" dataset from Kaggle. You can access the dataset [here](https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset).

## Project Structure

- `notebooks/`
  - `HEART_DISEASE_PREDICTION.ipynb`: This notebook includes Exploratory Data Analysis (EDA), including data cleaning, visualization, and feature engineering.
  - `Model_Training.ipynb`: This notebook contains the model training and evaluation process for various classifiers.

- `src/`
  - `components/`
    - `data_ingestion.py`: Handles data loading and ingestion.
    - `data_transformation.py`: Includes data preprocessing and feature engineering.
    - `model_trainer.py`: Contains the training logic for various classification models.
  - `pipeline/`
    - `predict_pipeline.py`: Manages the prediction pipeline for inference.
    - `train_pipeline.py`: Orchestrates the end-to-end training process.
  - `templates/`
    - `home.html`: Home page of the web application.
    - `index.html`: Index page of the web application.
    - `styles.css`: CSS styles for the web application.
  - `logger.py`: Provides logging functionality for tracking and debugging.
  - `exception.py`: Defines custom exceptions for error handling.
  - `utils.py`: Contains utility functions used throughout the project.
- `app.py`: Flask application entry point.
- `setup.py`: Script for setting up the package and dependencies.
- `requirements.txt`: Lists the Python packages required for running the notebooks, scripts, and Flask application.

## Models

The following models were implemented and evaluated:

- **Logistic Regression**: `LogisticRegression()`
  - **Parameters**:
    - `penalty`: ['l1', 'l2']
    - `C`: [0.1, 1, 10]
    - `solver`: ['liblinear']

- **Gradient Boosting**: `GradientBoostingClassifier()`
  - **Parameters**:
    - `n_estimators`: [50, 100]
    - `learning_rate`: [0.01, 0.1]
    - `max_depth`: [3, 5]
    - `min_samples_split`: [2, 5]
    - `min_samples_leaf`: [1, 2]
    - `subsample`: [0.8, 1.0]

- **AdaBoost**: `AdaBoostClassifier()`
  - **Parameters**:
    - `n_estimators`: [50, 100]
    - `learning_rate`: [0.01, 0.1]

- **CatBoost**: `CatBoostClassifier(silent=True)`
  - **Parameters**:
    - `iterations`: [100, 200]
    - `learning_rate`: [0.01, 0.1]
    - `depth`: [3, 5, 7]

- **LightGBM**: `lgb.LGBMClassifier()`
  - **Parameters**:
    - `num_leaves`: [31, 63]
    - `learning_rate`: [0.01, 0.1]
    - `n_estimators`: [50, 100]
    - `max_depth`: [10, 20]

The best-performing model was CatBoost with an accuracy o


## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Gauthamraj-S/Heart-Disease-Prediction.git
   cd heart-disease-prediction
   ```

2. **Install dependencies**:
   It is recommended to use a virtual environment. You can install the required packages using `pip`:
   ```bash
   pip install -r requirements.txt
   ```


 [Kaggle](https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset) 

## Running the Notebooks

1. Open the notebooks using Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook notebooks/HEART_DISEASE_PREDICTION.ipynb
   jupyter notebook notebooks/Model_Training.ipynb
   ```

2. Follow the instructions in the notebooks to perform EDA and model training.

## Running the Pipeline

1. **Data Ingestion and Transformation**: Run the data ingestion and transformation scripts to prepare the data.
   ```bash
   python src/components/data_ingestion.py
   python src/components/data_transformation.py
   ```

2. **Prediction**: Use the prediction pipeline for making predictions on new data.
   ```bash
   python src/components/predict_pipeline.py
   ```

## Results

The CatBoost model achieved the highest accuracy of `0.9016`. For more details on model performance and comparisons, please refer to the `Model_Training.ipynb` notebook.

## Contributing

Feel free to contribute to this project by submitting issues or pull requests. For any questions or feedback, please contact sgauthamraj4@gmail.com

