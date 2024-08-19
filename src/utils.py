import os
import sys
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """Save a Python object to a file using dill."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved successfully to {file_path}")
    except Exception as e:
        logging.error(f"Error in saving object to {file_path}: {str(e)}")
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """Evaluate multiple models using GridSearchCV for hyperparameter tuning."""
    try:
        report = {}
        logging.info("Starting to evaluate models")

        for model_name, model in models.items():
            logging.info(f"Evaluating model: {model_name}")

            # Get parameters for the current model
            model_params = param.get(model_name, {})
            if not model_params:
                logging.warning(f"No parameters provided for {model_name}. Skipping GridSearchCV.")
                continue

            # Perform GridSearchCV
            gs = GridSearchCV(model, model_params, cv=3, scoring='accuracy')
            gs.fit(X_train, y_train)
            best_params = gs.best_params_
            logging.info(f"Best parameters for {model_name}: {best_params}")

            # Train model with best parameters
            model.set_params(**best_params)
            model.fit(X_train, y_train)
            logging.info(f"Model training completed for {model_name}")

            # Predict and evaluate
            y_test_pred = model.predict(X_test)
            test_model_score = accuracy_score(y_test, y_test_pred)
            logging.info(f"Model: {model_name} Test Accuracy: {test_model_score:.4f}")

            report[model_name] = test_model_score

        logging.info("Model evaluation completed")
        return report

    except Exception as e:
        logging.error(f"Error during model evaluation: {str(e)}")
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
