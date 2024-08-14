import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
import lightgbm as lgb
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def _get_models(self):
        return {
            'Logistic Regression': LogisticRegression(),
            'Gradient Boosting': GradientBoostingClassifier(),
            'AdaBoost': AdaBoostClassifier(),
            'CatBoost': CatBoostClassifier(silent=True),
            'LightGBM': lgb.LGBMClassifier()
        }

    def _get_params(self):
        return {
            "Logistic Regression": {
                'penalty': ['l1', 'l2'],
                'C': [0.1, 1, 10],
                'solver': ['liblinear']
            },
            "Gradient Boosting": {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'subsample': [0.8, 1.0]
            },
            "AdaBoost": {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1]
            },
            "CatBoost": {
                'iterations': [100, 200],
                'learning_rate': [0.01, 0.1],
                'depth': [3, 5, 7]
            },
            "LightGBM": {
                'num_leaves': [31, 63],
                'learning_rate': [0.01, 0.1],
                'n_estimators': [50, 100],
                'max_depth': [10, 20]
            }
        }

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting Train and Test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = self._get_models()
            params = self._get_params()

            logging.info("Loading all the models")
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            logging.info("Sorting the best model")
            best_model_score = max(model_report.values())
            best_model_name = [name for name, score in model_report.items() if score == best_model_score][0]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No Best model found")

            logging.info(f"Best model found: {best_model_name}")

            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
            logging.info(f"Model saved at {self.model_trainer_config.trained_model_file_path}")

            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)
            logging.info(f"Accuracy of the best model: {accuracy}")

            return accuracy

        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            raise CustomException(e, sys)
