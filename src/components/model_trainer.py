import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, r2_score
from xgboost import XGBClassifier

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,evaluate_models

@dataclass

class ModelTrainerConfig:
    trained_model_file_path= os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Spliting Train and test input data")
            X_train, y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )

            models = {
                'Logistic Regression': LogisticRegression(),
                'Gradient Boosting': GradientBoostingClassifier(),
                'K-Nearest Neighbors': KNeighborsClassifier(),
                'XGBoost': XGBClassifier(eval_metric='logloss')
            }
            params = {
                "Logistic Regression": {
                    'penalty': ['l1', 'l2'],
                    'C': [0.1, 1, 10],  # Reduced range for regularization strength
                    'solver': ['liblinear']  # 'liblinear' is suitable for smaller datasets and also supports 'l1' penalty
                    },
                "Gradient Boosting": {
                    'n_estimators': [50, 100],  # Fewer boosting stages to manage computation
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'subsample': [0.8, 1.0]
                    },
                "K-Nearest Neighbors": {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto'],
                    'p': [1, 2]  # Only 'manhattan' (p=1) and 'euclidean' (p=2) distances
                    },
                "XGBoost": {
                    'learning_rate': [0.01, 0.1],
                    'n_estimators': [50, 100],
                    'max_depth': [3, 6],
                    'min_child_weight': [1, 3],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0],
                    'gamma': [0, 0.1]
                }
            }


                
            



            logging.info("Loading all the models")
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            logging.info("Sorting the best model")
            best_model_score= max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model= models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No Best model found")
            logging.info(f"Best model found on both Test and Train dataset : {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Predicting the accuracy of the Best model:{best_model}")

            predicted = best_model.predict(X_test)

            accuracy_score=accuracy_score(y_test,predicted)
            
            return accuracy_score
        except Exception as e:
            CustomException(e,sys)

