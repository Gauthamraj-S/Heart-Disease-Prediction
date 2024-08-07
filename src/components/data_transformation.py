import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object



@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['HeartDiseaseorAttack',
                                 'HighBP',
                                 'HighChol',
                                 'CholCheck',
                                 'BMI',
                                 'Smoker',
                                 'Stroke',
                                 'Diabetes',
                                 'PhysActivity',
                                 'Fruits',
                                 'Veggies',
                                 'HvyAlcoholConsump',
                                 'AnyHealthcare',
                                 'NoDocbcCost',
                                 'GenHlth',
                                 'MentHlth',
                                 'PhysHlth',
                                 'DiffWalk',
                                 'Sex',
                                 'Age',
                                 'Education',
                                 'Income'
            ]
            num_pipeline= Pipeline(
                steps=[
                    ("scalar",StandardScaler())
                ]
            )
            
            logging.info(f"Numerical Coulmns: {numerical_columns}")
            logging.info("Numerical coulmn Encoding completed")

            return num_pipeline
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):

        try:
            logging.info("Entered Initiate data transformation")
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read test train data completed")

            logging.info("Obtaining preprocessing object")

            preprocessor_obj= self.get_data_transformer_object()

            target_column_name='HeartDiseaseorAttack'

            numerical_columns=['HeartDiseaseorAttack', 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'Diabetes', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']
            
            input_feature_train_df= train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df= test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.fit_transform(input_feature_test_df)

            train_arr= np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr= np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path,

            )
        except Exception as e:
            raise CustomException(e,sys)
        

