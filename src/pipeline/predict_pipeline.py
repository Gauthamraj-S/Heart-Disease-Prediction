import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        # Placeholder for pipeline initialization
        pass

    def predict(self, features):
        try:

            model_path='artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'

            model = load_object(file_path=model_path)
            preprocessor= load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 HighBP: int,
                 HighChol: int,
                 CholCheck: int,
                 BMI: float,
                 Smoker: int,
                 Stroke: int,
                 Diabetes: int,
                 PhysActivity: int,
                 Fruits: int,
                 Veggies: int,
                 HvyAlcoholConsump: int,
                 AnyHealthcare: int,
                 NoDocbcCost: int,
                 GenHlth: int,
                 MentHlth: int,
                 PhysHlth: int,
                 DiffWalk: int,
                 Sex: int,
                 Age: int,
                 Education: int,
                 Income: int):
        self.HighBP = HighBP
        self.HighChol = HighChol
        self.CholCheck = CholCheck
        self.BMI = BMI
        self.Smoker = Smoker
        self.Stroke = Stroke
        self.Diabetes = Diabetes
        self.PhysActivity = PhysActivity
        self.Fruits = Fruits
        self.Veggies = Veggies
        self.HvyAlcoholConsump = HvyAlcoholConsump
        self.AnyHealthcare = AnyHealthcare
        self.NoDocbcCost = NoDocbcCost
        self.GenHlth = GenHlth
        self.MentHlth = MentHlth
        self.PhysHlth = PhysHlth
        self.DiffWalk = DiffWalk
        self.Sex = Sex
        self.Age = Age
        self.Education = Education
        self.Income = Income

    def get_data_as_data_frame(self):
        try:
            # Create a dictionary with all attributes of the class
            custom_data_input_dict = {
                "HighBP": [self.HighBP],
                "HighChol": [self.HighChol],
                "CholCheck": [self.CholCheck],
                "BMI": [self.BMI],
                "Smoker": [self.Smoker],
                "Stroke": [self.Stroke],
                "Diabetes": [self.Diabetes],
                "PhysActivity": [self.PhysActivity],
                "Fruits": [self.Fruits],
                "Veggies": [self.Veggies],
                "HvyAlcoholConsump": [self.HvyAlcoholConsump],
                "AnyHealthcare": [self.AnyHealthcare],
                "NoDocbcCost": [self.NoDocbcCost],
                "GenHlth": [self.GenHlth],
                "MentHlth": [self.MentHlth],
                "PhysHlth": [self.PhysHlth],
                "DiffWalk": [self.DiffWalk],
                "Sex": [self.Sex],
                "Age": [self.Age],
                "Education": [self.Education],
                "Income": [self.Income]
            }

            # Convert the dictionary to a DataFrame
            df = pd.DataFrame(custom_data_input_dict)
            return df

        except Exception as e:
            raise CustomException(e, sys)
