import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.utils import load_objext

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        pass

class CustomData:
    def __init__(self,
                 data_path,
                 model_path,
                 model_name,
                 model_type,
                 model_version,
                 model_params,
                 model_ext,
                 model_type_ext,
                 model_version_ext,
                 model_params_ext):
        self.data_path = data_path
        self.model_path = model_path
        self.model_name = model_name
        self.model_type = model_type
        self.model_version = model_version
        self.model_params = model_params

