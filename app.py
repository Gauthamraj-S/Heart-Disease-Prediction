import sys

from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.exception import CustomException

applicatoin = Flask(__name__)

app = applicatoin

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        try:
            # Retrieve data from the form
            data = CustomData(
                HighBP=int(request.form.get('HighBP')),
                HighChol=int(request.form.get('HighChol')),
                CholCheck=int(request.form.get('CholCheck')),
                BMI=float(request.form.get('BMI')),
                Smoker=int(request.form.get('Smoker')),
                Stroke=int(request.form.get('Stroke')),
                Diabetes=int(request.form.get('Diabetes')),
                PhysActivity=int(request.form.get('PhysActivity')),
                Fruits=int(request.form.get('Fruits')),
                Veggies=int(request.form.get('Veggies')),
                HvyAlcoholConsump=int(request.form.get('HvyAlcoholConsump')),
                AnyHealthcare=int(request.form.get('AnyHealthcare')),
                NoDocbcCost=int(request.form.get('NoDocbcCost')),
                GenHlth=int(request.form.get('GenHlth')),
                MentHlth=int(request.form.get('MentHlth')),
                PhysHlth=int(request.form.get('PhysHlth')),
                DiffWalk=int(request.form.get('DiffWalk')),
                Sex=int(request.form.get('Sex')),  
                Age=int(request.form.get('Age')),
                Education=int(request.form.get('Education')),
                Income=int(request.form.get('Income'))
            )

            # Convert data to DataFrame
            pred_df = data.get_data_as_data_frame()

            print(pred_df)

            # Load prediction model
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            if results[0] == 0:
                prediction_output = "Disease"
            elif results[0] == 1:
                prediction_output = "Attack"
            else:
                prediction_output = "Unknown"

            # Return the prediction result
            return render_template('home.html', results=prediction_output)
       
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)