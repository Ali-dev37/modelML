from datetime import datetime
import sys
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle



app = FastAPI()

class InputData(BaseModel):
    Origine_IATACode: str
    Destination_IATACode: str
    Airline_IATACode: str
    Scheduled_Year : int
    Scheduled_Month : int
    Scheduled_Day : int
    Scheduled_Hour : int
    Scheduled_Minute : int
    Scheduled_Day_Of_Week : int


# Load the trained model
with open('../Model/flight_delay_model.pkl', 'rb') as model_file:
    trained_model = pickle.load(model_file)

# API endpoint for making predictions
@app.post("/predict")
def predict_flight_delay(data: InputData):
    try:
        
        input_data = pd.DataFrame([data.model_dump()])
        #return input_data

        #Predict using the saved model
        prediction = trained_model.predict(input_data)[0]

        retard = "en retard" if prediction > 15 else "On time"

        return {"Retard en minute":  prediction, "statut": retard}
        #return data.dict()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/daily/predict")
def predict_flight_delay(data:list[InputData]):
    predicted_data = []
    try:
        for item in data:
            input_data = pd.DataFrame([item.model_dump()])
            #return input_data

            #Predict using the saved model
            prediction = trained_model.predict(input_data)[0]

            retard = "en retard" if prediction > 15 else "On time"

            item = item.model_dump()

            schedule_Time = datetime(item['Scheduled_Year'], item['Scheduled_Month'], item['Scheduled_Day'], item['Scheduled_Hour'], item['Scheduled_Minute'])

            item.update({"Scheduled_Time":schedule_Time,"delay_predection":  prediction, "statut": retard})
            
            keys_to_remove = ['Scheduled_Year', 'Scheduled_Month', 'Scheduled_Day','Scheduled_Hour','Scheduled_Minute','Scheduled_Day_Of_Week']
            for key in keys_to_remove:
                del item[key]

            predicted_data.append(item)
        
        return predicted_data

    except HTTPException as e:

        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback_details = traceback.extract_tb(exc_traceback)
        handled_line = traceback_details[-1][1]

        # You can print the line number or handle it as needed
        print(f"HTTPException handled at line {handled_line}")

        # Return an error response, for example
        return {"error": f"HTTPException handled at line {handled_line}"}
        # raise HTTPException(status_code=500, detail=str(e))
  