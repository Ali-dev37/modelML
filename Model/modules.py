
import pandas as pd
# model.py

import pandas as pd
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
import pickle

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def extract_datetime_features(X):
    X['Scheduled_Year'] = X['Scheduled_Time'].dt.year
    X['Scheduled_Month'] = X['Scheduled_Time'].dt.month
    X['Scheduled_Day'] = X['Scheduled_Time'].dt.day
    X['Scheduled_Hour'] = X['Scheduled_Time'].dt.hour
    X['Scheduled_Minute'] = X['Scheduled_Time'].dt.minute
    X["Scheduled_Day_Of_Week"] = X['Scheduled_Time'].dt.weekday
    return X.drop('Scheduled_Time',axis=1)


def select_features(df):
    df = df.set_index('id')
    df = df[df['direction'] == 'arrival']
    Flights = df.loc[:,['flightNumber','originIATACode','destIATACode','scheduledTime','actualBlockTime','differ']].rename(columns={'flightNumber': "Flight_Identity", 'originIATACode': "Origine_IATACode", 'destIATACode': "Destination_IATACode", 'scheduledTime': "Scheduled_Time", 'actualBlockTime': "Actual_Time",'differ':"Delay"})
    Flights['Scheduled_Time'] = pd.to_datetime(Flights['Scheduled_Time'])
    Flights['Actual_Time'] = pd.to_datetime(Flights['Actual_Time'])
    Flights['Airline_IATACode'] = Flights['Flight_Identity'].str[:2]
    Flights['Flight_Number'] = Flights['Flight_Identity'].str[2:]
    return Flights





def train_model(df):
    Flights = df
    Flights['Scheduled_Year'] = Flights['Scheduled_Time'].dt.year
    Flights['Scheduled_Month'] = Flights['Scheduled_Time'].dt.month
    Flights['Scheduled_Day'] = Flights['Scheduled_Time'].dt.day
    Flights['Scheduled_Hour'] = Flights['Scheduled_Time'].dt.hour
    Flights['Scheduled_Minute'] = Flights['Scheduled_Time'].dt.minute
    Flights["Scheduled_Day_Of_Week"] = Flights['Scheduled_Time'].dt.weekday

    
    Flights = Flights[['Flight_Identity','Airline_IATACode', 'Flight_Number','Origine_IATACode','Destination_IATACode','Scheduled_Time','Scheduled_Year','Scheduled_Month','Scheduled_Day','Scheduled_Day_Of_Week','Scheduled_Hour','Scheduled_Minute','Actual_Time','Delay']]
    
    Flights = Flights.dropna(axis=0)
    Flights = Flights.drop_duplicates()

    Flights = Flights.dropna(subset=['Flight_Identity']).reset_index(drop=True)
    Flights = Flights.dropna(subset=['Actual_Time']).reset_index(drop=True)

    df = Flights[['Airline_IATACode', 'Origine_IATACode', 'Destination_IATACode','Scheduled_Year','Scheduled_Month','Scheduled_Day','Scheduled_Day_Of_Week','Scheduled_Hour','Scheduled_Minute'
            ,'Delay']]


    df = df[df['Delay'] != 0]

    y = df['Delay']
    X = df.drop('Delay', axis=1)

    y = y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    categorical_features = X.select_dtypes(include=['object','category']).columns
    
    categorical_transformer = Pipeline(steps=[
        ('encode',OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat',categorical_transformer,categorical_features)
        ]
    )

    
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor()
    
    
    pipeline = Pipeline([
        ('preprocessor',preprocessor),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)
    

    return pipeline

def save_model(model, filename='flight_delay_model.pkl'):
    with open(filename, 'wb') as model_file:
        pickle.dump(model, model_file)