# update_model.py

import datetime
import pandas as pd
import sqlite3

#from Model.modules import train_model, save_model
import sys

import requests

sys.path.append('../Model')
import modules as md

import schedule
import time


def get_yesturday_predicted_flights(conn):
    dailyflightsWithPredictedDelays= 'dailyflightsWithPredictedDelays'
    query = f'SELECT * FROM {dailyflightsWithPredictedDelays}'
    j_1_data = pd.read_sql_query(query, conn)
    #j_1_data["Scheduled_Time"] = j_1_data["Scheduled_Time"].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

    j_1_data["Scheduled_Time"] = pd.to_datetime(j_1_data["Scheduled_Time"]).dt.tz_localize('UTC')
    return j_1_data

def get_daily_new_flights(conn):
    daily_scheduled_flights= 'daily_flights_with_real_delays'
    query = f'SELECT * FROM {daily_scheduled_flights}'
    new_data = pd.read_sql_query(query, conn)

    new_data["Scheduled_Time"] = pd.to_datetime(new_data["Scheduled_Time"])
    return new_data

def getHistoricalFlights(conn):
    h_flights= 'historical_Flights'
    query = f'SELECT * FROM {h_flights}'
    historical_data = pd.read_sql_query(query, conn)
    historical_data["Scheduled_Time"] = pd.to_datetime(historical_data["Scheduled_Time"])
    return historical_data

def append_new_data_to_historical_data(conn, cur, new_data):
    new_data["Scheduled_Time"] = new_data["Scheduled_Time"].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    sql = "INSERT INTO historical_Flights VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    values = new_data.values.tolist()
    cur.executemany(sql, values)
    conn.commit()

def save_the_prediction_real_delays(conn, cur, compared_flight_delays_data):
    compared_flight_delays_data["Scheduled_Time"] = compared_flight_delays_data["Scheduled_Time"].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    sql = "INSERT INTO DailyFlightsDelayPredictionCompare VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    values = compared_flight_delays_data.values.tolist()
    cur.executemany(sql, values)
    conn.commit()


def get_daily_data_no_delay():
    conn = sqlite3.connect('../Flights.db')
    daily_scheduled_flights= 'dailyflightsNoDelay'
    query = f'SELECT * FROM {daily_scheduled_flights}'
    daily_data = pd.read_sql_query(query, conn)
    daily_data['Scheduled_Time'] = pd.to_datetime(daily_data['Scheduled_Time'])
    daily_data = md.extract_datetime_features(daily_data)
    input_columns = [
        "Origine_IATACode",
        "Destination_IATACode",
        "Airline_IATACode",
        "Scheduled_Year",
        "Scheduled_Month",
        "Scheduled_Day",
        "Scheduled_Hour",
        "Scheduled_Minute",
        "Scheduled_Day_Of_Week"
    ]

    daily_data = daily_data[input_columns]
    return conn,daily_data



#===============================================================================================

# Function to update and retrain the model
def update_and_retrain_model():
    print("⚡ le réentrainement du modèle...")

    # Load existing historical data
    print("⚡ recupération des anciens données...")
    conn = sqlite3.connect('../Flights.db')
    cur = conn.cursor()
    historical_data = getHistoricalFlights(conn)

    # Load daily data with real Delays
    print("⚡ recupération des données du jour avec les retard réelle...")
    new_data = get_daily_new_flights(conn)

    # Load yesterday data with Delays predicted
    j_1_data = get_yesturday_predicted_flights(conn)

    # Merge the real and predicted data to compare and calculate ratio
    compared_flight_delays_data  = pd.merge(new_data, j_1_data,
                                            on=[
                                                "Origine_IATACode",
                                                "Destination_IATACode",
                                                "Airline_IATACode",
                                                "Scheduled_Time"
                                                ])

    # Save this to a database
    save_the_prediction_real_delays(conn, cur, compared_flight_delays_data)

    print("⚡ Fusion des donnée pour le réentrainement ...")
    
    # concatenate data
    updated_data = pd.concat([historical_data,new_data])

    print("⚡ réentrainement est commancer ...")
    # Train the model with the updated data
    updated_model = md.train_model(updated_data)

    # Save the updated model
    md.save_model(updated_model, '../Model/flight_delay_model.pkl')

    print("✔️  le modèle est à jour et prêt pour les prédictions")

    # append the new data to the main dataset
    append_new_data_to_historical_data(conn, cur, new_data)

    conn.close()
    print("\n----------------------------------------\n")



def daily_predictions():
    print("⚡ Collection de la liste des vols du jour ... ")
    conn, daily_data = get_daily_data_no_delay()
    print("✔️  la liste a été collecter avec succés.")

    input_data = daily_data.to_dict(orient='records')

    print("⚡ Lancement des prediction des retard pour la liste des vols...")

    api_url = "http://127.0.0.1:8000/daily/predict"  
    response = requests.post(api_url, json=input_data)

    # Display the prediction result
    if response.status_code == 200:
        result = response.json()
        print(result)
        daily_flights_delay_predictions = pd.json_normalize(result)
        cur = conn.cursor()
        sql = "INSERT INTO dailyflightsWithPredictedDelays VALUES (?, ?, ?, ?, ?, ?)"
        values = daily_flights_delay_predictions.values.tolist()
        cur.executemany(sql, values)
        conn.commit()
        print("✔️  les prediction ont été effectées avec succée.")
    else:
        raise Exception("❌  No Acess To API")
    conn.close()
    print("\n----------------------------------------\n")





# Schedule the update daily at midnight (adjust as needed)
# schedule.every().day.at("00:00").do(update_and_retrain_model)

# schedule.every().day.at("08:30").do(daily_predictions)
schedule.every(1).minutes.do(daily_predictions)
schedule.every(1).minutes.do(update_and_retrain_model)



# Run the scheduler
while True:
    schedule.run_pending()
    time.sleep(1)
