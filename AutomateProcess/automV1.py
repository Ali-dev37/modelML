import datetime
import logging
import pandas as pd
import sqlite3
import sys
import requests
import schedule
import time
sys.path.append('../Model')
import modules as md


# Configure logging
logging.basicConfig(filename='update_model.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = '../Flights.db'
MODEL_PATH = '../Model/flight_delay_model.pkl'
API_URL = "http://127.0.0.1:8000/daily/predict"

def get_yesterday_predicted_flights(conn):
    query = f'SELECT * FROM dailyflightsWithPredictedDelays'
    j_1_data = pd.read_sql_query(query, conn)
    j_1_data["Scheduled_Time"] = pd.to_datetime(j_1_data["Scheduled_Time"]).dt.tz_localize('UTC')
    return j_1_data

def get_daily_new_flights(conn):
    query = f'SELECT * FROM daily_flights_with_real_delays'
    new_data = pd.read_sql_query(query, conn)
    new_data["Scheduled_Time"] = pd.to_datetime(new_data["Scheduled_Time"])
    return new_data

def get_historical_flights(conn):
    query = f'SELECT * FROM historical_Flights'
    historical_data = pd.read_sql_query(query, conn)
    historical_data["Scheduled_Time"] = pd.to_datetime(historical_data["Scheduled_Time"])
    return historical_data

def append_new_data_to_historical_data(conn, new_data):
    with conn:
        new_data.to_sql('historical_Flights', conn, if_exists='append', index=False)

def save_the_prediction_real_delays(conn, compared_flight_delays_data):
    with conn:
        compared_flight_delays_data.to_sql('DailyFlightsDelayPredictionCompare', conn, if_exists='append', index=False)

def get_daily_data_no_delay():
    with sqlite3.connect(DB_PATH) as conn:
        query = f'SELECT * FROM dailyflightsNoDelay'
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

    return daily_data[input_columns]

def update_and_retrain_model():
    print("⚡ Starting model update and retraining...")
    logger.info("Initiating model retraining...")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            logger.info("Fetching historical data...")
            historical_data = get_historical_flights(conn)

            logger.info("Fetching new daily data with real delays...")
            new_data = get_daily_new_flights(conn)

            logger.info("Fetching predicted data from yesterday...")
            j_1_data = get_yesterday_predicted_flights(conn)
        
            logger.info("Merging data for comparing...")
            compared_flight_delays_data = pd.merge(new_data, j_1_data, on=[
                "Origine_IATACode",
                "Destination_IATACode",
                "Airline_IATACode",
                "Scheduled_Time"
            ])
            save_the_prediction_real_delays(conn,compared_flight_delays_data)

            logger.info("Training the model...")
            updated_data = pd.concat([historical_data, new_data])
            updated_model = md.train_model(updated_data)

            logger.info("Saving the updated model...")
            md.save_model(updated_model, MODEL_PATH)
            print("✔️ Model updated and saved successfully.")
            logger.info("Model update and retraining completed successfully")

            logger.info("Appending new data to the historical dataset...")
            append_new_data_to_historical_data(conn, new_data)
            logger.info("\n----------------------------------------\n")

    except Exception as e:
        print(f"❌ Error during model update and retraining: {str(e)}")
        logger.error(f"Error occurred during model update and retraining: {str(e)}")


def daily_predictions():
    print("⚡ Collecting today's flight list...")
    logger.info("Retrieving today's flight list...")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            daily_data = get_daily_data_no_delay()
            logger.info("Flight list successfully retrieved.")
            input_data = daily_data.to_dict(orient='records')

        print("⚡ Making delay predictions for today's flights...")
        logger.info("Initiating delay predictions for today's flights...")

        response = requests.post(API_URL, json=input_data)

        if response.status_code == 200:
            result = response.json()
            daily_flights_delay_predictions = pd.json_normalize(result)

            with sqlite3.connect(DB_PATH) as conn:
                daily_flights_delay_predictions.to_sql('dailyflightsWithPredictedDelays', conn, if_exists='replace', index=False)

            print("✔️ Predictions made and saved successfully.")
            logger.info("Predictions successfully made.")
        else:
            print("❌ Access to API denied.")
            logger.error("Access to API denied")

        logger.info("----------------------------------------")

    except Exception as e:
        print(f"❌ Error during daily predictions: {str(e)}")
        logger.error(f"Error occurred during daily predictions: {str(e)}")

# Schedule tasks
schedule.every(1).minutes.do(daily_predictions)
schedule.every(1).minutes.do(update_and_retrain_model)

# Run the scheduler
while True:
    schedule.run_pending()
    time.sleep(1)
