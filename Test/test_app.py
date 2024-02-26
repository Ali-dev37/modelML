# app.py

from datetime import datetime,time
import streamlit as st
import requests
import pandas as pd 

def extract_datetime_features(X:datetime):
    return [X.year,X.month,X.day,X.hour, X.minute,X.weekday()]

@st.cache_resource
def getCategoryData():
    df = pd.read_csv('flights2023.csv', encoding = 'ISO-8859-1',delimiter=';')
    df = df.set_index('id')
    df = df[df['direction'] == 'arrival']
    Flights = df.loc[:,['flightNumber','originIATACode','destIATACode','scheduledTime','actualBlockTime','differ']].rename(columns={'flightNumber': "Flight_Identity", 'originIATACode': "Origine_IATACode", 'destIATACode': "Destination_IATACode", 'scheduledTime': "Scheduled_Time", 'actualBlockTime': "Actual_Time",'differ':"Delay"})

    Flights['Scheduled_Time'] = pd.to_datetime(Flights['Scheduled_Time'])
    Flights['Actual_Time'] = pd.to_datetime(Flights['Actual_Time'])   

    Flights = Flights.dropna(axis=0)
    
    Flights['Airline_IATACode'] = Flights['Flight_Identity'].str[:2]
    Flights = Flights[['Flight_Identity','Airline_IATACode','Origine_IATACode','Destination_IATACode','Scheduled_Time','Actual_Time','Delay']]
    
    Flights = Flights.drop_duplicates()

    Flights = Flights.dropna(subset=['Flight_Identity']).reset_index(drop=True)
    Flights = Flights.dropna(subset=['Actual_Time']).reset_index(drop=True)

    features_data = Flights[['Airline_IATACode', 'Origine_IATACode', 'Destination_IATACode','Scheduled_Time','Delay']]

    df = features_data

    return df

# Streamlit app title and description
st.title("Flight Delay Prediction App")
st.write("Enter the details below to predict flight delay.")


df = getCategoryData()
Flights = df


# Input form for user to enter details
st.sidebar.header("Choose your filter: ")
# Create for Region

airiata = st.sidebar.selectbox("Airline IATA Code ", df["Airline_IATACode"].unique())
# if not airiata:
#     Flights = df.copy()
# else:
#     Flights = df[df["Airline_IATACode"].isin(airiata)]

# Create for Region
origineiata = st.sidebar.selectbox("Origine IATA Code", df["Origine_IATACode"].unique())

destinationiata = st.sidebar.selectbox("Destination IATA Code", df["Destination_IATACode"].unique())

scheduled_arrival_date = st.sidebar.date_input("Scheduled arrival date", "today")
scheduled_arrival_time = st.sidebar.time_input("Scheduled arrival time",'now')


scheduled_arrival_datetime = datetime.combine(scheduled_arrival_date,scheduled_arrival_time)
datesplitlist = extract_datetime_features(scheduled_arrival_datetime)

if st.button("Predict"):
    # Prepare the input data in JSON format
    input_data = {
        "Origine_IATACode": str(origineiata),
        "Destination_IATACode": str(destinationiata),
        "Airline_IATACode": str(airiata),
        'Scheduled_Year' : datesplitlist[0],
        'Scheduled_Month' : datesplitlist[1],
        'Scheduled_Day' : datesplitlist[2],
        'Scheduled_Hour' : datesplitlist[3],
        'Scheduled_Minute' : datesplitlist[4],
        'Scheduled_Day_Of_Week' : datesplitlist[5]
    }
    st.write(input_data)
    # Make a POST request to the FastAPI endpoint
    api_url = "http://127.0.0.1:8000/predict"  # Update the URL if needed
    response = requests.post(api_url, json=input_data)

    # Display the prediction result
    if response.status_code == 200:
        result = response.json()
        st.write(response.json())
        #st.success(f"Predicted Flight Delay: {result['prediction']} minutes")
    else:
        st.error(response.json())









# if not airiata and not origineiata and not destinationiata and not scheduled_arrival_datetime :
#     Flights = df.copy()
# elif not origineiata and not destinationiata and not scheduled_arrival_datetime:
#     Flights = df[df["Airline_IATACode"].isin(airiata)]
# elif not airiata and not destinationiata and not scheduled_arrival_datetime:
#     Flights = df[df["Origine_IATACode"].isin(origineiata)]
# elif origineiata and destinationiata and scheduled_arrival_datetime:
#     Flights = df[df["Origine_IATACode"].isin(origineiata) & df["Destination_IATACode"].isin(destinationiata)]
# elif airiata and destinationiata and scheduled_arrival_datetime:
#     Flights = df[df["Airline_IATACode"].isin(airiata) & df["Destination_IATACode"].isin(destinationiata)]
# elif airiata and origineiata and scheduled_arrival_datetime:
#     Flights = df[df["Airline_IATACode"].isin(airiata) & df["Origine_IATACode"].isin(origineiata)]
# elif destinationiata:
#     Flights = df[df["Destination_IATACode"].isin(destinationiata)]
# else:
#     Flights = df[df["Origine_IATACode"].isin(origineiata) & df["Destination_IATACode"].isin(destinationiata) & df["Airline_IATACode"].isin(airiata)]




# st.write(Flights)
# st.write(str(scheduled_arrival_datetime))




# Button to trigger the prediction

