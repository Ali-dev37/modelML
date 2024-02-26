# app.py

import streamlit as st
import requests
import pandas as pd

def getCategoryData():
    df = pd.read_csv('../AutomateProcess/flights2023 - Copie.csv', encoding = 'ISO-8859-1',delimiter=';')
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
airiata = st.sidebar.multiselect("Airline IATA Code ", df["Airline_IATACode"].unique())
# if not airiata:
#     Flights = df.copy()
# else:
#     Flights = df[df["Airline_IATACode"].isin(airiata)]

# Create for Region
origineiata = st.sidebar.multiselect("Origine IATA Code", df["Origine_IATACode"].unique())

destinationiata = st.sidebar.multiselect("Destination IATA Code", df["Destination_IATACode"].unique())







if not airiata and not origineiata and not destinationiata:
    Flights = df.copy()
elif not origineiata and not destinationiata:
    Flights = df[df["Airline_IATACode"].isin(airiata)]
elif not airiata and not destinationiata:
    Flights = df[df["Origine_IATACode"].isin(origineiata)]
elif origineiata and destinationiata:
    Flights = df[df["Origine_IATACode"].isin(origineiata) & df["Destination_IATACode"].isin(destinationiata)]
elif airiata and destinationiata:
    Flights = df[df["Airline_IATACode"].isin(airiata) & df["Destination_IATACode"].isin(destinationiata)]
elif airiata and origineiata:
    Flights = df[df["Airline_IATACode"].isin(airiata) & df["Origine_IATACode"].isin(origineiata)]
elif destinationiata:
    Flights = df[df["Destination_IATACode"].isin(destinationiata)]
else:
    Flights = df[df["Origine_IATACode"].isin(origineiata) & df["Destination_IATACode"].isin(destinationiata) & df["Airline_IATACode"].isin(airiata)]

st.write(Flights)










# st.write(Flights)



# # Create for State
# state = st.sidebar.multiselect("Pick the State", df2["State"].unique())
# if not state:
#     df3 = df2.copy()
# else:
#     df3 = df2[df2["State"].isin(state)]


# scheduled_arrival_date = st.text_input("Scheduled Arrival Date (YYYY-MM-DD HH:MM):")
# departure_city = st.text_input("Departure City:")
# arrival_city = st.text_input("Arrival City:")
# air_carrier = st.text_input("Air Carrier:")
# flight_number = st.text_input("Flight Number:")
# # Button to trigger the prediction
# if st.button("Predict"):
#     # Prepare the input data in JSON format
#     input_data = {
#         "scheduled_arrival_date": scheduled_arrival_date,
#         "departure_city": departure_city,
#         "arrival_city": arrival_city,
#         "air_carrier": air_carrier,
#         "flight_number": flight_number
#     }

#     # Make a POST request to the FastAPI endpoint
#     api_url = "http://127.0.0.1:8000/predict"  # Update the URL if needed
#     response = requests.post(api_url, json=input_data)

#     # Display the prediction result
#     if response.status_code == 200:
#         result = response.json()
#         #st.success(f"Predicted Flight Delay: {result['prediction']} minutes")
#     else:
#         st.error("Failed to get prediction. Please check your input and try again.")





