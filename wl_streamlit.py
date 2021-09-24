import streamlit as st
import pandas as pd
import numpy as np
import base64  
from fbprophet import Prophet
from fbprophet.diagnostics import performance_metrics
from fbprophet.diagnostics import cross_validation
from fbprophet.plot import plot_cross_validation_metric

#1 title & import CSV
st.title('Time Series Forecasting Win loss')
st.write("Import the time series CSV file") 
data = st.file_uploader('Upload here',type='csv')

#2 check the data is none or not
if data is not None:
    appdata = pd.read_csv(data)  #read the data fro
    appdata['ds'] = pd.to_datetime(appdata['ds'],errors='coerce') 
    st.write(data) #display the data  
    max_date = appdata['ds'].max() #compute latest date in the data 

#3 prediction date
st.write("SELECT FORECAST PERIOD") #text displayed
periods_input = st.number_input('How many days forecast do you want?',min_value = 1, max_value = 365)
#The minimum number of days a user can select is one, while the maximum is  #365 (yearly forecast) 

#4 check the not none then fit data
if data is not None:
    obj = Prophet() #Instantiate Prophet object
    obj.fit(appdata)  #fit the data 

#5text to be displayed
st.write("VISUALIZE FORECASTED DATA")  
st.write("The following plot shows future predicted values. 'yhat' is the  predicted value; upper and lower limits are 80% confidence intervals by  default")

#6 future date for prediction
if data is not None:
    future = obj.make_future_dataframe(periods=periods_input)
#Prophet.make_future_dataframe() takes the Prophet model object and   
#extends the time series dataframe for specified period for which user needs #the forecast
    fcst = obj.predict(future)  #make prediction for the extended data
    forecast = fcst[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
#The predict() method assigns each row in the ‘future’ dataframe a predicted #value denoted as yhat
#Choose only the forecasted records (having date after the latest date in #original data)
    forecast_filtered =  forecast[forecast['ds'] > max_date]    
    st.write(forecast_filtered)  #Display some forecasted records
    st.write("The next visual shows the actual (black dots) and predicted (blue line) values over time.") 
    figure1 = obj.plot(fcst) #plot the actual and predicted values
    st.write(figure1)  #display the plot
    st.write("The following plots show a high level trend of predicted values")
    figure2 = obj.plot_components(fcst) 
    st.write(figure2) 
