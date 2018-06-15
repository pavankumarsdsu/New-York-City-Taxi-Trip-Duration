import pandas as pd
import numpy as np

train= pd.read_csv('train_final_feat.csv')
weather_data= pd.read_csv('weather_nyc.csv')
train.pickup_date = pd.to_datetime(train.pickup_date)
weather_data.date = pd.to_datetime(weather_data.date)
train=train.drop([0,3,4])
print(train.shape)
weather_data = weather_data.replace("T", np.nan)


weather_data = weather_data.drop(["maximum temerature", "minimum temperature", "average temperature"], axis=1)

train = pd.merge(left=train,right=weather_data, left_on='pickup_date', right_on='date')
print(weather_data.info())
print(train.shape)
print(weather_data.shape)



print(train.shape)
print(weather_data.info())
print("---> Saving dataframes to CSV files")

train.to_csv("train_with_weather.csv", sep=",")