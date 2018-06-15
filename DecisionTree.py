import pandas as pd
import numpy as np
import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import sys
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


def decisionTreeRegressor(trainSet):
    start = time.time()
    y = np.log(trainSet["trip_duration"].values + 1)

    Xtr, Xte, Ytr, Yte = train_test_split(trainSet[feature_names].values, y, test_size=0.25, random_state=1987)
    print("Training Random Forest Regressor")
    model = DecisionTreeRegressor(max_depth=15, min_samples_leaf=25, min_samples_split=10)
    model.fit(Xtr, Ytr)
    Yprediction = model.predict(Xte)

    error_ = np.sqrt(mean_squared_error(Yte, Yprediction))
    end = time.time()

    print("RMSLE with Decision Tree ={}".format(error_))
    print("Execution time for tuning ={}".format(end - start))
    print("---------------------------------")

if __name__ == "__main__":

    # import engineered features
    # Performance metrics is root mean square log error

    print("Reading Data")
    train_weather = pd.read_csv("train_final_feat.csv",nrows=30000)
    train_weather=train_weather.dropna()

    do_not_use_for_train = ["id","date", "pickup_datetime", "trip_duration", "pickup_date", "dropoff_datetime", "avg_speed_h",
                            "avg_speed_m", "avg_speed"]

    feature_names = [f for f in train_weather.columns if f not in do_not_use_for_train]

    print("Starting Training")
    print("Features with weather: Decision Tree")
    decisionTreeRegressor(train_weather)