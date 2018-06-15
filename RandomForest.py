import pandas as pd
import numpy as np
import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import sys
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


def randomForestRegressor(train):
    start = time.time()
    y = np.log(train["trip_duration"].values + 1)

    Xtr, Xte, Ytr, Yte = train_test_split(train[feature_names].values, y, test_size=0.25, random_state=1987)

    print("Training a Random Forest Regressor")

    model = RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_leaf=10, n_jobs=-1)
    model.fit(Xtr, Ytr)
    Y_pred = model.predict(Xte)

    error_ = np.sqrt(mean_squared_error(Yte, Y_pred))

    print("RMSLE with Random forest = {}".format(error_))

    end = time.time()
    print("Random Forest Execution Time={}".format(end - start))
    print("-------------------------------------------")


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
    print("Features with weather: Random Forest Tree")
    randomForestRegressor(train_weather)
