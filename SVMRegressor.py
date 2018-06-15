import pandas as pd
import numpy as np
import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import sys
from sklearn.metrics import mean_squared_error

from sklearn.svm import SVR



def svmRegressor(trainSet):

    start = time.time()

    print("SVM Regressor Training")
    scalar = StandardScaler().fit(trainSet[feature_names].values)
    X_train = scalar.fit_transform(trainSet[feature_names].values)

    # To make it Root Mean Square Log Error
    y = np.log(trainSet["trip_duration"].values + 1)

    Xtr, XTe, Ytr, YTe = train_test_split(X_train, y, test_size=0.25, random_state=1987)

    model = SVR(kernel="rbf", C=1e3, gamma=0.1)
    model.fit(Xtr, Ytr)
    Ypred = model.predict(XTe)

    error = np.sqrt(mean_squared_error(YTe, Ypred))

    print("SVM regression error ={}".format(error))

    end = time.time()
    print("Execution time for SVM ={}".format(end - start))
    print("---------------------------------------")

if __name__ == "__main__":

    # import engineered features
    # Performance metrics is root mean square log error

    print("Reading Data")
    train_weather = pd.read_csv("train_final_feat.csv",nrows=100)
    train_weather=train_weather.dropna()

    do_not_use_for_train = ["id","date", "pickup_datetime", "trip_duration", "pickup_date", "dropoff_datetime", "avg_speed_h",
                            "avg_speed_m", "avg_speed"]

    feature_names = [f for f in train_weather.columns if f not in do_not_use_for_train]

    print("Starting Training")
    print("Feature with weather: SVM Regressor")
    svmRegressor(train_weather)