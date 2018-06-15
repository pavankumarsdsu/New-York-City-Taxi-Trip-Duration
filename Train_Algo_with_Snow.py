import pandas as pd
import numpy as np
import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
#from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import sys
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

sys.stdout = open("Boosting_thenew algo_with_snow_1.txt", "wt")

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


def decisionTreeRegressor(trainSet):
    start = time.time()
    y = np.log(trainSet["trip_duration"].values + 1)

    Xtr, Xte, Ytr, Yte = train_test_split(trainSet[feature_names].values, y, test_size=0.25, random_state=1987)

    print("Training Decision Tree Regressor")
    model = DecisionTreeRegressor(max_depth=15, min_samples_leaf=25, min_samples_split=10)
    model.fit(Xtr, Ytr)
    Yprediction = model.predict(Xte)

    error_ = np.sqrt(mean_squared_error(Yte, Yprediction))
    end = time.time()

    print("RMSLE with Decision Tree ={}".format(error_))
    print("Execution time for tuning ={}".format(end - start))
    print("---------------------------------")


def randonForestRegressor(train):

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



def gbtRegressor(trainSet):
    start = time.time()
    print("sklearn GBRT is running")
    y = np.log(trainSet["trip_duration"].values + 1)

    Xtr, Xte, Ytr, Yte = train_test_split(trainSet[feature_names].values, y, test_size=0.25, random_state=1987)

    model = GradientBoostingRegressor(max_depth=15, learning_rate=1.0, n_estimators=1000, min_samples_leaf=25,
                                      min_samples_split=10)
    model.fit(Xtr, Ytr)
    print("Predicting Results")
    Ypred = model.predict(Xte)
    mse = mean_squared_error(Yte, Ypred)
    rmse = np.sqrt(mse)

    print("Accuracy with Gradient Boosted Tree={}".format(rmse))
    end = time.time()

    print("sklearn GBRT execution time ={}".format(end - start))
    print("-----------------------------------------")


if __name__ == "__main__":

    # import engineered features
    # Performance metrics is root mean square log error

    print("Reading Data")
    train_weather = pd.read_csv("train_with_weather.csv",nrows=30000)
    train_weather=train_weather.dropna()

    do_not_use_for_train = ["id","date", "pickup_datetime", "trip_duration", "pickup_date", "dropoff_datetime", "avg_speed_h",
                            "avg_speed_m", "avg_speed"]

    feature_names = [f for f in train_weather.columns if f not in do_not_use_for_train]

    print("Starting Training")
    print("Features with weather: Decision Tree")
    decisionTreeRegressor(train_weather)
    print("Features with weather: Random Forest Tree")
    randonForestRegressor(train_weather)
    print("Features with weather: sklearn GBRT")
    gbtRegressor(train_weather)
    print("Feature with weather: SVM Regressor")
    svmRegressor(train_weather)



