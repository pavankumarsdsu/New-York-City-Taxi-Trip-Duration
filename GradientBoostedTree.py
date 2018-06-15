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
import matplotlib.pyplot as plt
from sklearn import ensemble


from sklearn import ensemble
def gbtRegressor(trainSet):
    start = time.time()
    print("sklearn GBRT is running")
    y = np.log(trainSet["trip_duration"].values + 1)

    Xtr, Xte, Ytr, Yte = train_test_split(trainSet[feature_names].values, y, test_size=0.25, random_state=1987)

    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
    clf = ensemble.GradientBoostingRegressor(**params)

    clf.fit(Xtr, Ytr)
    mse = mean_squared_error(Yte, clf.predict(Xte))
    print("MSE: %.4f" % mse)

    # #############################################################################
    # Plot training deviance

    # compute test set deviance
    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

    for i, y_pred in enumerate(clf.staged_predict(Xte)):
        test_score[i] = clf.loss_(Yte, y_pred)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
             label='Training Set Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
             label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')
    plt.show()

    # #######################


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
    print("Features with weather: sklearn GBRT")
    gbtRegressor(train_weather)