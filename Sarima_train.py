import pandas as pd
import numpy as np
import itertools

import statsmodels.api as sm

from sklearn import linear_model
from sklearn.model_selection import train_test_split as tts

def train(powertrain ,temptrain ):
    power_path = str(powertrain) # "power.csv"
    power = pd.read_csv(power_path, index_col=[0], parse_dates=True, names=["dt", "power"])
    # Read Ambient Temperature Data
    temp_path = str(temptrain) # "amb_temp.csv"
    temp = pd.read_csv(temp_path, index_col=[0], parse_dates=True, names=["dt", "temp"])
    df = pd.DataFrame()
    df = power.resample('15T').mean()
    df["temp"] = temp.resample('15T').mean()
    # finding the best A and b
    train_boundry = int(df.shape[0] * 0.6)

    findx = df.index.values[0]
    mindx = df.index.values[train_boundry]
    X_train = pd.DataFrame(df.ix[findx:mindx, "temp"])
    y_train = pd.DataFrame(df.ix[findx:mindx, "power"])
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    coef, intercept  = regr.coef_, regr.intercept_


    new_df = pd.DataFrame(data = power.power.values)
    start = new_df.index[0]
    end = new_df.index[-1]

    mod = sm.tsa.statespace.SARIMAX(df['power'], order=(3, 1, 2),
                            seasonal_order=(1, 1, 1, 96),
                            enforce_stationarity=False,
                            enforce_invertibility=False)
    results = mod.fit()

    df_thresh = pd.DataFrame(results.predict(start, end))
    df_thresh.columns = ["pred"]

    df_thresh["actual"] = new_df
    df_thresh["epsilon"] = (df_thresh["actual"] - df_thresh["pred"]) / df_thresh["pred"]
    alpha = 0.998

    H = df_thresh.epsilon.quantile(alpha)
    L = df_thresh.epsilon.quantile(1.0 - alpha)

    print (H, L)
    print("reached here")

    print("fit model")
    results.save("sarima.pkl")
    main_list = [H, L, coef , intercept]
    df = pd.DataFrame(main_list)
    df.to_csv("ImportantList.csv" , index = True )
    return main_list

print("Enter both the power and amb_temp training files")
powe = input()
tem = input()

train(powe, tem)
