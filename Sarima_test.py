import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAXResults as sresults

def test(csv_file,pkl_file, power_test ,amb_test ):

    power_path = str(power_test) # "power.csv"
    power = pd.read_csv(power_path, index_col=[0], parse_dates=True, names=["dt", "power"])
    # Read Ambient Temperature Data
    temp_path = str(amb_test)# "amb_temp.csv"
    temp = pd.read_csv(temp_path, index_col=[0], parse_dates=True, names=["dt", "temp"])
    df = pd.DataFrame()
    df = power.resample('15T').mean()
    df["temp"] = temp.resample('15T').mean()

    new_predicted = []
    output = []

    important = pd.read_csv("csv_file")
    H = important[0]
    L = important[1]
    coef = important[2]
    intercept = important[3]

    load_ = str(pkl_file)
    load_model =  sresults.load(load_)

    start = df.index[0]
    end = df.index[-1]

    predicted = loaded.predict(start, end).values
    actual = power.power.values

    df["pred"] = predicted
    df["epsilon"] = (df["power"] - df["pred"]) / df["pred"]

    epsilon = df.epsilon.values
    power_val = df.power.values
    temp_val = df.temp.values


    for i in range(len(epsilon)):
        new_p = 0
        if epsilon[i] > H:
            output.append("Positive Outlier")
            new_p = coef * temp_val[i] + intercept
            new_predicted.append(new_p)
        elif epsilon[i] < L:
            output.append("Negetive Outlier")
            new_p = coef * temp_val[i] + intercept
            new_predicted.append(new_p)
        else:
            outout.append("No Outlier")
            new_predicted.append(power_val[i])

    new_df = pd.DataFrame(new_predicted)
    new_df.columns = ["Result"]
    new_df["Remark"] = output
    new_df.to_csv("Final_result.csv", index = False)
    return

csv_fil ,pkl_fil ,power_tes ,amb_tes = input() , input() , input() , input()
test(csv_fil ,pkl_fil ,power_tes ,amb_tes)
