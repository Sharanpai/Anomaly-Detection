import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.externals import joblib

def train(power_file , power_factor_file, temp_file):
    power_path = str(power_file) # "power.csv"
    df_in = pd.read_csv(power_path, index_col=[0], parse_dates=True, names=["dt", "power"])

    pf_path = str(power_factor_file)
    df_in["pf"] = pd.read_csv(pf_path, index_col=[0], parse_dates=True, names=["dt", "pf"])

    pf_path = str(temp_file)
    df_in["temp"] = pd.read_csv(pf_path, index_col=[0], parse_dates=True, names=["dt", "temp"])
    df_in["theta"] = np.arccos(df_in["pf"])
    df_in["active"] = np.cos(df_in["theta"]) * df_in["power"]
    df_in["reactive"] = np.sin(df_in["theta"]) * df_in["power"]

    df_rs = df_in.resample('1T').mean().dropna()
    df_train = df_rs.ix[:, ["pf", "active"]]

    db = DBSCAN(eps=50, min_samples=10)
    db.fit(df_train)
    joblib.dump(db, 'DBCSAN_train.pkl')
    return

print("Enter Power, Power_factor and MCU_temp files")
powe = input()
pfactor = input()
tem = input()

train(powe, pfactor, tem)
