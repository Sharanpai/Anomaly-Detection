import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.externals import joblib
import collections
from collections import Counter

def test(power_file , power_factor_file, best_csv):
    power_path = str(power_file) # "power.csv"
    df_in = pd.read_csv(power_path, index_col=[0], parse_dates=True, names=["dt", "power"])

    pf_path = str(power_factor_file)
    df_in["pf"] = pd.read_csv(pf_path, index_col=[0], parse_dates=True, names=["dt", "pf"])

    #pf_path = str(temp_file)
    #df_in["temp"] = pd.read_csv(pf_path, index_col=[0], parse_dates=True, names=["dt", "temp"])
    df_in["theta"] = np.arccos(df_in["pf"])
    df_in["active"] = np.cos(df_in["theta"]) * df_in["power"]
    df_in["reactive"] = np.sin(df_in["theta"]) * df_in["power"]

    df_rs = df_in.resample('1T').mean().dropna()
    df_train = df_rs.ix[:, ["pf", "active"]]


    db = DBSCAN(eps=50, min_samples=10)
    db.fit(df_train)

    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    df_train["labels"] = labels

    step1_result = checkstep1(df_train)

    step2_result = checkstep2(df_train , best_csv)
    
    results_df = pd.DataFrame(step1_result)
    results_df.columns = ["States result"]
    results_df["Sub States Results"] = step2_result
    results_df.to_csv("Socket_results.csv")

    return

#def addtime(df_labels):
    num = 0
    count = 0
    prev_label = -2

    df_out = pd.DataFrame()
    print("fafa" ,df_labels.shape[0] )
    for i in range(df_labels.shape[0]):
        if i == 0:
            num += 1
        elif df_labels.ix[i] == df_labels.ix[i-1]:
            num+= 1
        else:
            df_out.loc[count, "label"] = df_labels.ix[i-1]
            df_out.loc[count, "count"] = num
            num = 1
            count += 1
    print("fafaedasfsa" ,df_out.shape[0] )
    return df_out

def addtime(enter):
    copy = enter[:]
    num = 0
    for i in range(0,len(enter)):
        if i == 0:
            num += 30
        elif enter[i] == enter[i-1]:
            num+= 30
        else:
            num = 30
        copy[i] = num
    return copy

def checkstep1(data):
    step1_output = []
    dblabels_vals = data["labels"].values
    for i in range(len(dblabels_vals)):

        if dblabels_vals[i] == -1:
            step1_output.append("Malfunction")
        else:
            step1_output.append("Step 1 Cleared")

    return step1_output


def checkstep2(data, csv_file):
    checkstep2_output = []
    dblabels_vals = data['labels']

    new_sub_states = addtime(list(dblabels_vals.values))

    name = str(csv_file)
    best_sub_ = pd.read_csv(name)

    one = best_sub_.ones.values
    two = best_sub_.twos.values
    zero = best_sub_.zeros.values
    #print(test_labels_kmeans)
    subb = new_sub_states

    for i in range(len(subb)):
        if subb[i] == 1:

            if subb[i] in one:
                checkstep2_output.append("Step2 Cleared")
            else:
                checkstep2_output.append("Step2_malfunction")
        elif subb[i] == 2:
            if subb[i] in two:
                checkstep2_output.append("Step2 Cleared")
            else:
                checkstep2_output.append("Step2_malfunction")
        else:
            if subb[i] in zero:
                checkstep2_output.append("Step2 Cleared")
            else:
                checkstep2_output.append("Step2_malfunction")

    return checkstep2_output

print("Enter Power and Power_factor files")
powe = input()
pfactor = input()
bestt = input()
#tem = input()

test(powe, pfactor , bestt)
