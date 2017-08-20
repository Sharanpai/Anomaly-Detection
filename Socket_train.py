import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.externals import joblib
import collections
from collections import Counter

def train(power_file , power_factor_file):
    power_path = str(power_file) # "power.csv"
    df_in = pd.read_csv(power_path, index_col=[0], parse_dates=True, names=["dt", "power"])

    pf_path = str(power_factor_file)
    df_in["pf"] = pd.read_csv(pf_path, index_col=[0], parse_dates=True, names=["dt", "pf"])

    df_in["theta"] = np.arccos(df_in["pf"])
    df_in["active"] = np.cos(df_in["theta"]) * df_in["power"]
    df_in["reactive"] = np.sin(df_in["theta"]) * df_in["power"]

    df_rs = df_in.resample('1T').mean().dropna()
    df_train = df_rs.ix[:, ["pf", "active"]]

    db = DBSCAN(eps=50, min_samples=10)
    db.fit(df_train)
    joblib.dump(db, 'DBCSAN_train.pkl')

    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    df_train["labels"] = labels

    sub_states = addtime(list(df_train["labels"].values))
    #now lets divide the data
    df = pd.DataFrame(data = [df_train["labels"].values,sub_states])
    df = df.T
    df.columns = ["label", "substates"]
    #df.head(20)
    ones = df[df['label'] == 1]
    zeros = df[df['label'] == 0]
    twos = df[df["label"] == 2]

    dfones = pd.DataFrame(best_sub_states(ones))
    dftwos = pd.DataFrame(best_sub_states(twos))
    dfzeros = pd.DataFrame(best_sub_states(zeros))

    #dfones.to_csv("best_ones.csv", index = False)
    #dftwos.to_csv("best_twos.csv", index = False)
    #dfzeros.to_csv("best_zeros.csv", index = False)

    best_sub_states_df = pd.DataFrame(dfones)
    best_sub_states_df.columns = ["ones"]
    best_sub_states_df["zeros"] = dfzeros
    best_sub_states_df["twos"] = dftwos
    best_sub_states_df.to_csv("best_sub_states_df.csv", index = False)

    df_labels = df_train["labels"].reset_index(drop=True)
    periodicity = dict()

    labels = np.delete(df_train["labels"].unique(), -1)
    df_s3 = pd.DataFrame(index=labels, columns=["mean", "std", "min", "max"])
    for label in labels:
        periodicity[label] = np.ediff1d(df_labels[df_labels==label].index.values) - 1
        df_s3.loc[label, "mean"] = np.mean(periodicity[label])
        df_s3.loc[label, "std"] = np.std(periodicity[label])
        df_s3.loc[label, "min"] = min(periodicity[label])
        df_s3.loc[label, "max"] = max(periodicity[label])

    df_s3.to_csv("periodicity.csv", index = False)


    n_labels_ = len(labels)
    tm = np.zeros((n_labels_, n_labels_), dtype=float)
    l_vals = df_train[df_train["labels"] != -1]["labels"].values

    trans_count = Counter(zip(l_vals, l_vals[1:]))
    for (x,y), c in trans_count.iteritems():
        tm[x,y] = c

    tm.astype(int)
    for i in range(n_labels_):
        tm[i,:] = tm[i,:] / np.sum(tm[i, :])

    print(tm)
    dff = pd.DataFrame(tm)
    dff.to_csv("transition.csv", index = False)
    return


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

def best_sub_states(dat):

    a = dat["substates"].values
    counttype=collections.Counter(a)
    #print(counttype)
    keytype = list(counttype.keys())
    populationtype = list(counttype.values())
    #print(sum(populationtype)/4)
    best_sub = []
    for i in range(len(populationtype)):
        if populationtype[i] > sum(populationtype)/4:
            best_sub.append(keytype[i])

    return best_sub

print("Enter Power and Power_factor files")
powe = input()
pfactor = input()
#tem = input()

train(powe, pfactor)
