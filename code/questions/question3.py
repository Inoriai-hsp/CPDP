import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def getData():
    path = "D:/Inoriai/shipingHuang/code/paraADPT/EQ/DBSCANfilter-Boost.txt"
    X = []
    Y = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.replace("\n", "").replace("[", "").replace("]", "").replace(" ", "")
            tmp = line.split(",")
            if len(tmp) == 2:
                continue
            x = tmp[0:2]
            y = tmp[2]
            X.append(list(map(float, x)))
            Y.append(float(y))
    return np.array(X), np.array(Y)

def fit(X, Y):
    model = RandomForestRegressor()
    model.fit(X, Y)
    print(model.feature_importances_)

if __name__ == "__main__":
    X, Y = getData()
    fit(X, Y)