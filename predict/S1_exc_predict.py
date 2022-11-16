import pickle

import matplotlib
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBRFRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

paths = [
    "dimer_data",
]

model = pickle.load(open("../models/S1_exc_model.sav", "rb"))
ys = []
names = []

for path in paths:
    # Load representations, labels and values
    X_pool = np.load(f"{path}/repr.npy")
    names_pool = np.load(f"{path}/names.npy")

    print("Shape of data: ", X_pool.shape)

    # Evaluate performance
    y_hat = model.predict(X_pool)
    ys.append(y_hat)
    names.append(names_pool)

ys_all = np.concatenate(ys)
names_all = np.concatenate(names)

f = open("S1_exc_prediction.csv", "w+")
print(f"Filename,Prediction", file=f)
for i, name in enumerate(names_all):
    predicted = ys_all[i]
    print(f"{name},{predicted}", file=f)

f.close()
