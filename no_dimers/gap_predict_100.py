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

path = "../data_dimers"
model_1 = pickle.load(open("gap_model.sav", "rb"))

# Load representations, labels and values
X_pool = np.load(f"{path}/repr_dimers.npy")
y_pool = np.load(f"{path}/gap_dimers.npy")
names_pool = np.load(f"{path}/names_dimers.npy")


def unison_shuffled_copies(a, b, c):
    assert len(a) == len(b) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]


X_pool, y_pool, names_pool = unison_shuffled_copies(X_pool, y_pool, names_pool)

X_1 = X_pool[:1250]
y_1 = y_pool[:1250]
names_1 = names_pool[:1250]
X_2 = X_pool[-100:]
y_2 = y_pool[-100:]
names_2 = names_pool[-100:]

# Evaluate performance without dimers
ys = []
names = []
y_hat = model_1.predict(X_1)
ys.append(y_hat)
names.append(names_1)

ys_all = np.concatenate(ys)
names_all = np.concatenate(names)
mae = mean_absolute_error(y_1, ys_all)
print(f"Without dimers in training the MAE on holdout dimers is {mae}.")

# Now we add some dimers
fw_one_b = 1 / (3 * 13)
fw_two_b = 1 / (3 * 4368)
fw_three_b = 1 / (3 * 46137)
fws = np.zeros((50518))
for i in range(50518):
    if i < 13:
        fws[i] = fw_one_b
    elif i < 4381:
        fws[i] = fw_two_b
    else:
        fws[i] = fw_three_b
assert np.isclose(np.sum(fws), 1)

# Define the model
model_2 = XGBRegressor(
    n_estimators=5000,
    eta=0.05,
    colsample_bytree=0.25,
    max_depth=8,
    eval_metric="mae",
)
# Define the data_no_dimerssets to evaluate each iteration for early stopping
evalset = [(X_2, y_2), (X_1, y_1)]

# Refit the model
model_2.fit(
    X_2,
    y_2,
    eval_set=evalset,
    early_stopping_rounds=10,
    feature_weights=fws,
    xgb_model=model_1,
)

# Evaluate performance with dimers
ys = []
names = []
y_hat = model_2.predict(X_1)
ys.append(y_hat)
names.append(names_1)

ys_all = np.concatenate(ys)
names_all = np.concatenate(names)
mae = mean_absolute_error(y_1, ys_all)
print(f"With some dimers in training the MAE on holdout dimers is {mae}.")
