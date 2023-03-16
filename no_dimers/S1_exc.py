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

# Load representations, labels and values
X_pool = np.load("../data_no_dimers/repr.npy")
y_pool = np.load("../data_no_dimers/S1_exc.npy")

print("Shape of data_no_dimers: ", X_pool.shape, y_pool.shape)

# Split data_no_dimers into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_pool, y_pool, test_size=0.1, random_state=1
)

# 13 one body pot, 91 2 body pot and 1183 3 body pots.
# 2 body pots are 48 long, 3 body pots are 39 long
# for a total of 50518 SLATM terms
ic = []
one_b = list(range(0, 13))
ic.append(one_b)
two_b = list(range(13, 4368 + 13))
ic.append(two_b)
three_b = list(range(4368 + 13, 46137 + 4368 + 13))
ic.append(three_b)

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
model = XGBRegressor(
    n_estimators=5000,
    eta=0.05,
    colsample_bytree=0.75,
    max_depth=8,
    eval_metric="mae",
)
# Define the data_no_dimerssets to evaluate each iteration for early stopping
evalset = [(X_train, y_train), (X_test, y_test)]

# Fit the model
model.fit(
    X_train, y_train, eval_set=evalset, early_stopping_rounds=10, feature_weights=fws
)
# Evaluate performance
y_hat = model.predict(X_test)
err = np.abs(y_hat - y_test)
print(
    f"Max error at id {np.where(y_pool == y_test[np.argmax(err)])} : {np.max(err)} out of reference {y_test[np.argmax(err)]}"
)
print(
    f"Min error at id {np.where(y_pool == y_test[np.argmin(err)])} : {np.min(err)} out of reference {y_test[np.argmin(err)]}"
)
print(f"std of errors is {np.std(err)}")
r2 = r2_score(y_test, y_hat)
mae = mean_absolute_error(y_test, y_hat)
print("r2: %.3f" % r2)
print("mae: %.3f" % mae)

# Retrieve performance metrics
results = model.evals_result()

# Plot learning curves
plt.rc("font", size=22)
plt.plot(results["validation_0"]["mae"], label="train")
plt.plot(results["validation_1"]["mae"], label="test")
plt.xlabel("Estimators")
plt.ylabel(r"$S_{1}^{vert}$ MAE $\it{(eV)}$")

# Show the legend
plt.legend()

# Show and save the plot
plt.show()
plt.savefig("xgboost_S1_exc_lc.png")
pickle.dump(model, open("S1_exc_model.sav", "wb"))
