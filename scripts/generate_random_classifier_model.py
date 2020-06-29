import lightgbm
import numpy as np
import pandas as pd
import sklearn

np.random.seed(30)
n_columns = 40
n_rows = 10000

X = pd.DataFrame(np.random.uniform(-100,100,size=(n_rows, n_columns)), columns=list(range(n_columns)))
y = pd.Series(np.random.randint(0, 5, size=(n_rows)))

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)

train_dataset = lightgbm.Dataset(X_train, y_train)
model = lightgbm.LGBMClassifier(boosting_type="gbdt")
model.fit(X_train, y_train, eval_set=(X_test, y_test))
model.booster_.save_model("./temp/test-model")
