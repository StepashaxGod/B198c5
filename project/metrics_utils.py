import time
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(df, target="quality", test_size=0.2, seed=42, n_estimators=200):
    """
    Train/evaluate RandomForestRegressor on a prepared dataframe.
    Returns rmse, mae, r2, runtime_seconds.
    RMSE is computed as sqrt(MSE).
    """
    start = time.time()

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    model = RandomForestRegressor(
        random_state=seed,
        n_estimators=n_estimators,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    runtime = float(time.time() - start)

    return rmse, mae, r2, runtime

