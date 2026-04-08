from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def run_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n=== REGRESIÓN ===")
    print("MSE:", mse)
    print("R2:", r2)

    return y_test, y_pred