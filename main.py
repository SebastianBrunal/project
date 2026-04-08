from src.preprocessing import load_data, clean_data, prepare_data, split_and_scale
from src.classification import run_classification
from src.regression import run_regression
from src.clustering import run_clustering


def main():
    df = load_data("data/sdss_sample.csv")
    df = clean_data(df)

    X, y_class, y_reg = prepare_data(df)

    # Clasificación
    X_train, X_test, y_train, y_test = split_and_scale(X, y_class)
    run_classification(X_train, X_test, y_train, y_test)

    # Regresión
    X_train, X_test, y_train, y_test = split_and_scale(X, y_reg)
    run_regression(X_train, X_test, y_train, y_test)

    # Clustering
    run_clustering(X.values)


if __name__ == "__main__":
    main()