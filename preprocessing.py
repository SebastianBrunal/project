import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path):
    return pd.read_csv(path)


def clean_data(df):
    return df.dropna()


def prepare_data(df):
    X = df[['u', 'g', 'r', 'i', 'z']]
    y_class = df['class']
    y_reg = df['redshift']

    return X, y_class, y_reg


def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test