import streamlit as st
import pandas as pd

from src.preprocessing import load_data, clean_data, prepare_data, split_and_scale
from src.classification import run_classification
from src.regression import run_regression
from src.clustering import run_clustering

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ML Dashboard", layout="wide")

st.title("🚀 Machine Learning Dashboard")

# =========================
# CARGA DE DATOS
# =========================
df = load_data("data/sdss_sample.csv")
df = clean_data(df)

st.subheader("📊 Dataset")
st.dataframe(df.head())

# =========================
# PREPARACIÓN
# =========================
X, y_class, y_reg = prepare_data(df)

# =========================
# MENÚ
# =========================
option = st.sidebar.selectbox(
    "Selecciona modelo",
    ["Clasificación", "Regresión", "Clustering"]
)

# =========================
# CLASIFICACIÓN
# =========================
if option == "Clasificación":
    st.header("🔵 Clasificación (KNN)")

    X_train, X_test, y_train, y_test = split_and_scale(X, y_class)

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

    k = st.slider("Número de vecinos (k)", 1, 15, 5)

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.metric("Accuracy", acc)

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
    st.pyplot(fig)

# =========================
# REGRESIÓN
# =========================
elif option == "Regresión":
    st.header("🟢 Regresión Lineal")

    X_train, X_test, y_train, y_test = split_and_scale(X, y_reg)

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    st.metric("R2 Score", r2)

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("Real")
    ax.set_ylabel("Predicción")
    ax.set_title("Real vs Predicho")

    st.pyplot(fig)

# =========================
# CLUSTERING
# =========================
elif option == "Clustering":
    st.header("🟣 Clustering (KMeans)")

    from sklearn.cluster import KMeans

    k = st.slider("Número de clusters", 2, 6, 3)

    model = KMeans(n_clusters=k, random_state=42)
    clusters = model.fit_predict(X)

    # PCA para visualización
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots()
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, ax=ax)
    ax.set_title("Clusters con PCA")

    st.pyplot(fig)