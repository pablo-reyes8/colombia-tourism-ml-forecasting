"""Streamlit app for prediction + SHAP explanations."""

from __future__ import annotations

import io
import os
import tempfile

import pandas as pd
import streamlit as st
import mlflow

from colombia_tourism.inference import (
    align_features,
    load_feature_names_from_run,
    load_model,
    parse_run_id,
    predict_dataframe,
)
from colombia_tourism.interpretation.shap_utils import shap_summary

st.set_page_config(page_title="Colombia Tourism Forecasting", layout="wide")

st.title("Colombia Tourism Forecasting")
st.write(
    "Sube una base, carga un modelo entrenado y obtén predicciones con explicaciones SHAP."
)

with st.sidebar:
    st.header("Modelo")
    tracking_uri = st.text_input(
        "MLflow tracking URI",
        value=os.getenv("MLFLOW_TRACKING_URI", ""),
    )
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    source = st.radio(
        "Fuente del modelo",
        ["MLflow run", "Ruta local", "Subir archivo"],
        index=0,
    )

    model_uri = None
    run_id = None
    if source == "MLflow run":
        run_id = st.text_input("Run ID")
        artifact_path = st.text_input("Artifact path", value="model")
        if run_id:
            model_uri = f"runs:/{run_id}/{artifact_path}"
    elif source == "Ruta local":
        model_uri = st.text_input("Ruta del modelo (.pkl, .joblib o MLflow dir)")
    else:
        uploaded_model = st.file_uploader("Sube el modelo", type=["pkl", "joblib"])
        if uploaded_model is not None:
            tmpdir = tempfile.mkdtemp(prefix="ctf_model_")
            path = os.path.join(tmpdir, uploaded_model.name)
            with open(path, "wb") as f:
                f.write(uploaded_model.read())
            model_uri = path

st.header("Datos")
file = st.file_uploader("Sube el CSV", type=["csv"])

if file is None:
    st.stop()

raw = pd.read_csv(file)
st.write(f"Filas: {len(raw)} | Columnas: {raw.shape[1]}")
st.dataframe(raw.head(20))

st.subheader("Configuracion")
target_col = st.selectbox(
    "Columna objetivo (si existe)",
    ["(ninguna)"] + list(raw.columns),
)

max_samples = st.slider("Muestras para SHAP", min_value=50, max_value=500, value=200)

if st.button("Predecir"):
    if not model_uri:
        st.error("Necesitas cargar un modelo")
        st.stop()

    try:
        model = load_model(model_uri)
    except Exception as exc:
        st.error(f"No se pudo cargar el modelo: {exc}")
        st.stop()

    df = raw.copy()
    if target_col != "(ninguna)" and target_col in df.columns:
        df = df.drop(columns=[target_col])

    missing = []
    extra = []
    if run_id:
        feature_names = load_feature_names_from_run(run_id)
        if feature_names:
            df, missing, extra = align_features(df, feature_names)

    try:
        preds = predict_dataframe(model, df)
    except Exception as exc:
        st.error(f"Error en prediccion: {exc}")
        st.stop()

    result = raw.copy()
    result["prediction"] = preds

    st.subheader("Predicciones")
    st.dataframe(result.head(20))

    if missing:
        st.warning(f"Se agregaron columnas faltantes con 0: {missing}")
    if extra:
        st.info(f"Columnas extra ignoradas: {extra}")

    csv = result.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar predicciones", csv, "predicciones.csv", "text/csv")

    st.subheader("SHAP")
    try:
        summary, shap_values, sample = shap_summary(model, df, max_samples=max_samples)
        st.dataframe(summary)

        import matplotlib.pyplot as plt
        import shap

        fig1 = plt.figure()
        shap.summary_plot(shap_values, sample, show=False)
        st.pyplot(fig1)

        fig2 = plt.figure()
        shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
        st.pyplot(fig2)
    except Exception as exc:
        st.error(f"No se pudo calcular SHAP: {exc}")
