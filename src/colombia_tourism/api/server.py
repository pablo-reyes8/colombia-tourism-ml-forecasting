"""FastAPI service for predictions and SHAP summaries."""

from __future__ import annotations

import io
import os
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile

from colombia_tourism.inference import (
    align_features,
    load_feature_names,
    load_model,
    predict_dataframe,
)
from colombia_tourism.interpretation import shap_summary
from colombia_tourism.api.schemas import ExplainRequest, PredictRequest

DEFAULT_MODEL_URI = os.getenv("CTF_MODEL_URI")

app = FastAPI(title="Colombia Tourism Forecasting API")


@app.get("/health")
def health():
    return {"status": "ok"}


def _load_with_fallback(model_uri: Optional[str]):
    uri = model_uri or DEFAULT_MODEL_URI
    if not uri:
        raise HTTPException(status_code=400, detail="model_uri is required")
    return load_model(uri), uri


def _prepare_df(df: pd.DataFrame, target: Optional[str], model_uri: str):
    if target and target in df.columns:
        df = df.drop(columns=[target])

    feature_names = load_feature_names(model_uri)
    if feature_names:
        df, missing, extra = align_features(df, feature_names)
        return df, missing, extra

    return df, [], []


@app.post("/predict")
def predict(payload: PredictRequest):
    model, uri = _load_with_fallback(payload.model_uri)
    df = pd.DataFrame(payload.records)
    df, missing, extra = _prepare_df(df, payload.target, uri)

    preds = predict_dataframe(model, df)
    return {
        "predictions": preds.tolist(),
        "missing_features": missing,
        "extra_features": extra,
    }


@app.post("/predict-file")
def predict_file(
    file: UploadFile = File(...),
    model_uri: Optional[str] = None,
    target: Optional[str] = None,
):
    model, uri = _load_with_fallback(model_uri)
    content = file.file.read()
    df = pd.read_csv(io.BytesIO(content))
    df, missing, extra = _prepare_df(df, target, uri)

    preds = predict_dataframe(model, df)
    return {
        "predictions": preds.tolist(),
        "missing_features": missing,
        "extra_features": extra,
    }


@app.post("/explain")
def explain(payload: ExplainRequest):
    model, uri = _load_with_fallback(payload.model_uri)
    df = pd.DataFrame(payload.records)
    df, missing, extra = _prepare_df(df, payload.target, uri)

    summary, _, _ = shap_summary(model, df, max_samples=payload.max_samples)
    return {
        "mean_abs_shap": summary.to_dict(orient="records"),
        "missing_features": missing,
        "extra_features": extra,
    }
