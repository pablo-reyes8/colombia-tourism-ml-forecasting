"""Professional FastAPI service for tourism predictions."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from colombia_tourism.api.schemas import (
    ExplainRequest,
    ExplanationResponse,
    FeatureSchemaResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictSingleRequest,
    PredictionFileResponse,
    PredictionResponse,
    RegisteredModelsResponse,
)
from colombia_tourism.api.service import PredictionService


@lru_cache(maxsize=1)
def get_prediction_service() -> PredictionService:
    return PredictionService()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Colombia Tourism Forecasting API",
        version="1.0.0",
        summary="Serve registered MLflow tourism models with production-friendly endpoints.",
        description=(
            "FastAPI service for serving the Colombia tourism forecasting models. "
            "By default it resolves the MLflow champion model configured through environment variables."
        ),
    )

    @app.get("/health/live", response_model=HealthResponse, tags=["health"])
    def live_health():
        service = get_prediction_service()
        return HealthResponse(
            status="ok",
            service="colombia-tourism-api",
            model_ready=service.readiness(),
            default_model=service.default_model_info(),
        )

    @app.get("/health/ready", response_model=HealthResponse, tags=["health"])
    def ready_health():
        service = get_prediction_service()
        ready = service.readiness()
        if not ready:
            raise HTTPException(status_code=503, detail="Default model is not ready")
        return HealthResponse(
            status="ok",
            service="colombia-tourism-api",
            model_ready=True,
            default_model=service.default_model_info(),
        )

    @app.get("/api/v1/model", response_model=ModelInfoResponse, tags=["model"])
    def get_default_model():
        service = get_prediction_service()
        info = service.default_model_info()
        if info is None:
            raise HTTPException(status_code=503, detail="Default model is not configured")
        return ModelInfoResponse(model=info)

    @app.get("/api/v1/features", response_model=FeatureSchemaResponse, tags=["model"])
    def get_feature_schema():
        service = get_prediction_service()
        try:
            payload = service.feature_schema()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return FeatureSchemaResponse(**payload)

    @app.get(
        "/api/v1/models/registered",
        response_model=RegisteredModelsResponse,
        tags=["model"],
    )
    def get_registered_models(
        registered_model_name: str | None = Query(
            default=None,
            description="Optional override for the registered model name.",
        ),
    ):
        service = get_prediction_service()
        try:
            payload = service.list_registered_versions(registered_model_name)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return RegisteredModelsResponse(**payload)

    @app.post("/api/v1/predict", response_model=PredictionResponse, tags=["predict"])
    def predict(payload: PredictRequest):
        service = get_prediction_service()
        try:
            model_selector = payload.model.model_dump()
            if payload.model_uri:
                model_selector["model_uri"] = payload.model_uri
            options = payload.options.model_dump()
            if payload.target:
                options["target"] = payload.target
            response = service.predict_records(
                payload.records,
                model=model_selector,
                **options,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return PredictionResponse(**response)

    @app.post(
        "/api/v1/predict/single",
        response_model=PredictionResponse,
        tags=["predict"],
    )
    def predict_single(payload: PredictSingleRequest):
        service = get_prediction_service()
        try:
            model_selector = payload.model.model_dump()
            if payload.model_uri:
                model_selector["model_uri"] = payload.model_uri
            options = payload.options.model_dump()
            if payload.target:
                options["target"] = payload.target
            response = service.predict_records(
                [payload.record],
                model=model_selector,
                **options,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return PredictionResponse(**response)

    @app.post(
        "/api/v1/predict/file",
        response_model=PredictionFileResponse,
        tags=["predict"],
    )
    async def predict_file(
        file: UploadFile = File(...),
        model_uri: str | None = Query(default=None),
        registered_model_name: str | None = Query(default=None),
        model_alias: str | None = Query(default=None),
        model_version: str | None = Query(default=None),
        artifact_path: str = Query(default="model"),
        target: str | None = Query(default=None),
        strict_features: bool = Query(default=False),
        fill_missing_value: float = Query(default=0.0),
        prediction_column: str = Query(default="prediction"),
        include_input_records: bool = Query(default=False),
    ):
        service = get_prediction_service()
        try:
            content = await file.read()
            response = service.predict_csv_bytes(
                content,
                filename=file.filename,
                model={
                    "model_uri": model_uri,
                    "registered_model_name": registered_model_name,
                    "model_alias": model_alias,
                    "model_version": model_version,
                    "artifact_path": artifact_path,
                },
                target=target,
                strict_features=strict_features,
                fill_missing_value=fill_missing_value,
                prediction_column=prediction_column,
                include_input_records=include_input_records,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return PredictionFileResponse(**response)

    @app.post("/api/v1/explain", response_model=ExplanationResponse, tags=["explain"])
    def explain(payload: ExplainRequest):
        service = get_prediction_service()
        try:
            model_selector = payload.model.model_dump()
            if payload.model_uri:
                model_selector["model_uri"] = payload.model_uri
            target = payload.target or payload.options.target
            response = service.explain_records(
                payload.records,
                model=model_selector,
                target=target,
                strict_features=payload.options.strict_features,
                fill_missing_value=payload.options.fill_missing_value,
                max_samples=payload.max_samples,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return ExplanationResponse(**response)

    @app.get("/health", include_in_schema=False)
    def legacy_health():
        return live_health()

    @app.post("/predict", include_in_schema=False)
    def legacy_predict(payload: PredictRequest):
        return predict(payload)

    @app.post("/predict-file", include_in_schema=False)
    async def legacy_predict_file(
        file: UploadFile = File(...),
        model_uri: str | None = Query(default=None),
        registered_model_name: str | None = Query(default=None),
        model_alias: str | None = Query(default=None),
        model_version: str | None = Query(default=None),
        artifact_path: str = Query(default="model"),
        target: str | None = Query(default=None),
        strict_features: bool = Query(default=False),
        fill_missing_value: float = Query(default=0.0),
        prediction_column: str = Query(default="prediction"),
        include_input_records: bool = Query(default=False),
    ):
        return await predict_file(
            file=file,
            model_uri=model_uri,
            registered_model_name=registered_model_name,
            model_alias=model_alias,
            model_version=model_version,
            artifact_path=artifact_path,
            target=target,
            strict_features=strict_features,
            fill_missing_value=fill_missing_value,
            prediction_column=prediction_column,
            include_input_records=include_input_records,
        )

    @app.post("/explain", include_in_schema=False)
    def legacy_explain(payload: ExplainRequest):
        return explain(payload)

    return app


app = create_app()
