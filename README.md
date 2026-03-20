# Colombia International Tourism Forecasting


![Repo size](https://img.shields.io/github/repo-size/pablo-reyes8/colombia-tourism-ml-forecasting)
![Last commit](https://img.shields.io/github/last-commit/pablo-reyes8/colombia-tourism-ml-forecasting)
![Open issues](https://img.shields.io/github/issues/pablo-reyes8/colombia-tourism-ml-forecasting)
![Forks](https://img.shields.io/github/forks/pablo-reyes8/colombia-tourism-ml-forecasting?style=social)
![Stars](https://img.shields.io/github/stars/pablo-reyes8/colombia-tourism-ml-forecasting?style=social)


A comprehensive, end‐to‐end forecasting project that leverages both machine‐learning and econometric methods to predict monthly foreign tourist arrivals across 81 Colombian cities from 2018 to 2023. We begin by assembling a high‐dimensional panel dataset, merging:

- **Remote Sensing Features**: Land‐cover metrics (urban, rural, water areas) and key indices (NDVI, NDWI) extracted from Sentinel-2 imagery via Google Earth Engine.  
- **Economic Indicators**: Weighted GDP, inflation, and exchange rate data to capture purchasing power and macro‐conditions.  
- **Security Statistics**: Monthly counts of homicides, thefts, and sexual offenses as proxies for traveler safety perceptions.  
- **Tourism Infrastructure**: Numbers of hotels, rooms, beds, main road connections, and international entry points (air and land).  
- **Climate Variables**: Average monthly temperature layers to account for seasonal comfort.  
- **Event & Expenditure Proxies**: Counts of local festivals and detailed tourist spending categories (daily and per‐trip costs).


On this foundation, we implement and compare:
1. **Machine‐Learning Regressors**: Linear, Ridge, Lasso, Elastic Net, KNN, Decision Tree, Random Forest, Gradient Boosting and XGBoost.  
2. **Econometric Benchmarks**: OLS (autoregressive, GDP‐only, multi‐indicator) and Random Effects panel models.

Models are evaluated via cross‐validation and held‐out testing (R², MAE, MSE, RMSE), with XGBoost ultimately chosen for its superior accuracy (~0.99 R²), fast training and flexibility. We then apply:

- **LIME** for localized, interpretable explanations in both major hubs and mid‐sized cities.  
- **Partial Dependence Plots** to visualize global, non‐linear feature effects and critical thresholds.

Our findings highlight spatial extent (urban/rural/water areas), economic capacity, connectivity, and local events as the dominant drivers of tourism flows. Strategic recommendations emerge—enhancing connectivity, promoting signature events, expanding infrastructure and fine‐tuning pricing—to support evidence‐based tourism development across Colombia.  


---

## Project Pipeline

1. **Data Ingestion & Initial Exploration**  
   - **Loads:** monthly tourist arrivals; GDP, inflation, exchange rates; city‐level crime (homicide, theft, sexual offenses); hotel/bed counts, roads, airports; temperatures; travel spending; event counts
   - **Cleaning:** deduplicate, format standardization, outlier detection.  
   - **Interpolation:** use kriging to fill climate and cost values at each city centroid.  
   - **Summaries:** generate key histograms, boxplots and correlation matrices. 

2. **Feature Engineering from Satellite Imagery**  
   - Use Google Earth Engine and Sentinel-2 composites to extract urban, rural and water area metrics.  
   - Compute additional spatial indicators (e.g. green/water indices, Otsu thresholding).

3. **Final Dataset Assembly & Imputation**  
   - Merge all data sources into a panel of 81 cities × 60 months.  
   - Impute missing entries via KNN to preserve patterns and avoid row deletion.

4. **Descriptive Statistics & Visualization**  
   - Generate summary tables, correlation matrices and exploratory plots.  
   - Map city locations, connectivity networks, temperature distributions, etc.

5. **Modeling & Econometric Comparison**  
   - Fit baseline OLS specifications (autoregressive, GDP-only, multi-indicator).  
   - Train ML models: Linear, Ridge, Lasso, Elastic Net, KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost.  
   - Evaluate with cross-validation, test-set R², MAE, MSE, RMSE.  

6. **Best Model Interpretation & Insights**  
   - Select XGBoost as the champion (Test R² ≈ 0.99).  
   - Apply LIME for local explanations on large and mid-sized cities.  
   - Use PDPs to reveal global non-linear effects and critical thresholds.  
   - Synthesize strategic recommendations: connectivity, event programming, infrastructure and pricing.

---

## File Descriptions

- **1_Initial_Data_Construction_and_Descriptive_Analysis**  
  Notebooks for raw data cleaning, exploratory plots and summary statistics for a single year.

- **2_Satellite_Image_Feature_Engineering**  
  Code to fetch Sentinel-2 composites via Google Earth Engine, compute area masks (urban/rural/water), vegetation and water indices.

- **3_Final_Dataset_Assembly_and_Missing_Value_Imputation**  
  Merges all sources into one panel (applay the first script 5 times for every year); applies KNN imputation to fill gaps in features.

- **4_Descriptive_Statistics_and_Final_Dataset_Plots**  
  Generates correlation heatmaps, distribution charts and geospatial plots on the assembled dataset.

- **5_Modeling_and_Econometric_Comparison**  
  Implements OLS benchmarks and trains ML models; compares performance metrics and runtimes.

- **6_Best_Model_Interpretation_and_Insights**  
  Contains the final XGBoost pipeline, LIME explanation scripts, PDP generation and strategic conclusions.
  
- **Base Final1**
  It is the final database that is reached after running scripts 1-3, in this way the graphics and modeling can be done without running everything.

- **Release Databases**
  In the *Realeses* section you can find the .rar file with all the bases to run scripts 1-3

- **apps/streamlit_app.py**
  Streamlit interface for predictions and SHAP explanations.

- **scripts/train.py**
  CLI training/orchestration with MLflow tracking, optional hyperparameter search, model registration and champion promotion.

- **scripts/package_data.py**
  CLI dataset packaging step that materializes the data contract: raw snapshot, processed modeling dataset, feature list, schema, profile and manifest.

- **scripts/select_features.py**
  CLI feature-selection suite over `Base Final1.csv` with polynomial Lasso, adaptive Lasso, sparse group Lasso, forward/backward elimination, PCA loadings and PLS-VIP consensus ranking.

- **scripts/check_drift.py**
  CLI drift monitoring step that compares a current dataset against a packaged baseline and can trigger retraining.

- **scripts/promote_data_package.py**
  Promotes a packaged current dataset into the production baseline used by drift monitoring and retraining.

- **scripts/api_smoke_test.py**
  Sends a real HTTP request to the FastAPI service to validate that the deployed champion model is serving predictions correctly.

- **scripts/benchmark.py**
  CLI model comparison on a shared split.

- **scripts/eda.py**
  CLI descriptive analysis that exports summary tables and figures.

- **scripts/infer.py**
  CLI batch inference on CSVs.

- **scripts/interpret.py**
  CLI SHAP interpretation and plots.

- **Demo Images**
  In this folder you can find some demonstration images of how the urban/rural area of ​​cities was calculated based on satellite images.

- **dags/colombia_tourism_mlops_dag.py**
  Airflow DAG orchestrating dataset packaging, bootstrap training, drift detection, conditional retraining, batch inference and API smoke testing.

- **docker/airflow/Dockerfile**
  Custom Airflow image based on the official Airflow runtime, with project dependencies and DAG code baked in.

---

## Modular Code (src/)

- `src/colombia_tourism/preprocessing` data cleaning utilities.
- `src/colombia_tourism/features` feature engineering helpers + `satellite.py` for Sentinel-2 / Earth Engine land-cover extraction.
- `src/colombia_tourism/modeling` pipelines, PCA, polynomial features, model factory, plus:
  - LOESS monthly decomposition helpers
  - spatial kriging imputation
  - KNN mixed-type imputation pipeline
  - econometric comparison helpers (OLS / random effects)
  - feature-selection suite (Lasso, adaptive Lasso, sparse group Lasso, stepwise selection, PCA, PLS)
- `src/colombia_tourism/interpretation` SHAP, PDP, permutation importance.
- `src/colombia_tourism/api` professional FastAPI service with versioned endpoints, serving schemas and a dedicated prediction service layer.
- `src/colombia_tourism/inference.py` model resolution, MLflow registry support, metadata loading and feature alignment.
- `src/colombia_tourism/mlflow_utils.py` MLflow orchestration wrapper for training, tuning, version registration and champion alias promotion.
- `src/colombia_tourism/mlops` data contracts, dataset packaging, drift monitoring and scheduler-friendly orchestration helpers.

---

## Local Setup

1. Install dependencies: `pip install -r requirements.txt`
2. (Optional, for full preprocessing + Sentinel-2 + kriging + econometrics) `pip install -r requirements-geospatial.txt`
3. Add sources to path: `export PYTHONPATH=src`
4. (Recommended for registry/API workflows) set `MLFLOW_TRACKING_URI`

Example:
- `export PYTHONPATH=src`
- `export MLFLOW_TRACKING_URI=file:./mlruns`
- `export CTF_REGISTERED_MODEL_NAME=colombia-tourism-forecasting`
- `export CTF_MODEL_ALIAS=champion`

---

## CLI Workflows

### Data Packaging
1. `python scripts/package_data.py --data "Data/Base Final1.csv" --output-dir artifacts/data/base_final_package`
2. `python scripts/package_data.py --data "Data/Base Final1.csv" --engineer-features --output-dir artifacts/data/base_final_engineered`

Each package contains:
- `raw/base_final_snapshot.csv`
- `processed/modeling_dataset.csv`
- `metadata/manifest.json`
- `metadata/feature_list.txt`
- `metadata/schema.json`
- `metadata/profile.json`
- `metadata/feature_manifest.json`
- `metadata/ingestion_summary.md`

### Feature Selection
1. `python scripts/select_features.py --data "Data/Base Final1.csv" --output-dir outputs/feature_selection`
2. `python scripts/select_features.py --data "Data/Base Final1.csv" --engineer-features --max-features 20 --consensus-min-votes 2`
3. `python scripts/train.py --model xgboost --features outputs/feature_selection/consensus_features.txt --registered-model-name colombia-tourism-forecasting`

The selector exports:
- `feature_selection_summary.csv` with method-level results
- `feature_selection_rankings.csv` with per-method rankings in original feature space
- `feature_selection_consensus.csv` with votes and normalized consensus scores
- `consensus_features.txt` with the top recommended variables ready to pass into `scripts/train.py --features`

### Train + Register (MLflow)
1. `python scripts/train.py --model xgboost --registered-model-name colombia-tourism-forecasting`
2. `python scripts/train.py --model xgboost --tune --n-iter 50 --registered-model-name colombia-tourism-forecasting`
3. `python scripts/train.py --candidate-models xgboost random_forest ridge --registered-model-name colombia-tourism-forecasting`
4. `python scripts/train.py --model xgboost --model-params '{"preset":"notebook_best"}' --registered-model-name colombia-tourism-forecasting`
5. `python scripts/train.py --model xgboost --features outputs/feature_selection/consensus_features.txt --registered-model-name colombia-tourism-forecasting`
6. `python scripts/train.py --data-package-dir artifacts/data/base_final_package --model xgboost --registered-model-name colombia-tourism-forecasting`

The MLflow wrapper now handles:
- Run-level metrics and params
- Optional hyperparameter search
- Dataset snapshot + fingerprint
- Feature list, target name and serving metadata
- Packaged data contract artifacts when training from `--data-package-dir`
- Model registration in MLflow Model Registry
- Promotion of the best version to the `champion` alias

### Drift Monitoring
1. `python scripts/check_drift.py --reference-dir artifacts/data/base_final_package --current-data "Data/Base Final1.csv" --output-dir artifacts/drift/latest`
2. `python scripts/check_drift.py --reference-dir artifacts/data/base_final_package --current-data new_data.csv --output-dir artifacts/drift/latest --retrain-on-drift`
3. `python scripts/check_drift.py --reference-dir artifacts/data/base_final_package --current-data new_data.csv --output-dir artifacts/drift/latest --fail-on-drift`

The drift monitor exports:
- `drift_feature_report.csv`
- `drift_summary.json`
- `drift_report.md`
- `retrain_decision.json`
- `current_package/` with the regenerated dataset package for the candidate data

### API Smoke Test
1. `python scripts/api_smoke_test.py --api-base-url http://localhost:8000 --input artifacts/data/base_final_package/processed/modeling_dataset.csv --target "Nmero Extranjeros"`

### Inference (batch)
1. `python scripts/infer.py --model-uri runs:/<RUN_ID>/model --input data.csv --output preds.csv`
2. `python scripts/infer.py --model-uri models:/colombia-tourism-forecasting@champion --input data.csv --output preds.csv`
3. `python scripts/infer.py --model-uri path/to/model.joblib --input data.csv --output preds.csv --target "Nmero Extranjeros"`

### Interpretation (SHAP)
1. `python scripts/interpret.py --model-uri runs:/<RUN_ID>/model --input data.csv --output shap_summary.csv`
2. `python scripts/interpret.py --model-uri models:/colombia-tourism-forecasting@champion --input data.csv --output shap_summary.csv --plot-dir outputs/shap`

### EDA
1. `python scripts/eda.py --output-dir outputs/eda`

### Benchmark
1. `python scripts/benchmark.py --engineer-features --models linear ridge random_forest xgboost`

---

## API

Run:
- `export CTF_REGISTERED_MODEL_NAME=colombia-tourism-forecasting`
- `export CTF_MODEL_ALIAS=champion`
- `uvicorn colombia_tourism.api.server:app --host 0.0.0.0 --port 8000`

Endpoints:
- `GET /health/live`
- `GET /health/ready`
- `GET /api/v1/model`
- `GET /api/v1/features`
- `GET /api/v1/models/registered`
- `POST /api/v1/predict`
- `POST /api/v1/predict/single`
- `POST /api/v1/predict/file`
- `POST /api/v1/explain`

Default serving behavior:
- The API resolves the default model from MLflow Registry using `CTF_REGISTERED_MODEL_NAME` + `CTF_MODEL_ALIAS`
- If needed, requests can still override the model with an explicit `model_uri`
- Feature names and target metadata are loaded from MLflow artifacts so requests can be aligned consistently

Example prediction payload:

```json
{
  "records": [
    {
      "Ciudad": "medellin",
      "Temperatura": 24.5,
      "Pib Ponderado": 1500.0,
      "Inflacion": 8.3,
      "Eventos": 3,
      "Area Urbana": 120.0,
      "Area Rural": 80.0,
      "Area Agua": 4.0,
      "N Camas": 25000
    }
  ],
  "model": {
    "registered_model_name": "colombia-tourism-forecasting",
    "model_alias": "champion"
  },
  "options": {
    "strict_features": false,
    "fill_missing_value": 0.0
  }
}
```

---

## Airflow Orchestration

The repository now includes an end-to-end Airflow DAG in `dags/colombia_tourism_mlops_dag.py`.

Pipeline stages inside the DAG:
- package current data into `artifacts/data/current_candidate`
- bootstrap train and promote the baseline if no production reference exists
- compare current data against `artifacts/data/production_baseline`
- retrain and promote the baseline if drift thresholds are exceeded
- run batch inference with the `champion` model
- call the FastAPI service with a real prediction request as a serving smoke test

Main DAG:
- `colombia_tourism_end_to_end_mlops`

Airflow services in Docker Compose:
- `airflow-init`
- `airflow-apiserver`
- `airflow-scheduler`
- `airflow-dag-processor`
- `airflow-triggerer`

Run the platform:
1. `docker compose up --build mlflow api app airflow-init airflow-apiserver airflow-scheduler airflow-dag-processor airflow-triggerer`
2. Open Airflow at `http://localhost:8080`
3. Trigger `colombia_tourism_end_to_end_mlops`

---

## Streamlit App

Run:
- `streamlit run apps/streamlit_app.py`

The app lets you:
- Upload a CSV
- Load a trained model (MLflow, local, or upload)
- Generate predictions
- Generate SHAP explanations

---

## MLflow Tracking & Data Lineage

Set tracking URI (local or remote):
- `export MLFLOW_TRACKING_URI=file:./mlruns`

Recommended registry variables:
- `export CTF_REGISTERED_MODEL_NAME=colombia-tourism-forecasting`
- `export CTF_MODEL_ALIAS=champion`

The training wrapper logs:
- Full production pipeline
- Feature list, target name and model metadata
- Dataset sample + fingerprint
- Train/test/CV metrics
- Optional tuning summary
- Registered model versions for deployment
- Champion alias promotion for the best model

---

## Deployment

### Docker
This repository now uses one multi-stage `Dockerfile` with dedicated targets for:
- `mlflow-server`
- `data-pipeline`
- `trainer`
- `drift-monitor`
- `api-server`
- `streamlit-app`

Examples:
- `docker build --target api-server -t colombia-tourism-api .`
- `docker build --target trainer -t colombia-tourism-trainer .`
- `docker build --target drift-monitor -t colombia-tourism-drift .`

### Docker Compose
1. Core serving stack: `docker compose up --build mlflow api app`
2. Data + training jobs: `docker compose --profile jobs up --build data_pipeline trainer`
3. Drift monitoring: `docker compose --profile monitoring up --build drift_monitor`
4. Full orchestration with Airflow: `docker compose up --build mlflow api app airflow-init airflow-apiserver airflow-scheduler airflow-dag-processor airflow-triggerer`

Compose services:
- `mlflow` on port `5000`
- `api` on port `8000`
- `app` on port `8501`
- `airflow-apiserver` on port `8080`
- `data_pipeline` for dataset packaging jobs
- `trainer` for model-training jobs
- `drift_monitor` for scheduled drift checks
- `postgres` for Airflow metadata

---

## Key Takeaways

- **Spatial Primacy:** Urban/rural/water areas are the strongest predictors.  
- **Economic & Security Signals:** GDP and crime rates add meaningful context.  
- **Model Power:** XGBoost delivers near-perfect fit with fast training and deep interpretability via LIME/PDP.  
- **Actionable Strategies:** Improve connectivity, promote events, expand infrastructure and manage costs tailored to city size.

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests at  
https://github.com/pablo-reyes8

## License

This project is licensed under the Apache License 2.0.  

