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
  CLI training with MLflow logging.

- **scripts/infer.py**
  CLI batch inference on CSVs.

- **scripts/interpret.py**
  CLI SHAP interpretation and plots.

- **Demo Images**
  In this folder you can find some demonstration images of how the urban/rural area of ​​cities was calculated based on satellite images.

---

## Modular Code (src/)

- `src/colombia_tourism/preprocessing` data cleaning utilities.
- `src/colombia_tourism/features` feature engineering helpers + `satellite.py` for Sentinel-2 / Earth Engine land-cover extraction.
- `src/colombia_tourism/modeling` pipelines, PCA, polynomial features, model factory, plus:
  - LOESS monthly decomposition helpers
  - spatial kriging imputation
  - KNN mixed-type imputation pipeline
  - econometric comparison helpers (OLS / random effects)
- `src/colombia_tourism/interpretation` SHAP, PDP, permutation importance.
- `src/colombia_tourism/api` FastAPI service.
- `src/colombia_tourism/inference.py` model loading + feature alignment.
- `src/colombia_tourism/mlflow_utils.py` experiment tracking + dataset logging.

---

## Local Setup

1. Install dependencies: `pip install -r requirements.txt`
2. (Optional, for full preprocessing + Sentinel-2 + kriging + econometrics) `pip install -r requirements-geospatial.txt`
3. Add sources to path: `export PYTHONPATH=src`

---

## CLI Workflows

### Train + Track (MLflow)
1. `python scripts/train.py --model xgboost`
2. `python scripts/train.py --model lightgbm --poly-degree 2`
3. `python scripts/train.py --model catboost --pca-components 10`

This logs:
- Model artifacts
- Metrics (R2, MAE, MSE, RMSE, CV)
- Feature list + target name
- Optional data sample (default enabled)

### Inference (batch)
1. `python scripts/infer.py --model-uri runs:/<RUN_ID>/model --input data.csv --output preds.csv`
2. `python scripts/infer.py --model-uri path/to/model.joblib --input data.csv --output preds.csv --target "Nmero Extranjeros"`

### Interpretation (SHAP)
1. `python scripts/interpret.py --model-uri runs:/<RUN_ID>/model --input data.csv --output shap_summary.csv`
2. `python scripts/interpret.py --model-uri runs:/<RUN_ID>/model --input data.csv --output shap_summary.csv --plot-dir outputs/shap`

---

## API

Run:
- `uvicorn colombia_tourism.api.server:app --host 0.0.0.0 --port 8000`

Endpoints:
- `GET /health`
- `POST /predict` (JSON records)
- `POST /predict-file` (CSV upload)
- `POST /explain` (SHAP summary)

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

The training CLI logs:
- Full model pipeline
- Feature list and target name
- Dataset sample + fingerprint (for reproducibility)
- Metrics and CV scores

---

## Deployment

### Docker
1. Build: `docker build -t colombia-tourism .`
2. Run API: `docker run -p 8000:8000 colombia-tourism`

### Docker Compose
1. `docker-compose up --build`

This runs:
- API on port `8000`
- Streamlit app on port `8501`

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

