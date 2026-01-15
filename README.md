# CVC Lojas - Sales Forecasting Project

This project implements a complete MLOps pipeline for forecasting sales for CVC stores. It leverages **Databricks**, **Delta Lake**, **Darts** (Time Series), and **MLflow**.

## 📂 Project Structure

```text
databricks/
├── src/                       # Custom Python Package (Modularized Logic)
│   ├── ingestion/             # JDBC Connectors & Feature Store Logic
│   ├── validation/            # Validation Pipeline, Dataloaders, & Config
│   └── deploy/                # Deployment Logic (Wrapper for Inference)
├── cvc_ingestao_features_validacao.ipynb  # 1. Ingestion: SQL -> Feature Store
├── cvc_validacao_modelos_lojas.ipynb      # 2. Validation: Backtesting & Model Selection
├── cvc_treino_final_deploy.ipynb          # 3. Training: Final Model Training (2021-2025)
└── cvc_inferencia_recorrente.ipynb        # 4. Inference: Recurring Scoring
```

## 🚀 Workflows

### 1. Data Ingestion
*   **Notebook:** `cvc_ingestao_features_validacao.ipynb`
*   **Goal:** Ingests raw data from Azure SQL/JDBC.
*   **Features:** Liquid Clustering, Deduplication, Feature Table Management.

### 2. Model Validation
*   **Notebook:** `cvc_validacao_modelos_lojas.ipynb`
*   **Goal:** Runs Walk-Forward Validation to select the best model.
*   **Tech:** Darts (TFT, N-BEATS, LightGBM), MLflow Tracking.

### 3. Production Deployment
*   **Notebook:** `cvc_treino_final_deploy.ipynb`
*   **Goal:** Trains the final model on strict post-pandemic data (2021-2025).
*   **Artifact:** Registers a `UnifiedForecaster` model in Unity Catalog.

### 4. Recurring Inference
*   **Notebook:** `cvc_inferencia_recorrente.ipynb`
*   **Goal:** Loads the `UnifiedForecaster`, feeds it the last 90 days of context, and predicts the next 35 days.
*   **Output:** Saves to `bip_vprevisao_lojas_futuro` (Delta Table).

## 🛠 Setup in Databricks
Ensure that the `src` folder is present in the same directory as the notebooks. The notebooks automatically append the current working directory to `sys.path`.
