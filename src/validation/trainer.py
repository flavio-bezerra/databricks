import mlflow
import pickle
import pandas as pd
import numpy as np
import hashlib
import traceback
from darts.metrics import mape, rmse, smape
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from .models import DartsWrapper

class ModelTrainer:
    def __init__(self, config, models_dict):
        self.config = config
        self.models = models_dict
        self.success_models = []
        self.failed_models = []
        self.spark_session = getattr(config, 'spark_session', None)
        if not self.spark_session:
             from pyspark.sql import SparkSession
             self.spark_session = SparkSession.builder.getOrCreate()

    def _get_store_ids(self, series_list):
        """Extrai IDs das lojas das TimeSeries de forma segura"""
        ids = []
        for ts in series_list:
            try:
                # Tenta pegar do índice das covariáveis estáticas (padrão Darts Group)
                if ts.static_covariates is not None:
                    ids.append(str(ts.static_covariates.index[0]))
                else:
                    ids.append("UNKNOWN")
            except:
                ids.append("ERROR")
        return ids

    def train_evaluate_walkforward(self, train_series_static, train_covs_static, full_series_scaled, full_covariates_scaled, val_series_original, target_pipeline):
        """
        Executa treinamento estático (2024) e validação mensal progressiva (2025).
        """
        mlflow.set_experiment(self.config.EXPERIMENT_NAME)
        
        # Prepara metadados das lojas
        store_ids = self._get_store_ids(train_series_static)
        store_ids_str = ",".join(store_ids)
        # Cria um hash curto para identificar o conjunto de lojas nos parâmetros
        stores_hash = hashlib.md5(store_ids_str.encode()).hexdigest()[:8]

        # Salva covariáveis completas para artefato (usado no wrapper)
        covariates_path = f"{self.config.VOLUME_ROOT}/temp/temp_future_covariates_v{self.config.VERSION}.pkl"
        
        # Ensure temp dir exists
        import os
        os.makedirs(os.path.dirname(covariates_path), exist_ok=True)
        
        with open(covariates_path, "wb") as f:
            pickle.dump(full_covariates_scaled, f)
        
        # Datas de corte para Walk-Forward
        validation_range = pd.date_range(start=self.config.VAL_START_DATE, end=self.config.INGESTION_END, freq='MS')

        for model_name, model in self.models.items():
            print(f"\\n🚀 [Model: {model_name}] Iniciando Processo...")
            model_metrics_global = {}
            all_predictions = []
            
            try:
                with mlflow.start_run(run_name=f"{model_name}_v{self.config.VERSION}") as run:
                    # --- PARTE 1: LOGGING DE METADADOS RICOS ---
                    print(f"   📝 Registrando metadados do experimento...")
                    
                    # 1. Parâmetros de Configuração Básica
                    mlflow.log_param("version", self.config.VERSION)
                    mlflow.log_param("horizon", self.config.FORECAST_HORIZON)
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_param("lags", self.config.LAGS)
                    mlflow.log_param("epochs", self.config.N_EPOCHS)
                    
                    # 2. Intervalos de Datas (Fundamental para reprodutibilidade)
                    mlflow.log_param("data_start_date", self.config.DATA_START)
                    mlflow.log_param("train_cutoff_date", self.config.TRAIN_END_DATE)
                    mlflow.log_param("validation_start", self.config.VAL_START_DATE)
                    mlflow.log_param("validation_end", self.config.INGESTION_END)
                    
                    # 3. Contexto das Lojas (Dataset Info)
                    mlflow.log_param("n_stores_trained", len(store_ids))
                    mlflow.log_param("stores_hash", stores_hash)
                    
                    # Salva a lista completa de lojas como arquivo de texto (Artifact)
                    mlflow.log_text(store_ids_str, "metadata/trained_store_ids.txt")
                    
                    # 4. Hiperparâmetros do Modelo Específico
                    if hasattr(model, 'model_params'):
                         mlflow.log_param("model_internal_params", str(model.model_params)[:250])

                    # --- PARTE 2: TREINAMENTO ESTÁTICO ---
                    print(f"   🏋️ Treinando com dados até {self.config.TRAIN_END_DATE}...")
                    
                    kwargs = {}
                    if model.supports_past_covariates:
                        kwargs['past_covariates'] = train_covs_static
                    if model.supports_future_covariates:
                        kwargs['future_covariates'] = train_covs_static
                    
                    model.fit(train_series_static, **kwargs)
                    
                    # Salvar Modelo e Registrar no MLflow (Estado Base)
                    filename = f"{model_name}_v{self.config.VERSION}.pkl"
                    local_path = f"{self.config.PATH_MODELS}/{filename}"
                    model.save(local_path)                  

                    # ========================================================
                    # AQUI ESTÁ A ALTERAÇÃO PARA REGISTRAR NO UNITY CATALOG
                    # ========================================================
                    full_model_name = f"{self.config.CATALOG}.{self.config.SCHEMA}.loja_{model_name}"
                    
                    artifacts = {"darts_model": local_path}
                    if model.supports_future_covariates:
                        artifacts["future_covariates"] = covariates_path                    

                    input_schema = Schema([ColSpec("long", "n")]) 
                    output_schema = Schema([ColSpec("double", "prediction")])
                    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
                    
                    # Tags para facilitar busca visual no MLflow UI
                    mlflow.set_tag("model_type", model_name)
                    mlflow.set_tag("validation_mode", "walk-forward-strict")

                    print(f"   💾 Registrando modelo como: {full_model_name}")
                    mlflow.pyfunc.log_model(
                        artifact_path="model",
                        python_model=DartsWrapper(),
                        artifacts=artifacts,
                        pip_requirements=["darts", "pandas", "numpy", "torch", "pytorch_lightning"],
                        input_example=pd.DataFrame({"n": [self.config.FORECAST_HORIZON]}), 
                        signature=signature,
                        # Adiciona o registro no catálogo
                        registered_model_name=full_model_name
                    )

                    # --- PARTE 3: INFERÊNCIA WALK-FORWARD (Mês a Mês) ---
                    print(f"   🔮 Iniciando Inferência Walk-Forward ({len(validation_range)} folds)...")
                    
                    for month_start in validation_range:
                        context_cutoff = month_start - pd.Timedelta(days=1)
                        metrica_mes = month_start.strftime("%Y-%m")
                        
                        val_context_series = [s.drop_after(context_cutoff) for s in full_series_scaled]
                        
                        predict_kwargs = {"n": self.config.FORECAST_HORIZON}
                        predict_kwargs['series'] = val_context_series
                        
                        if model.supports_past_covariates:
                             val_context_covs = [c.drop_after(context_cutoff) for c in full_covariates_scaled]
                             predict_kwargs['past_covariates'] = val_context_covs
                        if model.supports_future_covariates:
                             predict_kwargs['future_covariates'] = full_covariates_scaled 
                        
                        preds_scaled = model.predict(**predict_kwargs)
                        preds_inverse = target_pipeline.inverse_transform(preds_scaled, partial=True)
                        
                        metrics_month = self._calc_metrics_and_format(preds_inverse, val_series_original, metrica_mes, model_name)
                        
                        smape_m = metrics_month['metrics']['SMAPE']
                        rmse_m = metrics_month['metrics']['RMSE']
                        
                        # Log métricas mensais
                        mlflow.log_metric(f"SMAPE_{metrica_mes}", smape_m)
                        
                        print(f"     📅 {metrica_mes}: SMAPE={smape_m:.2f}%, RMSE={rmse_m:.2f}")
                        
                        all_predictions.extend(metrics_month['dfs'])

                    # --- PARTE 4: CONSOLIDAÇÃO ---
                    if all_predictions:
                        final_df = pd.concat(all_predictions)
                        final_df['VERSAO'] = self.config.VERSION
                        
                        global_mape = np.mean(np.abs((final_df['REAL'] - final_df['PREVISAO']) / final_df['REAL'])) * 100
                        global_rmse = np.sqrt(np.mean((final_df['REAL'] - final_df['PREVISAO'])**2))
                        
                        mlflow.log_metric("Global_MAPE", global_mape)
                        mlflow.log_metric("Global_RMSE", global_rmse)
                        
                        print(f"   📊 GLOBAL: MAPE={global_mape:.2f}%, RMSE={global_rmse:.2f}")
                        
                        self._save_to_delta(final_df)
                        self.success_models.append(model_name)
            
            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
                traceback.print_exc()
                self.failed_models.append(model_name)

    def _calc_metrics_and_format(self, preds, reals_full, metrica_mes, model_name):
        if not isinstance(preds, list): preds = [preds]
        if not isinstance(reals_full, list): reals_full = [reals_full]

        valid_preds, valid_reals, res_dfs = [], [], []
        
        for ts_pred, ts_real_full in zip(preds, reals_full):
            try:
                ts_real_sliced = ts_real_full.slice_intersect(ts_pred)
                valid_preds.append(ts_pred)
                valid_reals.append(ts_real_sliced)
                
                res_dfs.append(pd.DataFrame({
                    'DATA': ts_pred.time_index,
                    'PREVISAO': ts_pred.values().flatten(),
                    'REAL': ts_real_sliced.values().flatten(),
                    'CODIGO_LOJA': str(ts_pred.static_covariates.index[0]) if ts_pred.static_covariates is not None else "UNKNOWN",
                    'MODELO': model_name,
                    'METRICA_MES': metrica_mes
                }))
            except:
                continue 
        
        metrics = {"SMAPE": 0.0, "RMSE": 0.0}
        if valid_preds:
            metrics["SMAPE"] = float(np.mean(smape(valid_reals, valid_preds)))
            metrics["RMSE"] = float(np.mean(rmse(valid_reals, valid_preds)))
            
        return {"metrics": metrics, "dfs": res_dfs}

    def _save_to_delta(self, pdf):
        table_name = f"{self.config.CATALOG}.{self.config.SCHEMA}.bip_vresultado_metricas_treinamento_lojas"
        
        try:
            self.spark_session.createDataFrame(pdf).write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(table_name)
            
            # --- BEST PRACTICE: MAINTENANCE ---
            print(f"   ⚡ Otimizando tabela de resultados: {table_name}")
            self.spark_session.sql(f"OPTIMIZE {table_name}")
            self.spark_session.sql(f"VACUUM {table_name} RETAIN 168 HOURS")

        except Exception as e:
            print(f"Save error: {e}")
