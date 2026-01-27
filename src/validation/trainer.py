import mlflow
import pickle
import pandas as pd
import numpy as np
import hashlib
import traceback
from typing import Dict, List, Any, Optional
from darts import TimeSeries
from darts.metrics import mape, rmse, smape
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from pyspark.sql import SparkSession
from .models import DartsWrapper

class ModelTrainer:
    """
    Classe responsável pelo treinamento e validação walk-forward dos modelos.
    """
    def __init__(self, config: Any, models_dict: Dict[str, Any]):
        self.config = config
        self.models = models_dict
        self.success_models: List[str] = []
        self.failed_models: List[str] = []
        self.spark_session: Optional[SparkSession] = getattr(config, 'spark_session', None)
        if not self.spark_session:
             self.spark_session = SparkSession.builder.getOrCreate()

    def _extract_id_from_original(self, ts: TimeSeries) -> str:
        """
        Extrai ID apenas de séries ORIGINAIS (Input), onde o índice é confiável.
        """
        try:
            if ts.static_covariates is not None:
                # Prioridade: Coluna explicita
                if "codigo_loja" in ts.static_covariates.columns:
                    val = str(ts.static_covariates["codigo_loja"].iloc[0])
                    if val.endswith(".0"): val = val[:-2]
                    return val
                
                # Fallback: Índice (lógica antiga, mantida por segurança)
                val = str(ts.static_covariates.index[0])
                if val.endswith(".0"): val = val[:-2]
                return val
        except:
            pass
        return "UNKNOWN"

    def _get_store_ids(self, series_list: List[TimeSeries]) -> List[str]:
        return [self._extract_id_from_original(ts) for ts in series_list]

    def train_evaluate_walkforward(
        self, 
        train_series_static: List[TimeSeries], 
        train_covs_static: List[TimeSeries], 
        full_series_scaled: List[TimeSeries], 
        full_covariates_scaled: List[TimeSeries], 
        val_series_original: List[TimeSeries], 
        target_pipeline: Any,
        allow_new_run: bool = True
    ) -> None:
        from contextlib import nullcontext

        mlflow.set_experiment(self.config.EXPERIMENT_NAME)
        
        # --- PASSO CRUCIAL 1: Capturar a ordem dos IDs da fonte confiável ---
        # full_series_scaled tem a mesma ordem e tamanho que entra no predict
        ordered_store_ids = self._get_store_ids(full_series_scaled)
        
        store_ids_str = ",".join(ordered_store_ids)
        stores_hash = hashlib.md5(store_ids_str.encode()).hexdigest()[:8]

        # Salva artefatos
        covariates_path = f"{self.config.VOLUME_ROOT}/temp/temp_future_covariates_v{self.config.VERSION}.pkl"
        import os
        os.makedirs(os.path.dirname(covariates_path), exist_ok=True)
        with open(covariates_path, "wb") as f:
            pickle.dump(full_covariates_scaled, f)
        
        validation_range = pd.date_range(start=self.config.VAL_START_DATE, end=self.config.INGESTION_END, freq='MS')

        for model_name, model in self.models.items():
            print(f"\\n🚀 [Model: {model_name}] Iniciando Processo...")
            all_predictions = []
            
            run_context = mlflow.start_run(run_name=f"{model_name}_v{self.config.VERSION}", nested=True) if allow_new_run else nullcontext()

            try:
                with run_context:
                    print(f"   📝 Registrando metadados...")
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_param("n_stores", len(ordered_store_ids))
                    
                    # --- TREINO ---
                    print(f"   🏋️ Treinando...")
                    kwargs = {}
                    if model.supports_past_covariates: kwargs['past_covariates'] = train_covs_static
                    if model.supports_future_covariates: kwargs['future_covariates'] = train_covs_static
                    
                    model.fit(train_series_static, **kwargs)
                    
                    # Salva e Registra Modelo
                    filename = f"{model_name}_v{self.config.VERSION}.pkl"
                    local_path = f"{self.config.PATH_MODELS}/{filename}"
                    model.save(local_path)                  

                    full_model_name = f"{self.config.CATALOG}.{self.config.SCHEMA}.loja_{model_name}"
                    artifacts = {"darts_model": local_path}
                    if model.supports_future_covariates: artifacts["future_covariates"] = covariates_path                    

                    input_schema = Schema([ColSpec("long", "n")]) 
                    output_schema = Schema([ColSpec("double", "prediction")])
                    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
                    
                    mlflow.pyfunc.log_model(
                        artifact_path=f"model_{model_name.lower()}",
                        python_model=DartsWrapper(),
                        artifacts=artifacts,
                        input_example=pd.DataFrame({"n": [self.config.FORECAST_HORIZON]}), 
                        signature=signature,
                        registered_model_name=full_model_name
                    )

                    # --- VALIDAÇÃO (Walk-Forward) ---
                    print(f"   🔮 Validando ({len(validation_range)} meses)...")
                    
                    for month_start in validation_range:
                        context_cutoff = month_start - pd.Timedelta(days=1)
                        metrica_mes = month_start.strftime("%Y-%m")
                        
                        val_context_series = [s.drop_after(context_cutoff) for s in full_series_scaled]
                        
                        predict_kwargs = {"n": self.config.FORECAST_HORIZON, 'series': val_context_series}
                        
                        if model.supports_past_covariates:
                             val_context_covs = [c.drop_after(context_cutoff) for c in full_covariates_scaled]
                             predict_kwargs['past_covariates'] = val_context_covs
                        if model.supports_future_covariates:
                             predict_kwargs['future_covariates'] = full_covariates_scaled 
                        
                        preds_scaled = model.predict(**predict_kwargs)
                        preds_inverse = target_pipeline.inverse_transform(preds_scaled, partial=True)
                        
                        # --- PASSO CRUCIAL 2: Passar a lista de IDs originais ---
                        metrics_month = self._calc_metrics_and_format(
                            preds_inverse, 
                            val_series_original, 
                            metrica_mes, 
                            model_name,
                            ordered_store_ids # << LISTA SEGURA
                        )
                        
                        smape_m = metrics_month['metrics']['smape']
                        rmse_m = metrics_month['metrics']['rmse']
                        mlflow.log_metric(f"smape_{metrica_mes}", smape_m)
                        print(f"     📅 {metrica_mes}: SMAPE={smape_m:.2f}%, RMSE={rmse_m:.2f}")
                        
                        all_predictions.extend(metrics_month['dfs'])

                    # --- CONSOLIDAÇÃO ---
                    if all_predictions:
                        final_df = pd.concat(all_predictions)
                        final_df['versao'] = self.config.VERSION
                        global_rmse = np.sqrt(np.mean((final_df['real'] - final_df['previsao'])**2))
                        print(f"   📊 GLOBAL RMSE={global_rmse:.2f}")
                        self._save_to_delta(final_df)
                        self.success_models.append(model_name)
            
            except Exception as e:
                print(f"❌ Error {model_name}: {e}")
                traceback.print_exc()
                self.failed_models.append(model_name)

    def _calc_metrics_and_format(self, preds: Any, reals_full: Any, metrica_mes: str, model_name: str, store_ids: List[str]) -> Dict[str, Any]:
        if not isinstance(preds, list): preds = [preds]
        if not isinstance(reals_full, list): reals_full = [reals_full]

        valid_preds, valid_reals, res_dfs = [], [], []
        
        # --- PASSO CRUCIAL 3: Zipar com os IDs corretos ---
        # A ordem de preds é GARANTIDA ser a mesma de store_ids (Darts não embaralha)
        for ts_pred, ts_real_full, store_id in zip(preds, reals_full, store_ids):
            try:
                ts_real_sliced = ts_real_full.slice_intersect(ts_pred)
                valid_preds.append(ts_pred)
                valid_reals.append(ts_real_sliced)
                
                res_dfs.append(pd.DataFrame({
                    'data': ts_pred.time_index,
                    'previsao': ts_pred.values().flatten(),
                    'real': ts_real_sliced.values().flatten(),
                    'codigo_loja': store_id, # << USA O ID QUE VEIO DA LISTA, NÃO DA SÉRIE
                    'modelo': model_name,
                    'metrica_mes': metrica_mes
                }))
            except Exception:
                continue 
        
        metrics = {"smape": 0.0, "rmse": 0.0}
        if valid_preds:
            metrics["smape"] = float(np.mean(smape(valid_reals, valid_preds)))
            metrics["rmse"] = float(np.mean(rmse(valid_reals, valid_preds)))
            
        return {"metrics": metrics, "dfs": res_dfs}

    def _save_to_delta(self, pdf: pd.DataFrame) -> None:
        table_name = f"{self.config.CATALOG}.{self.config.SCHEMA}.resultado_metricas_treinamento_lojas"
        try:
            self.spark_session.createDataFrame(pdf).write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(table_name)
            self.spark_session.sql(f"OPTIMIZE {table_name}")
        except Exception as e:
            print(f"Save error: {e}")