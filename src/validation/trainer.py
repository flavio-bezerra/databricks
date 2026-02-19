"""
Módulo de Treinamento e Avaliação (Validation).

Gerencia a execução dos testes dos modelos. O principal fluxo é o "Walk-Forward Validation" (Validação Cruzada Temporal),
que simula o desempenho do modelo se fosse usado no mês passado, no mês anterior, etc.

Classes:
- ModelTrainer: Classe orquestradora que itera sobre modelos e janelas de tempo.
"""
import mlflow
import pickle
import pandas as pd
import numpy as np
import hashlib
import traceback
import os
from typing import Dict, List, Any, Optional
from darts import TimeSeries
from darts.metrics import mape, rmse, smape
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from pyspark.sql import SparkSession
from .models import DartsWrapper

class ModelTrainer:
    """Orquestrador para treinamento e validação walk-forward de modelos de séries temporais.

    Esta classe gerencia o ciclo de vida completo de modelos Darts, incluindo o
    treinamento em dados históricos, validação temporal (backtesting), cálculo
    de métricas de erro, registro no MLflow e persistência de resultados no Delta Lake.
    """

    def __init__(self, config: Any, models_dict: Dict[str, Any]):
        """Inicializa o ModelTrainer.

        Args:
            config: Objeto de configuração contendo parâmetros de catálogo, schema e datas.
            models_dict: Dicionário mapeando nomes de modelos para objetos de modelo Darts.
        """
        self.config = config
        self.models = models_dict
        self.success_models: List[str] = []
        self.failed_models: List[str] = []
        
        # Recupera ou inicializa a sessão Spark para persistência em tabelas Delta
        self.spark_session: Optional[SparkSession] = getattr(config, 'spark_session', None)
        if not self.spark_session:
             self.spark_session = SparkSession.builder.getOrCreate()

    def _extract_id_from_original(self, ts: TimeSeries) -> str:
        """Extrai de forma robusta o ID da loja das covariáveis estáticas da série.

        O Darts pode reorganizar metadados durante transformações. Este método
        garante a recuperação do identificador original para rastreabilidade.

        Args:
            ts: Objeto TimeSeries do Darts.

        Returns:
            ID da loja formatado como string ou 'UNKNOWN' se não localizado.
        """
        try:
            if ts.static_covariates is not None and not ts.static_covariates.empty:
                val = str(ts.static_covariates.index[0])
                # Log de anomalia se o ID capturado parecer incorreto
                if "target_vendas" in val:
                     print(f"⚠️ DEBUG ANOMALY: ID extraído é '{val}'. Verifique Static Covs.")
                
                # Normalização de IDs numéricos convertidos para string
                if val.endswith(".0"): val = val[:-2]
                return val
        except Exception:
            pass
        return "UNKNOWN"

    def _get_store_ids(self, series_list: List[TimeSeries]) -> List[str]:
        """Converte uma lista de TimeSeries em uma lista ordenada de IDs de lojas.

        Args:
            series_list: Lista de objetos TimeSeries.

        Returns:
            Lista de strings com os IDs das lojas.
        """
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
        """Executa o pipeline de treinamento e validação cruzada temporal.

        O processo segue três fases:
        1. Treinamento (Fit): Ajusta os modelos ao período histórico estático.
        2. Registro: Salva os modelos e metadados no MLflow Model Registry.
        3. Validação Walk-Forward: Simula o uso mensal dos modelos para gerar métricas.

        Args:
            train_series_static: Dados de vendas limitados ao fim do período de treino.
            train_covs_static: Covariáveis limitadas ao fim do período de treino.
            full_series_scaled: Séries de vendas completas (treino + valid) normalizadas.
            full_covariates_scaled: Covariáveis completas expandidas temporalmente.
            val_series_original: Séries de vendas originais (escala real) para cálculo de erro.
            target_pipeline: Objeto de transformação para desnormalizar predições.
            allow_new_run: Se True, cria uma nova Run no MLflow para cada modelo.
        """
        from contextlib import nullcontext

        mlflow.set_experiment(self.config.EXPERIMENT_NAME)
        
        # Sincronização de metadados das lojas
        ordered_store_ids = self._get_store_ids(full_series_scaled)
        
        # Persistência temporária das covariáveis para o artefato do modelo
        covariates_path = f"{self.config.VOLUME_ROOT}/temp/temp_future_covariates_v{self.config.VERSION}.pkl"
        os.makedirs(os.path.dirname(covariates_path), exist_ok=True)
        with open(covariates_path, "wb") as f:
            pickle.dump(full_covariates_scaled, f)
        
        # Ajuste do horizonte de validação respeitando o limite real dos dados de target
        data_end_limit = full_series_scaled[0].end_time()
        validation_range = pd.date_range(
            start=self.config.VAL_START_DATE, 
            end=min(pd.to_datetime(self.config.INGESTION_END), data_end_limit), 
            freq='MS'
        )

        for model_name, model in self.models.items():
            print(f"\n🚀 [Model: {model_name}] Iniciando Execução...")
            all_predictions = []
            
            run_context = mlflow.start_run(
                run_name=f"{model_name}_v{self.config.VERSION}", 
                nested=True
            ) if allow_new_run else nullcontext()

            try:
                with run_context:
                    print(f"   📝 Registrando metadados e parâmetros...")
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_param("n_stores", len(ordered_store_ids))
                    
                    # FASE 1: TREINAMENTO
                    print(f"   🏋️ Treinando modelo...")
                    kwargs = {}
                    if model.supports_past_covariates: kwargs['past_covariates'] = train_covs_static
                    if model.supports_future_covariates: kwargs['future_covariates'] = train_covs_static
                    
                    model.fit(train_series_static, **kwargs)
                    
                    # FASE 2: PERSISTÊNCIA E MLFLOW
                    filename = f"{model_name}_v{self.config.VERSION}.pkl"
                    local_path = f"{self.config.PATH_MODELS}/{filename}"
                    model.save(local_path)                  

                    full_model_name = f"{self.config.CATALOG}.{self.config.SCHEMA}.loja_{model_name}"
                    artifacts = {"darts_model": local_path}
                    if model.supports_future_covariates: artifacts["future_covariates"] = covariates_path                    

                    signature = ModelSignature(
                        inputs=Schema([ColSpec("long", "n")]), 
                        outputs=Schema([ColSpec("double", "prediction")])
                    )
                    
                    mlflow.pyfunc.log_model(
                        artifact_path=f"model_{model_name.lower()}",
                        python_model=DartsWrapper(),
                        artifacts=artifacts,
                        input_example=pd.DataFrame({"n": [self.config.FORECAST_HORIZON]}), 
                        signature=signature,
                        registered_model_name=full_model_name
                    )

                    # FASE 3: VALIDAÇÃO TEMPORAL (BACKTESTING)
                    print(f"   🔮 Validando ({len(validation_range)} meses)...")
                    
                    for month_start in validation_range:
                        context_cutoff = month_start - pd.Timedelta(days=1)
                        
                        # Garante que o ponto de corte está dentro do intervalo da série
                        if context_cutoff < full_series_scaled[0].start_time():
                            continue
                            
                        metrica_mes = month_start.strftime("%Y-%m")
                        val_context_series = [s.drop_after(context_cutoff) for s in full_series_scaled]
                        
                        predict_kwargs = {"n": self.config.FORECAST_HORIZON, 'series': val_context_series}
                        
                        if model.supports_past_covariates:
                             predict_kwargs['past_covariates'] = [c.drop_after(context_cutoff) for c in full_covariates_scaled]
                        if model.supports_future_covariates:
                             predict_kwargs['future_covariates'] = full_covariates_scaled 
                        
                        preds_scaled = model.predict(**predict_kwargs)
                        preds_inverse = target_pipeline.inverse_transform(preds_scaled, partial=True)
                        
                        metrics_month = self._calc_metrics_and_format(
                            preds_inverse, 
                            val_series_original, 
                            metrica_mes, 
                            model_name,
                            ordered_store_ids
                        )
                        
                        smape_m = metrics_month['metrics']['smape']
                        mlflow.log_metric(f"smape_{metrica_mes}", smape_m)
                        print(f"     📅 {metrica_mes}: SMAPE={smape_m:.2f}%, RMSE={metrics_month['metrics']['rmse']:.2f}")
                        
                        all_predictions.extend(metrics_month['dfs'])

                    # FINALIZAÇÃO DOS RESULTADOS
                    if all_predictions:
                        final_df = pd.concat(all_predictions)
                        final_df['versao'] = self.config.VERSION
                        global_rmse = np.sqrt(np.mean((final_df['real'] - final_df['previsao'])**2))
                        print(f"   📊 GLOBAL RMSE={global_rmse:.2f}")
                        
                        self._save_to_delta(final_df)
                        self.success_models.append(model_name)
            
            except Exception as e:
                print(f"❌ Erro no Modelo {model_name}: {e}")
                traceback.print_exc()
                self.failed_models.append(model_name)

    def _calc_metrics_and_format(self, preds: Any, reals_full: Any, metrica_mes: str, model_name: str, store_ids: List[str]) -> Dict[str, Any]:
        """Calcula métricas de erro e consolida predições para persistência.

        Args:
            preds: Lista de séries temporais preditas.
            reals_full: Lista de séries temporais com valores reais completos.
            metrica_mes: Identificador do mês da janela de validação.
            model_name: Nome do modelo sendo avaliado.
            store_ids: Lista ordenada de identificadores de loja.

        Returns:
            Dicionário contendo as métricas agregadas e lista de DataFrames detalhados.
        """
        if not isinstance(preds, list): preds = [preds]
        if not isinstance(reals_full, list): reals_full = [reals_full]

        valid_preds, valid_reals, res_dfs = [], [], []
        
        for ts_pred, ts_real_full, store_id in zip(preds, reals_full, store_ids):
            try:
                # Alinha o real com o período exato da predição
                ts_real_sliced = ts_real_full.slice_intersect(ts_pred)
                valid_preds.append(ts_pred)
                valid_reals.append(ts_real_sliced)
                
                res_dfs.append(pd.DataFrame({
                    'data': ts_pred.time_index,
                    'previsao': ts_pred.values().flatten(),
                    'real': ts_real_sliced.values().flatten(),
                    'codigo_loja': store_id,
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
        """Persiste os resultados de validação em uma tabela Delta no Unity Catalog.

        Args:
            pdf: DataFrame pandas com os resultados consolidados.
        """
        table_name = f"{self.config.CATALOG}.{self.config.SCHEMA}.resultado_metricas_treinamento_lojas"
        try:
            (self.spark_session.createDataFrame(pdf)
             .write.format("delta")
             .mode("append")
             .option("mergeSchema", "true")
             .saveAsTable(table_name))
            self.spark_session.sql(f"OPTIMIZE {table_name}")
        except Exception as e:
            print(f"Erro ao salvar tabela Delta: {e}")