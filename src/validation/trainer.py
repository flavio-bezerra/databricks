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
from typing import Dict, List, Any, Optional
from darts import TimeSeries
from darts.metrics import mape, rmse, smape
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from pyspark.sql import SparkSession
from .models import DartsWrapper

class ModelTrainer:
    """
    Motor de treinamento e validação.
    
    Esta classe recebe dados preparados e uma lista de modelos Darts para:
    1. Treinar em um período de histórico fixo.
    2. Testar mês a mês (sliding window) no período de validação.
    3. Calcular métricas de erro (RMSE, SMAPE).
    4. Registrar tudo no MLflow.
    5. Salvar resultados detalhados (predição vs real por loja e dia) em tabela Delta.
    """
    def __init__(self, config: Any, models_dict: Dict[str, Any]):
        self.config = config
        self.models = models_dict # Dicionário {NomeModelo: ObjetoModeloDarts}
        self.success_models: List[str] = []
        self.failed_models: List[str] = []
        
        # Obtém sessão Spark (necessária para salvar tabelas)
        self.spark_session: Optional[SparkSession] = getattr(config, 'spark_session', None)
        if not self.spark_session:
             self.spark_session = SparkSession.builder.getOrCreate()

    def _extract_id_from_original(self, ts: TimeSeries) -> str:
        """
        Helper para recuperar o ID da loja de uma série temporal.
        
        Desafio: Darts otimiza o armazenamento e às vezes o metadado do ID se perde ou muda de lugar
        durante transformações. Aqui tentamos recuperar esse ID de forma robusta olhando as Covariáveis Estáticas.
        """
        try:
            if ts.static_covariates is not None:
                # O Darts costuma colocar o ID do grupo como índice das covariáveis estáticas
                if not ts.static_covariates.empty:
                    val = str(ts.static_covariates.index[0])
                    # Verificação de segurança: se vier uma string estranha (target name), logamos anomalia
                    if "target_vendas" in val:
                         print(f"⚠️ DEBUG ANOMALY: ID extracted is '{val}'. Static Covs:\n{ts.static_covariates.head()}")
                    
                    # Limpeza técnica: remove sufixo .0 que pandas pode adicionar em floats
                    if val.endswith(".0"): val = val[:-2]
                    return val
        except:
            pass
        return "UNKNOWN"

    def _get_store_ids(self, series_list: List[TimeSeries]) -> List[str]:
        # Aplica extração para lista inteira
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
        """
        Executa o loop principal de Treino e Validação.
        
        Args:
            train_series_static: Séries cortadas apenas no período de treino (para .fit()).
            full_series_scaled: Séries completas (Treino + Valid) escaladas.
            val_series_original: Séries completas na escala REAL (para calcular erro R$).
            target_pipeline: Objeto com scalers para reverter transformações.
        """
        from contextlib import nullcontext

        mlflow.set_experiment(self.config.EXPERIMENT_NAME)
        
        # --- PREPARAÇÃO DE METADADOS ---
        # Capturamos a ordem exata dos IDs das lojas na lista de input.
        # Isso é crucial porque o modelo vai devolver predições numa lista e precisamos saber quem é quem.
        ordered_store_ids = self._get_store_ids(full_series_scaled)
        
        # Hash dos IDs para garantir integridade e rastreabilidade se os dados mudarem
        store_ids_str = ",".join(ordered_store_ids)
        stores_hash = hashlib.md5(store_ids_str.encode()).hexdigest()[:8]

        # Salva as covariáveis (features futuras) em arquivo temporário para anexar ao modelo MLflow
        covariates_path = f"{self.config.VOLUME_ROOT}/temp/temp_future_covariates_v{self.config.VERSION}.pkl"
        import os
        os.makedirs(os.path.dirname(covariates_path), exist_ok=True)
        with open(covariates_path, "wb") as f:
            pickle.dump(full_covariates_scaled, f)
        
        # Define os meses em que faremos a validação (Backtesting manual)
        validation_range = pd.date_range(start=self.config.VAL_START_DATE, end=self.config.INGESTION_END, freq='MS')

        # --- LOOP POR MODELO ---
        for model_name, model in self.models.items():
            print(f"\\n🚀 [Model: {model_name}] Iniciando Processo...")
            all_predictions = []
            
            # Contexto MLflow: Cria uma "Run" para agrupar logs deste modelo
            run_context = mlflow.start_run(run_name=f"{model_name}_v{self.config.VERSION}", nested=True) if allow_new_run else nullcontext()

            try:
                with run_context:
                    # Log de parâmetros
                    print(f"   📝 Registrando metadados...")
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_param("n_stores", len(ordered_store_ids))
                    
                    # --- FASE 1: TREINAMENTO (FIT) ---
                    # O modelo aprende padrões usando apenas os dados até TRAIN_END_DATE
                    print(f"   🏋️ Treinando...")
                    kwargs = {}
                    # Injeta covariáveis se o modelo suportar
                    if model.supports_past_covariates: kwargs['past_covariates'] = train_covs_static
                    if model.supports_future_covariates: kwargs['future_covariates'] = train_covs_static
                    
                    model.fit(train_series_static, **kwargs)
                    
                    # --- FASE 2: REGISTRO DO MODELO ---
                    # Salvamos o modelo treinado para poder usá-lo depois sem re-treinar
                    filename = f"{model_name}_v{self.config.VERSION}.pkl"
                    local_path = f"{self.config.PATH_MODELS}/{filename}"
                    model.save(local_path)                  

                    full_model_name = f"{self.config.CATALOG}.{self.config.SCHEMA}.loja_{model_name}"
                    artifacts = {"darts_model": local_path}
                    if model.supports_future_covariates: artifacts["future_covariates"] = covariates_path                    

                    # Definição de assinatura I/O para o MLflow entender os dados
                    input_schema = Schema([ColSpec("long", "n")]) 
                    output_schema = Schema([ColSpec("double", "prediction")])
                    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
                    
                    # Log do modelo customizado (Wrapper)
                    mlflow.pyfunc.log_model(
                        artifact_path=f"model_{model_name.lower()}",
                        python_model=DartsWrapper(),
                        artifacts=artifacts,
                        input_example=pd.DataFrame({"n": [self.config.FORECAST_HORIZON]}), 
                        signature=signature,
                        registered_model_name=full_model_name
                    )

                    # --- FASE 3: VALIDAÇÃO (WALK-FORWARD) ---
                    # Simulamos o passado. Para cada mês no intervalo de validação:
                    # 1. Cortamos os dados como se estivéssemos naquele dia.
                    # 2. Pedimos previsão para N dias à frente.
                    # 3. Comparamos com o que realmente aconteceu.
                    print(f"   🔮 Validando ({len(validation_range)} meses)...")
                    
                    for month_start in validation_range:
                        # Data de corte: o dia anterior ao inicio do mês de previsão
                        context_cutoff = month_start - pd.Timedelta(days=1)
                        metrica_mes = month_start.strftime("%Y-%m")
                        
                        # Filtra dados para não vazar futuro ("drop_after")
                        val_context_series = [s.drop_after(context_cutoff) for s in full_series_scaled]
                        
                        predict_kwargs = {"n": self.config.FORECAST_HORIZON, 'series': val_context_series}
                        
                        # Ajusta covariáveis também para o corte temporal
                        if model.supports_past_covariates:
                             val_context_covs = [c.drop_after(context_cutoff) for c in full_covariates_scaled]
                             predict_kwargs['past_covariates'] = val_context_covs
                        if model.supports_future_covariates:
                             # Features futuras (feriados) nós SABEMOS o futuro, então passamos tudo
                             predict_kwargs['future_covariates'] = full_covariates_scaled 
                        
                        # Gera Predição (na escala normalizada)
                        preds_scaled = model.predict(**predict_kwargs)
                        
                        # Desnormaliza para escala real (R$) para calcular erro
                        preds_inverse = target_pipeline.inverse_transform(preds_scaled, partial=True)
                        
                        # Calcula erro e formata DataFrame de resultados
                        metrics_month = self._calc_metrics_and_format(
                            preds_inverse, 
                            val_series_original, 
                            metrica_mes, 
                            model_name,
                            ordered_store_ids # Passamos a lista confiável de IDs para garantir o match
                        )
                        
                        smape_m = metrics_month['metrics']['smape']
                        rmse_m = metrics_month['metrics']['rmse']
                        mlflow.log_metric(f"smape_{metrica_mes}", smape_m)
                        print(f"     📅 {metrica_mes}: SMAPE={smape_m:.2f}%, RMSE={rmse_m:.2f}")
                        
                        # Acumula resultados (DataFrame) na lista
                        all_predictions.extend(metrics_month['dfs'])

                    # --- FINALIZAÇÃO ---
                    if all_predictions:
                        final_df = pd.concat(all_predictions)
                        final_df['versao'] = self.config.VERSION
                        global_rmse = np.sqrt(np.mean((final_df['real'] - final_df['previsao'])**2))
                        print(f"   📊 GLOBAL RMSE={global_rmse:.2f}")
                        
                        # Salva tabela final de resultados no Delta Lake
                        self._save_to_delta(final_df)
                        self.success_models.append(model_name)
            
            except Exception as e:
                print(f"❌ Error {model_name}: {e}")
                traceback.print_exc()
                self.failed_models.append(model_name)

    def _calc_metrics_and_format(self, preds: Any, reals_full: Any, metrica_mes: str, model_name: str, store_ids: List[str]) -> Dict[str, Any]:
        """ Helper para empacotar predições em DataFrames e calcular erro. """
        if not isinstance(preds, list): preds = [preds]
        if not isinstance(reals_full, list): reals_full = [reals_full]

        valid_preds, valid_reals, res_dfs = [], [], []
        
        # Zipa Predição X Real X ID da Loja.
        # A suposição FORTE aqui é que o Darts mantém a ordem da lista de entrada na saída.
        for ts_pred, ts_real_full, store_id in zip(preds, reals_full, store_ids):
            try:
                # Pega apenas o pedaço da série real que corresponde ao período previsto (slice_intersect)
                ts_real_sliced = ts_real_full.slice_intersect(ts_pred)
                valid_preds.append(ts_pred)
                valid_reals.append(ts_real_sliced)
                
                # Cria DataFrame resultante para persistência
                res_dfs.append(pd.DataFrame({
                    'data': ts_pred.time_index,
                    'previsao': ts_pred.values().flatten(),
                    'real': ts_real_sliced.values().flatten(),
                    'codigo_loja': store_id, # Associa explicitamente o ID que veio da lista ordenada
                    'modelo': model_name,
                    'metrica_mes': metrica_mes
                }))
            except Exception:
                # Se não houver interseção temporal (ex: feriado fecha loja e não tem venda real), pula
                continue 
        
        metrics = {"smape": 0.0, "rmse": 0.0}
        if valid_preds:
            metrics["smape"] = float(np.mean(smape(valid_reals, valid_preds)))
            metrics["rmse"] = float(np.mean(rmse(valid_reals, valid_preds)))
            
        return {"metrics": metrics, "dfs": res_dfs}

    def _save_to_delta(self, pdf: pd.DataFrame) -> None:
        """ Persiste o DataFrame Pandas como tabela Delta Spark. """
        table_name = f"{self.config.CATALOG}.{self.config.SCHEMA}.resultado_metricas_treinamento_lojas"
        try:
            # createDataFrame pode ser lento para dados massivos, mas ok para resultados agregados
            self.spark_session.createDataFrame(pdf).write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(table_name)
            self.spark_session.sql(f"OPTIMIZE {table_name}")
        except Exception as e:
            print(f"Save error: {e}")