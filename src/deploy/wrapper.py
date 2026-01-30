"""
Módulo de Wrapper de Inferência (Deploy).

Este módulo define a classe `UnifiedForecaster`, que é o modelo final que vai para produção.
Diferente do `DartsWrapper` (usado apenas para validação e métricas), este wrapper é "autossuficiente".
Ele carrega não só o modelo preditivo, mas também todo o pipeline de escalonamento e a lógica de engenharia
de features (calendário, feriados), permitindo que o usuário envie apenas o histórico de vendas cru
e receba a previsão, sem precisar pré-processar os dados manualmente.

Classes:
- UnifiedForecaster: O modelo de produção All-in-One.
"""

import mlflow
import pickle
import pandas as pd
import numpy as np
from typing import Any, List, Optional, Dict
from darts import TimeSeries
from darts.utils.timeseries_generation import datetime_attribute_timeseries

class UnifiedForecaster(mlflow.pyfunc.PythonModel):
    """
    Wrapper Robusto para Inferência em Produção (Spark/Rest API).
    
    Principais features:
    1. Pipeline Embutido: Aplica automaticamente os mesmos Scalers/Encoders usados no treino.
    2. Auto-Extensão: Se o usuário pede previsão para 7 dias mas não manda linhas futuras, 
       esta classe cria as linhas de data futuras automaticamente.
    3. Resiliência: Embala erros em retornos "Dummy" (fallback) para evitar que um cluster Spark inteiro 
       falhe por causa de uma loja problemática.
    """
    def load_context(self, context: Any) -> None:
        """ Carrega modelo e pipeline (scalers) dos artefatos do MLflow. """
        with open(context.artifacts["darts_model"], "rb") as f:
            self.model = pickle.load(f)
        with open(context.artifacts["pipeline"], "rb") as f:
            self.pipeline = pickle.load(f)
        
        # Metadados opcionais (ex: ordem das colunas usadas no treino)
        if "metadata" in context.artifacts:
            with open(context.artifacts["metadata"], "rb") as f:
                self.metadata = pickle.load(f)
        else:
            self.metadata = {}

    def _ensure_future_horizon(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """
        Garante que o DataFrame tenha linhas suficientes cobrindo o futuro desejado.
        
        Cenário: Queremos prever D+1 até D+7, mas o input só tem dados até D (hoje).
        Ação: Criamos 7 novas linhas com timestamps D+1...D+7 e 'target_vendas' NaN.
        Isso é necessário porque o Darts precisa de "slots" de tempo futuros onde ele vai encaixar as covariáveis.
        """
        # Safety Buffer: Margem de segurança para lags (se usamos lag-3, precisamos de histórico anterior tbm)
        safety_buffer = self.metadata.get("max_lag", 15) + 2
        
        if 'data' not in df.columns or 'codigo_loja' not in df.columns:
            return df
        
        df['data'] = pd.to_datetime(df['data'])
        
        # 1. Identifica até onde temos histórico real (target não nulo)
        df_history = df.dropna(subset=['target_vendas'])
        if df_history.empty:
            last_history_date = df['data'].max()
        else:
            last_history_date = df_history['data'].max()
            
        # 2. Verifica até onde o input total vai
        last_input_date = df['data'].max()
        required_date = last_history_date + pd.Timedelta(days=n + safety_buffer)
        
        # Se já tivermos linhas futuras suficientes (o usuário mandou input estendido), retorna.
        if last_input_date >= required_date:
            return df
            
        # 3. Caso contrário, gera linhas futuras artificiais
        future_horizon = (required_date - last_input_date).days + 1
        if future_horizon < 1: future_horizon = 1
        
        future_dates = pd.date_range(start=last_input_date + pd.Timedelta(days=1), periods=future_horizon, freq='D')
        
        # Para cada loja, copia os atributos estáticos da última linha conhecida e cria novas datas
        last_rows = df.sort_values('data').groupby('codigo_loja').tail(1)
        future_dfs = []
        for _, row in last_rows.iterrows():
            temp_df = pd.DataFrame({'data': future_dates})
            for col in df.columns:
                # Replica colunas estáticas, deixa dinâmicas como NaN (serão preenchidas ou ignoradas)
                if col not in ['data', 'target_vendas']: 
                    temp_df[col] = row[col]
            temp_df['target_vendas'] = np.nan
            future_dfs.append(temp_df)
            
        if not future_dfs: return df

        df_future = pd.concat(future_dfs)
        df_extended = pd.concat([df, df_future], ignore_index=True).sort_values(['codigo_loja', 'data'])
        
        return df_extended

    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gera features de calendário (dia da semana, dia do ano) on-the-fly.
        Isso evita que o usuário precise mandar colunas como 'dayofweek' manualmente.
        """
        if 'data' not in df.columns:
            return df
            
        dates_unique = df['data'].unique()
        ts_idx = pd.Index(dates_unique)
        if not isinstance(ts_idx, pd.DatetimeIndex):
            ts_idx = pd.to_datetime(ts_idx)
        ts_idx = ts_idx.sort_values()
        
        # Cria as features usando utilitários do Darts
        ts_day = datetime_attribute_timeseries(ts_idx, attribute="dayofweek", cyclic=True)
        ts_quarter = datetime_attribute_timeseries(ts_idx, attribute="quarter", one_hot=True)
        ts_week = datetime_attribute_timeseries(ts_idx, attribute="week", cyclic=True)
        
        # Junta tudo
        ts_full = ts_day.stack(ts_quarter).stack(ts_week)
        df_cal = ts_full.pd_dataframe().reset_index().rename(columns={'time': 'data'})
        
        df['data'] = pd.to_datetime(df['data'])
        
        # Remove se já existirem para evitar duplicidade
        cal_cols = [c for c in df_cal.columns if c != 'data']
        df = df.drop(columns=[c for c in cal_cols if c in df.columns], errors='ignore')
        
        # Merge de volta ao dataframe principal
        df_merged = pd.merge(df, df_cal, on='data', how='left')
        return df_merged

    def predict(self, context: Any, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Método mestre de inferência.
        Aceita um DataFrame Pandas com histórico de vendas e retorna DataFrame com previsões.
        """
        # Re-importação necessária pois o pickle do PythonModel pode perder referências globais
        import mlflow
        import pickle
        import pandas as pd
        import numpy as np
        from darts import TimeSeries
        from darts.utils.timeseries_generation import datetime_attribute_timeseries
        
        # 1. Definição do Horizonte (n)
        # Tenta ler a coluna 'n' do input (passada pelo usuário), senão default=1
        n = 1
        if isinstance(model_input, pd.DataFrame) and 'n' in model_input.columns:
            try:
                n = int(model_input.iloc[0]['n'])
            except Exception:
                pass
        
        predict_kwargs = {"n": n}

        try:
            if isinstance(model_input, pd.DataFrame) and len(model_input) > 1:
                
                # --- PRÉ-PROCESSAMENTO (Feature Engineering) ---
                model_input = model_input.loc[:, ~model_input.columns.duplicated()]
                # Expande linhas para cobrir o futuro
                model_input = self._ensure_future_horizon(model_input, n)
                # Adiciona features de data
                model_input = self._add_calendar_features(model_input)

                # --- ORDENAÇÃO DE COLUNAS ---
                # Garante que as colunas estejam na mesma ordem que o modelo viu no treino.
                if hasattr(self, 'metadata') and self.metadata:
                    ordered_static = self.metadata.get("static_cols_order", [])
                    ordered_covariates = self.metadata.get("covariate_cols_order", [])
                    if "codigo_loja" in ordered_static:
                        ordered_static.remove("codigo_loja")
                else:
                    # Fallback heuristic se não tiver metadata
                    possible_static = ["cluster_loja", "sigla_uf", "tipo_loja", "modelo_loja"]
                    ordered_static = [c for c in possible_static if c in model_input.columns]
                    ordered_static = [c for c in ordered_static if c in model_input.columns]
                    reserved = set(['data', 'codigo_loja', 'target_vendas', 'n'] + ordered_static)
                    ordered_covariates = [c for c in model_input.columns if c not in reserved]

                # Preenche covariáveis faltantes com 0.0 (segurança)
                for col in ordered_covariates:
                    if col not in model_input.columns:
                        model_input[col] = 0.0

                # --- CONSTRUÇÃO DE SÉRIES DARTS ---
                df_history = model_input.dropna(subset=['target_vendas'])
                
                # Série Alvo (Target)
                target_series_list = TimeSeries.from_group_dataframe(
                    df_history,
                    group_cols="codigo_loja",
                    time_col="data",
                    value_cols="target_vendas",
                    static_cols=ordered_static,
                    freq='D',
                    fill_missing_dates=True,
                    fillna_value=0.0
                )

                # Mapeamento ID -> Objeto (para rastreabilidade)
                store_ids_map = []
                for ts in target_series_list:
                    if "codigo_loja" in ts.static_covariates.columns:
                        raw_id = str(ts.static_covariates["codigo_loja"].iloc[0])
                    else:
                        raw_id = str(ts.static_covariates.index[0])
                    store_ids_map.append(raw_id)

                # Séries Covariáveis (Features dinâmicas)
                cov_dict = {}
                if ordered_covariates:
                    covariate_series_list = TimeSeries.from_group_dataframe(
                        model_input,
                        group_cols="codigo_loja",
                        time_col="data",
                        value_cols=ordered_covariates, 
                        freq='D',
                        fill_missing_dates=True,
                        fillna_value=0.0
                    )
                    for ts in covariate_series_list:
                        if "codigo_loja" in ts.static_covariates.columns:
                            c_id = str(ts.static_covariates["codigo_loja"].iloc[0])
                        else:
                            c_id = str(ts.static_covariates.index[0])
                        cov_dict[c_id] = ts

                # --- TRANSFORM (SCALING) ---
                # Aplica o pipeline carregado do pickle (mesma média/desvio do treino)
                final_series_input = []
                final_covariates_input = []
                
                has_target_scaler = hasattr(self.pipeline, 'target_pipeline')
                has_static_encoder = hasattr(self.pipeline, 'static_pipeline')
                has_cov_scaler = hasattr(self.pipeline, 'covariate_pipeline')

                for i, ts_target in enumerate(target_series_list):
                    store_id = store_ids_map[i]
                    
                    ts_proc = ts_target
                    if has_target_scaler: ts_proc = self.pipeline.target_pipeline.transform(ts_proc)
                    if has_static_encoder: ts_proc = self.pipeline.static_pipeline.transform(ts_proc)
                    final_series_input.append(ts_proc)

                    if store_id in cov_dict:
                        ts_cov = cov_dict[store_id]
                        if has_cov_scaler: ts_cov = self.pipeline.covariate_pipeline.transform(ts_cov)
                        final_covariates_input.append(ts_cov)
                
                predict_kwargs['series'] = final_series_input
                if final_covariates_input:
                    predict_kwargs['future_covariates'] = final_covariates_input

            # --- PREDIÇÃO REAL ---
            pred_series_list = self.model.predict(**predict_kwargs)
            if not isinstance(pred_series_list, list): pred_series_list = [pred_series_list]
            
            # --- PÓS-PROCESSAMENTO ---
            final_df_list = []
            
            for pred_series, original_store_id in zip(pred_series_list, store_ids_map):
                # Desfaz o scaling (Inverte transformação)
                pred_inverse = self.pipeline.inverse_transform(pred_series, partial=True)
                df = pred_inverse.pd_dataframe()
                
                # Reconstrói dataframe de resposta
                df['codigo_loja'] = original_store_id
                col_val = [c for c in df.columns if c not in ['codigo_loja', 'data']][0]
                df.rename(columns={col_val: 'previsao_venda'}, inplace=True)
                final_df_list.append(df)
                
            return pd.concat(final_df_list).reset_index().rename(columns={'data': 'data_previsao'})

        except Exception as e:
            # --- TRATAMENTO DE ERRO / FALHA PARCIAL ---
            print(f"⚠️ [UnifiedForecaster] Erro Crítico: {str(e)}")
            
            # Retorno de segurança: DataFrame vazio ou com zeros, mas com Schema correto.
            # Isso impede que o Spark aborte o job inteiro pq 1 batch falhou.
            fallback_dates = pd.date_range(start="2025-01-01", periods=n, freq="D")
            df_error = pd.DataFrame({
                'data_previsao': fallback_dates,
                'previsao_venda': np.zeros(n, dtype=float),
                'codigo_loja': ["ERROR_FALLBACK"] * n
            })
            return df_error