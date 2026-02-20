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

import re
import mlflow
import pickle
import pandas as pd
import numpy as np
from typing import Any, List, Optional, Dict
from darts import TimeSeries
from darts.utils.timeseries_generation import datetime_attribute_timeseries

# Regex para identificar colunas de feriado POR LOJA (ex: 'feriado_1001').
# Colunas globais como 'feriado_nacional' NÃO casam com este padrão e permanecem
# nas covariáveis base, garantindo que o MinMaxScaler receba exatamente 62 features.
_PER_STORE_FERIADO_RE = re.compile(r'^feriado_\d+$')

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
        Gera features de calendário (dia da semana, trimestre, semana) on-the-fly.
        Isso evita que o usuário precise mandar colunas como 'dayofweek' manualmente.
        Segue exatamente o mesmo padrão do data.py (build_darts_objects).
        """
        if 'data' not in df.columns:
            return df
            
        dates_unique = df['data'].unique()
        ts_idx = pd.Index(dates_unique)
        if not isinstance(ts_idx, pd.DatetimeIndex):
            ts_idx = pd.to_datetime(ts_idx)
        ts_idx = ts_idx.sort_values()
        
        # Cria as features usando utilitários do Darts (idêntico ao data.py)
        ts_day     = datetime_attribute_timeseries(ts_idx, attribute="dayofweek", cyclic=True)
        ts_quarter = datetime_attribute_timeseries(ts_idx, attribute="quarter", one_hot=True)
        ts_week    = datetime_attribute_timeseries(ts_idx, attribute="week", cyclic=True)
        
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

        Fluxo interno:
        1. Extrai horizonte 'n' da coluna do input.
        2. Expande o DataFrame para cobrir datas futuras (_ensure_future_horizon).
        3. Gera features de calendário (_add_calendar_features).
        4. Constrói TimeSeries Darts POR LOJA em loop — mesmo padrão do data.py no treino:
           - Target: TimeSeries com .with_static_covariates() → exatamente 4 colunas estáticas
             (evita o bug do from_group_dataframe que adiciona 'codigo_loja' como 5ª coluna)
           - Covariáveis: colunas base (globais + calendário) + is_feriado renomeado
             para feriado_{store_id} (resolve o bug do metadata hardcodar feriado_1001)
        5. Aplica pipeline de scaling (target, static, covariate).
        6. Chama model.predict() e inverte o scaling.
        """
        # Re-importação necessária pois o pickle do PythonModel pode perder referências globais
        import re
        import pickle
        import pandas as pd
        import numpy as np
        import traceback
        from darts import TimeSeries
        from darts.utils.timeseries_generation import datetime_attribute_timeseries

        # Helpers locais (garantem disponibilidade mesmo após deserialização do pickle)
        def _is_per_store_feriado(col: str) -> bool:
            """True apenas para colunas do tipo 'feriado_{ID_numérico}' (ex: 'feriado_1001').
            Colunas globais como 'feriado_nacional' retornam False."""
            return bool(re.match(r'^feriado_\d+$', col))

        # ── 1. HORIZONTE DE PREVISÃO ────────────────────────────────────────────
        n = 1
        if isinstance(model_input, pd.DataFrame) and 'n' in model_input.columns:
            try:
                n = int(model_input.iloc[0]['n'])
            except Exception:
                pass

        predict_kwargs = {"n": n}

        try:
            if isinstance(model_input, pd.DataFrame) and len(model_input) > 1:

                # ── 2. PRÉ-PROCESSAMENTO ────────────────────────────────────────
                # Remove colunas duplicadas resultantes de merges anteriores
                model_input = model_input.loc[:, ~model_input.columns.duplicated()]
                # Expande o DataFrame para cobrir o futuro (criar "slots" para o Darts)
                model_input = self._ensure_future_horizon(model_input, n)
                # Gera features de calendário idênticas ao treino (dayofweek, quarter, week)
                # NUNCA vêm pré-calculadas do notebook — geradas aqui para garantir consistência
                model_input = self._add_calendar_features(model_input)
                model_input['data'] = pd.to_datetime(model_input['data'])

                # Limpa código de loja (remove ".0" de floats convertidos para string)
                model_input['codigo_loja'] = (
                    model_input['codigo_loja']
                    .astype(str)
                    .str.replace(r'\.0$', '', regex=True)
                )

                # ── 3. RESOLUÇÃO DE COLUNAS (metadata do treino) ─────────────────
                if hasattr(self, 'metadata') and self.metadata:
                    ordered_static = list(self.metadata.get("static_cols_order", []))
                    covariate_cols_meta = list(self.metadata.get("covariate_cols_order", []))

                    # codigo_loja não é covariável estática — é o índice/grupo
                    if "codigo_loja" in ordered_static:
                        ordered_static.remove("codigo_loja")

                    # Separação entre colunas de feriado GLOBAIS e POR LOJA:
                    #   - 'feriado_nacional' → global → fica em base_cov_cols → vai para o scaler
                    #   - 'feriado_1001'     → por loja → tratada separadamente por loja
                    # IMPORTANTE: usar startswith('feriado_') seria errado pois excluiria
                    # 'feriado_nacional', causando MinMaxScaler com 61 features vs 62 esperadas.
                    base_cov_cols = [c for c in covariate_cols_meta if not _is_per_store_feriado(c)]
                    has_feriado_in_meta = any(_is_per_store_feriado(c) for c in covariate_cols_meta)
                else:
                    # Fallback heurístico se não houver metadata
                    possible_static = ["cluster_loja", "sigla_uf", "tipo_loja", "modelo_loja"]
                    ordered_static = [c for c in possible_static if c in model_input.columns]
                    reserved = set(['data', 'codigo_loja', 'target_vendas', 'n', 'is_feriado'] + ordered_static)
                    # Exclui apenas colunas de feriado por loja (ex: feriado_1001), mantendo feriado_nacional
                    base_cov_cols = [c for c in model_input.columns if c not in reserved and not _is_per_store_feriado(c)]
                    has_feriado_in_meta = 'is_feriado' in model_input.columns

                # ── 4. CONSTRUÇÃO DE SÉRIES DARTS POR LOJA ───────────────────────
                #
                # PORQUÊ loop e não from_group_dataframe:
                #   - from_group_dataframe(group_cols="codigo_loja") adiciona automaticamente
                #     'codigo_loja' como COLUNA das static_covariates, gerando 5 dimensões
                #     enquanto o modelo foi treinado com 4 (via .with_static_covariates()).
                #   - Isso causa: "boolean index did not match... dimension is 5 but is 4"
                #
                # PORQUÊ renomear is_feriado → feriado_{store_id}:
                #   - O treino (data.py) cria 'feriado_{store_id}' por loja.
                #   - O metadata.covariate_cols_order guarda o nome da PRIMEIRA loja do treino.
                #   - Para qualquer outra loja, a coluna 'feriado_XXXX' não existe no input.
                #   - Solução: renomear 'is_feriado' para o nome correto por loja.

                target_series_list = []
                covariate_series_list = []
                store_ids_map = []

                for store_id_raw, store_df in model_input.groupby('codigo_loja'):
                    store_id = str(store_id_raw)
                    store_df = store_df.copy().sort_values('data').reset_index(drop=True)

                    # ── 4a. TARGET ──────────────────────────────────────────────
                    history_df = store_df.dropna(subset=['target_vendas'])
                    if history_df.empty:
                        print(f"⚠️  Loja {store_id}: sem histórico de target. Pulando.")
                        continue

                    # Constrói TimeSeries do target com nome da coluna = store_id
                    ts_target = TimeSeries.from_dataframe(
                        history_df.rename(columns={'target_vendas': store_id}),
                        time_col='data',
                        value_cols=[store_id],
                        freq='D',
                        fill_missing_dates=True,
                        fillna_value=0.0
                    )

                    # Adiciona static_covariates via .with_static_covariates()
                    # → garante EXATAMENTE as mesmas 4 colunas do treino, sem codigo_loja
                    avail_static = [c for c in ordered_static if c in history_df.columns]
                    if avail_static:
                        static_df = history_df[avail_static].iloc[0:1].copy()
                        static_df.index = pd.Index([store_id], name='codigo_loja')
                        ts_target = ts_target.with_static_covariates(static_df)

                    target_series_list.append(ts_target)
                    store_ids_map.append(store_id)

                    # ── 4b. COVARIÁVEIS ─────────────────────────────────────────
                    # Colunas base (globais + calendário) disponíveis no input desta loja
                    avail_cov = [c for c in base_cov_cols if c in store_df.columns]

                    # Renomeia is_feriado → feriado_{store_id} (padrão do treino)
                    feriado_col = f'feriado_{store_id}'
                    if 'is_feriado' in store_df.columns:
                        store_df[feriado_col] = store_df['is_feriado'].fillna(0.0)
                        avail_cov.append(feriado_col)
                    elif has_feriado_in_meta:
                        # Feriado esperado mas ausente → preenche com zeros
                        store_df[feriado_col] = 0.0
                        avail_cov.append(feriado_col)

                    if avail_cov:
                        ts_cov = TimeSeries.from_dataframe(
                            store_df,
                            time_col='data',
                            value_cols=avail_cov,
                            freq='D',
                            fill_missing_dates=True,
                            fillna_value=0.0
                        )
                        covariate_series_list.append(ts_cov)
                    else:
                        covariate_series_list.append(None)

                if not target_series_list:
                    raise ValueError("Nenhuma série de target válida encontrada no input.")

                # ── 5. TRANSFORM (SCALING) ────────────────────────────────────────
                has_target_scaler  = hasattr(self.pipeline, 'target_pipeline')
                has_static_encoder = hasattr(self.pipeline, 'static_pipeline')
                has_cov_scaler     = hasattr(self.pipeline, 'covariate_pipeline')

                final_series_input     = []
                final_covariates_input = []

                for i, ts_target in enumerate(target_series_list):
                    ts_proc = ts_target
                    if has_target_scaler:  ts_proc = self.pipeline.target_pipeline.transform(ts_proc)
                    if has_static_encoder: ts_proc = self.pipeline.static_pipeline.transform(ts_proc)
                    final_series_input.append(ts_proc)

                    ts_cov = covariate_series_list[i] if i < len(covariate_series_list) else None
                    if ts_cov is not None:
                        if has_cov_scaler: ts_cov = self.pipeline.covariate_pipeline.transform(ts_cov)
                        final_covariates_input.append(ts_cov)

                predict_kwargs['series'] = final_series_input
                if final_covariates_input and len(final_covariates_input) == len(final_series_input):
                    predict_kwargs['future_covariates'] = final_covariates_input

            # ── 6. PREDIÇÃO ───────────────────────────────────────────────────────
            pred_series_list = self.model.predict(**predict_kwargs)
            if not isinstance(pred_series_list, list):
                pred_series_list = [pred_series_list]

            # ── 7. PÓS-PROCESSAMENTO (Inverse Transform + Montagem do DF) ────────
            final_df_list = []
            for pred_series, original_store_id in zip(pred_series_list, store_ids_map):
                pred_inverse = self.pipeline.inverse_transform(pred_series, partial=True)
                df_pred = pred_inverse.pd_dataframe()

                df_pred['codigo_loja'] = original_store_id
                col_val = [c for c in df_pred.columns if c not in ['codigo_loja', 'data']][0]
                df_pred.rename(columns={col_val: 'previsao_venda'}, inplace=True)
                final_df_list.append(df_pred)

            return pd.concat(final_df_list).reset_index().rename(columns={'data': 'data_previsao'})

        except Exception as e:
            # ── FALLBACK DE SEGURANÇA ─────────────────────────────────────────────
            # Retorna DataFrame com zeros em vez de propagar a exceção.
            # Isso impede que o Spark aborte o job inteiro por causa de 1 loja com problema.
            print(f"⚠️ [UnifiedForecaster] Erro Crítico: {str(e)}")
            traceback.print_exc()

            fallback_dates = pd.date_range(start="2025-01-01", periods=n, freq="D")
            df_error = pd.DataFrame({
                'data_previsao': fallback_dates,
                'previsao_venda': np.zeros(n, dtype=float),
                'codigo_loja': ["ERROR_FALLBACK"] * n
            })
            return df_error