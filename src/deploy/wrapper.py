import mlflow
import pickle
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.utils.timeseries_generation import datetime_attribute_timeseries

class UnifiedForecaster(mlflow.pyfunc.PythonModel):
    """
    Wrapper "All-in-One" para deploy.
    Detecta se o input já possui futuro (covariáveis reais) ou se precisa estender artificialmente.
    Corrige erro de ordenação de datas e blinda IDs.
    """
    def load_context(self, context):
        with open(context.artifacts["darts_model"], "rb") as f:
            self.model = pickle.load(f)
        with open(context.artifacts["pipeline"], "rb") as f:
            self.pipeline = pickle.load(f)
        
        if "metadata" in context.artifacts:
            with open(context.artifacts["metadata"], "rb") as f:
                self.metadata = pickle.load(f)
        else:
            self.metadata = {}

    def _ensure_future_horizon(self, df, n):
        """
        Verifica se o DataFrame cobre o horizonte futuro necessário.
        Se cobrir (usuário passou dados reais), mantém.
        Se não cobrir (usuário passou só histórico), estende com Forward Fill (fallback).
        """
        if 'DATA' not in df.columns or 'CODIGO_LOJA' not in df.columns:
            return df
        
        # Garante datetime
        df['DATA'] = pd.to_datetime(df['DATA'])
        
        # 1. Identifica até onde vai o histórico real de vendas (Target não nulo)
        df_history = df.dropna(subset=['TARGET_VENDAS'])
        if df_history.empty:
            last_history_date = df['DATA'].max() # Fallback
        else:
            last_history_date = df_history['DATA'].max()
            
        # 2. Identifica até onde vai o input total (pode ter futuro de mercado)
        last_input_date = df['DATA'].max()
        
        # Buffer de segurança para lags
        safety_buffer = 15
        required_date = last_history_date + pd.Timedelta(days=n + safety_buffer)
        
        # --- VERIFICAÇÃO INTELIGENTE ---
        # Se os dados fornecidos já vão longe o suficiente no futuro, NÃO estendemos.
        if last_input_date >= required_date:
            return df
            
        # --- EXTENSÃO ARTIFICIAL (FALLBACK) ---
        future_horizon = (required_date - last_input_date).days + 1
        if future_horizon < 1: future_horizon = 1
        
        future_dates = pd.date_range(start=last_input_date + pd.Timedelta(days=1), periods=future_horizon, freq='D')
        
        # Identifica a última linha de cada loja para usar como base
        last_rows = df.sort_values('DATA').groupby('CODIGO_LOJA').tail(1)
        
        future_dfs = []
        for _, row in last_rows.iterrows():
            temp_df = pd.DataFrame({'DATA': future_dates})
            for col in df.columns:
                if col not in ['DATA', 'TARGET_VENDAS']: 
                    temp_df[col] = row[col]
            temp_df['TARGET_VENDAS'] = np.nan
            future_dfs.append(temp_df)
            
        if not future_dfs: return df

        df_future = pd.concat(future_dfs)
        df_extended = pd.concat([df, df_future], ignore_index=True).sort_values(['CODIGO_LOJA', 'DATA'])
        
        return df_extended

    def _add_calendar_features(self, df):
        if 'DATA' not in df.columns:
            return df
            
        # CORREÇÃO DO ERRO 'DatetimeArray':
        dates_unique = df['DATA'].unique()
        ts_idx = pd.Index(dates_unique)
        if not isinstance(ts_idx, pd.DatetimeIndex):
            ts_idx = pd.to_datetime(ts_idx)
        ts_idx = ts_idx.sort_values()
        
        ts_day = datetime_attribute_timeseries(ts_idx, attribute="dayofweek", cyclic=True)
        ts_quarter = datetime_attribute_timeseries(ts_idx, attribute="quarter", one_hot=True)
        ts_week = datetime_attribute_timeseries(ts_idx, attribute="week", cyclic=True)
        
        ts_full = ts_day.stack(ts_quarter).stack(ts_week)
        df_cal = ts_full.pd_dataframe().reset_index().rename(columns={'time': 'DATA'})
        
        df['DATA'] = pd.to_datetime(df['DATA'])
        
        cal_cols = [c for c in df_cal.columns if c != 'DATA']
        df = df.drop(columns=[c for c in cal_cols if c in df.columns], errors='ignore')
        
        df_merged = pd.merge(df, df_cal, on='DATA', how='left')
        return df_merged

    def predict(self, context, model_input):
        # 1. Definição do Horizonte (n)
        n = 1
        if isinstance(model_input, pd.DataFrame) and 'n' in model_input.columns:
            try:
                n = int(model_input.iloc[0]['n'])
            except:
                pass
        
        predict_kwargs = {"n": n}

        try:
            if isinstance(model_input, pd.DataFrame) and len(model_input) > 1:
                
                model_input = model_input.loc[:, ~model_input.columns.duplicated()]
                
                # CHAMA A NOVA EXTENSÃO INTELIGENTE
                model_input = self._ensure_future_horizon(model_input, n)
                
                model_input = self._add_calendar_features(model_input)

                if hasattr(self, 'metadata') and self.metadata:
                    ordered_static = self.metadata.get("static_cols_order", [])
                    ordered_covariates = self.metadata.get("covariate_cols_order", [])
                    if "CODIGO_LOJA" in ordered_static:
                        ordered_static.remove("CODIGO_LOJA")
                else:
                    possible_static = ["CLUSTER_LOJA", "SIGLA_UF", "TIPO_LOJA", "MODELO_LOJA"]
                    ordered_static = [c for c in possible_static if c in model_input.columns]
                    ordered_static = [c for c in ordered_static if c in model_input.columns] # Segurança extra
                    
                    reserved = set(['DATA', 'CODIGO_LOJA', 'TARGET_VENDAS', 'n'] + ordered_static)
                    ordered_covariates = [c for c in model_input.columns if c not in reserved]

                for col in ordered_covariates:
                    if col not in model_input.columns:
                        model_input[col] = 0.0

                df_history = model_input.dropna(subset=['TARGET_VENDAS'])
                
                target_series_list = TimeSeries.from_group_dataframe(
                    df_history,
                    group_cols="CODIGO_LOJA",
                    time_col="DATA",
                    value_cols="TARGET_VENDAS",
                    static_cols=ordered_static,
                    freq='D',
                    fill_missing_dates=True,
                    fillna_value=0.0
                )

                store_ids_map = []
                for ts in target_series_list:
                    if "CODIGO_LOJA" in ts.static_covariates.columns:
                        raw_id = str(ts.static_covariates["CODIGO_LOJA"].iloc[0])
                    else:
                        raw_id = str(ts.static_covariates.index[0])
                    store_ids_map.append(raw_id)

                if ordered_covariates:
                    covariate_series_list = TimeSeries.from_group_dataframe(
                        model_input,
                        group_cols="CODIGO_LOJA",
                        time_col="DATA",
                        value_cols=ordered_covariates, 
                        freq='D',
                        fill_missing_dates=True,
                        fillna_value=0.0
                    )
                    
                    cov_dict = {}
                    for ts in covariate_series_list:
                        if "CODIGO_LOJA" in ts.static_covariates.columns:
                            c_id = str(ts.static_covariates["CODIGO_LOJA"].iloc[0])
                        else:
                            c_id = str(ts.static_covariates.index[0])
                        cov_dict[c_id] = ts
                else:
                    cov_dict = {}

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

            # --- PREDIÇÃO ---
            # IMPORTANTE: Esta parte deve estar indentada dentro do TRY
            pred_series_list = self.model.predict(**predict_kwargs)
            if not isinstance(pred_series_list, list): pred_series_list = [pred_series_list]
            
            final_df_list = []
            
            for pred_series, original_store_id in zip(pred_series_list, store_ids_map):
                pred_inverse = self.pipeline.inverse_transform(pred_series, partial=True)
                df = pred_inverse.pd_dataframe()
                df['CODIGO_LOJA'] = original_store_id
                
                # Renomeia dinamicamente
                col_val = [c for c in df.columns if c not in ['CODIGO_LOJA', 'DATA']][0]
                df.rename(columns={col_val: 'PREVISAO_VENDA'}, inplace=True)
                final_df_list.append(df)
                
            return pd.concat(final_df_list).reset_index().rename(columns={'DATA': 'DATA_PREVISAO'})

        except Exception as e:
            print(f"⚠️ [UnifiedForecaster] Erro Crítico: {str(e)}")
            dates = pd.date_range(start="2025-01-01", periods=n, freq="D")
            return pd.DataFrame({'dummy_error': np.zeros(n)}, index=dates)