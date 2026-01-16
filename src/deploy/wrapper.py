import mlflow
import pickle
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.utils.timeseries_generation import datetime_attribute_timeseries

class UnifiedForecaster(mlflow.pyfunc.PythonModel):
    """
    Wrapper "All-in-One" para deploy.
    Gera automaticamente features de calendário e blinda a ordem das colunas.
    """
    def load_context(self, context):
        import os
        with open(context.artifacts["darts_model"], "rb") as f:
            self.model = pickle.load(f)
        with open(context.artifacts["pipeline"], "rb") as f:
            self.pipeline = pickle.load(f)
        
        if "metadata" in context.artifacts:
            with open(context.artifacts["metadata"], "rb") as f:
                self.metadata = pickle.load(f)
        else:
            self.metadata = {}

    def _add_calendar_features(self, df):
        """
        Recria as features de calendário (Seno/Cosseno/Dummies) usadas no treino.
        Isso evita que o usuário tenha que calculá-las manualmente.
        """
        if 'DATA' not in df.columns:
            return df
            
        # 1. Cria índice de datas únicas presentes no input
        dates = pd.to_datetime(df['DATA'].unique())
        dates = dates.sort_values()
        
        # 2. Gera as TimeSeries do Darts (Exatamente como no data.py)
        ts_idx = pd.DatetimeIndex(dates)
        
        # Day of Week (Cyclic) -> dayofweek_sin, dayofweek_cos
        ts_day = datetime_attribute_timeseries(ts_idx, attribute="dayofweek", cyclic=True)
        
        # Quarter (One Hot) -> quarter_0, quarter_1, quarter_2, quarter_3
        # Nota: OneHot pode gerar menos colunas se o input for curto (ex: só Jan). 
        # Vamos tratar isso depois preenchendo as faltantes.
        ts_quarter = datetime_attribute_timeseries(ts_idx, attribute="quarter", one_hot=True)
        
        # Week (Cyclic) -> week_sin, week_cos
        ts_week = datetime_attribute_timeseries(ts_idx, attribute="week", cyclic=True)
        
        # 3. Stack e Converte para DataFrame
        ts_full = ts_day.stack(ts_quarter).stack(ts_week)
        df_cal = ts_full.pd_dataframe().reset_index().rename(columns={'time': 'DATA'})
        
        # Converte DATA para string/object para merge seguro se o input for string
        df['DATA'] = pd.to_datetime(df['DATA'])
        
        # 4. Merge com o DataFrame original
        df_merged = pd.merge(df, df_cal, on='DATA', how='left')
        return df_merged

    def predict(self, context, model_input):
        from darts import TimeSeries
        import pandas as pd
        import numpy as np

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
                
                # --- LIMPEZA DE DUPLICATAS ---
                model_input = model_input.loc[:, ~model_input.columns.duplicated()]

                # --- PASSO 0: ENRIQUECIMENTO AUTOMÁTICO (CALENDÁRIO) ---
                model_input = self._add_calendar_features(model_input)

                # --- RECUPERAÇÃO DA ORDEM ---
                if hasattr(self, 'metadata') and self.metadata:
                    ordered_static = self.metadata.get("static_cols_order", [])
                    ordered_covariates = self.metadata.get("covariate_cols_order", [])
                    
                    # --- CORREÇÃO DO ERRO 'Grouper not 1-dimensional' ---
                    # O Darts adiciona o grupo nas estáticas automaticamente.
                    # Se passarmos ele em 'static_cols', cria conflito. Removemos aqui.
                    if "CODIGO_LOJA" in ordered_static:
                        ordered_static.remove("CODIGO_LOJA")
                        
                else:
                    # Fallback arriscado
                    possible_static = ["CLUSTER_LOJA", "SIGLA_UF", "TIPO_LOJA", "MODELO_LOJA"]
                    ordered_static = [c for c in possible_static if c in model_input.columns]
                    reserved = set(['DATA', 'CODIGO_LOJA', 'TARGET_VENDAS', 'n'] + ordered_static)
                    ordered_covariates = [c for c in model_input.columns if c not in reserved]

                # --- PREENCHIMENTO DE COLUNAS FALTANTES ---
                for col in ordered_covariates:
                    if col not in model_input.columns:
                        model_input[col] = 0.0

                # Validação Final
                missing_static = [c for c in ordered_static if c not in model_input.columns]
                # Nota: missing_cov pode ter falsos positivos se ordered_covariates tiver lixo, mas ok alertar
                
                if missing_static:
                     raise ValueError(f"Input incompleto! Faltando Estáticas: {missing_static}")

                # --- CONSTRUÇÃO DAS SÉRIES ---
                df_history = model_input.dropna(subset=['TARGET_VENDAS'])
                
                target_series_list = TimeSeries.from_group_dataframe(
                    df_history,
                    group_cols="CODIGO_LOJA",
                    time_col="DATA",
                    value_cols="TARGET_VENDAS",
                    static_cols=ordered_static, # AGORA ESTÁ LIMPO (Sem CODIGO_LOJA)
                    freq='D',
                    fill_missing_dates=True,
                    fillna_value=0.0
                )

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
                    cov_dict = {str(ts.static_covariates.index[0]): ts for ts in covariate_series_list}
                else:
                    cov_dict = {}

                # --- PIPELINE ---
                final_series_input = []
                final_covariates_input = []
                
                has_target_scaler = hasattr(self.pipeline, 'target_pipeline')
                has_static_encoder = hasattr(self.pipeline, 'static_pipeline')
                has_cov_scaler = hasattr(self.pipeline, 'covariate_pipeline')

                for ts_target in target_series_list:
                    store_id = str(ts_target.static_covariates.index[0])
                    
                    # Target
                    ts_proc = ts_target
                    if has_target_scaler: ts_proc = self.pipeline.target_pipeline.transform(ts_proc)
                    if has_static_encoder: ts_proc = self.pipeline.static_pipeline.transform(ts_proc)
                    final_series_input.append(ts_proc)

                    # Covariates
                    if store_id in cov_dict:
                        ts_cov = cov_dict[store_id]
                        if has_cov_scaler: ts_cov = self.pipeline.covariate_pipeline.transform(ts_cov)
                        final_covariates_input.append(ts_cov)
                
                predict_kwargs['series'] = final_series_input
                if final_covariates_input:
                    predict_kwargs['future_covariates'] = final_covariates_input

            # --- PREDIÇÃO ---
            pred_series_list = self.model.predict(**predict_kwargs)
            if not isinstance(pred_series_list, list): pred_series_list = [pred_series_list]
            
            final_df_list = []
            for pred_series in pred_series_list:
                pred_inverse = self.pipeline.inverse_transform(pred_series, partial=True)
                df = pred_inverse.pd_dataframe()
                
                store_id = "UNKNOWN"
                if pred_inverse.static_covariates is not None:
                     store_id = str(pred_inverse.static_covariates.index[0])
                df['CODIGO_LOJA'] = store_id
                
                col_val = [c for c in df.columns if c not in ['CODIGO_LOJA', 'DATA']][0]
                df.rename(columns={col_val: 'PREVISAO_VENDA'}, inplace=True)
                final_df_list.append(df)
                
            return pd.concat(final_df_list).reset_index().rename(columns={'DATA': 'DATA_PREVISAO'})

        except Exception as e:
            print(f"⚠️ [UnifiedForecaster] Erro Crítico: {str(e)}")
            dates = pd.date_range(start="2025-01-01", periods=n, freq="D")
            return pd.DataFrame({'dummy': np.zeros(n)}, index=dates)