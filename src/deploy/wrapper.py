import mlflow
import pickle
import pandas as pd
import numpy as np

class UnifiedForecaster(mlflow.pyfunc.PythonModel):
    """
    Wrapper "All-in-One" para deploy.
    Encapsula o pipeline de transformação (Darts) e o modelo treinado.
    Realiza o fluxo completo: Raw Data -> Normalize -> Predict -> Inverse Transform
    """
    def load_context(self, context):
        # Carrega o modelo treinado
        with open(context.artifacts["darts_model"], "rb") as f:
            self.model = pickle.load(f)
            
        # Carrega o pipeline de preprocessamento (Scaler, MissingFiller)
        with open(context.artifacts["pipeline"], "rb") as f:
            self.pipeline = pickle.load(f)
            
        # Carrega covariáveis futuras (se houver)
        self.future_covariates = None
        if "future_covariates" in context.artifacts:
            try:
                with open(context.artifacts["future_covariates"], "rb") as f:
                    self.future_covariates = pickle.load(f)
            except Exception as e:
                print(f"⚠️ [UnifiedForecaster] Erro ao carregar covariáveis: {e}")

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
                
                # --- DETECÇÃO DE COLUNAS ---
                required = ['DATA', 'CODIGO_LOJA', 'TARGET_VENDAS']
                if not all(c in model_input.columns for c in required):
                    raise ValueError(f"Input deve conter {required}")

                # Estáticas (Texto -> Encoding depois)
                possible_static = ["CLUSTER_LOJA", "SIGLA_UF", "TIPO_LOJA", "MODELO_LOJA"]
                static_cols = [c for c in possible_static if c in model_input.columns]

                # Covariáveis Dinâmicas (Tudo que não é Target, ID, Data, n ou Estática)
                # Ex: IS_FERIADO, PIB, DOLAR...
                reserved = set(required + static_cols + ['n'])
                covariate_cols = [c for c in model_input.columns if c not in reserved]

                # --- CONSTRUÇÃO DAS SÉRIES (TARGET) ---
                # Importante: O Target só deve ir até onde temos dados reais. 
                # Se o input tiver o futuro preenchido com 0/NaN, o modelo vai achar que a venda foi zero.
                # Vamos filtrar apenas onde o TARGET não é NaN para criar a série de contexto.
                
                df_history = model_input.dropna(subset=['TARGET_VENDAS'])
                # Opcional: Se você usa 0.0 para futuro em vez de NaN, precisaria de uma lógica de corte por data.
                # Assumindo aqui que você mandará NaN no futuro ou cortará antes no 'df_history'.
                
                target_series_list = TimeSeries.from_group_dataframe(
                    df_history, # Usa apenas histórico válido
                    group_cols="CODIGO_LOJA",
                    time_col="DATA",
                    value_cols="TARGET_VENDAS",
                    static_cols=static_cols,
                    freq='D',
                    fill_missing_dates=True,
                    fillna_value=0.0
                )

                # --- CONSTRUÇÃO DAS COVARIÁVEIS (FUTURO + PASSADO) ---
                # As covariáveis devem usar o dataframe COMPLETO (Histórico + n dias futuros)
                if covariate_cols:
                    covariate_series_list = TimeSeries.from_group_dataframe(
                        model_input, # Usa dataframe completo
                        group_cols="CODIGO_LOJA",
                        time_col="DATA",
                        value_cols=covariate_cols, # Pega IS_FERIADO, etc.
                        freq='D',
                        fill_missing_dates=True,
                        fillna_value=0.0
                    )
                    # Cria dicionário para alinhamento rápido
                    cov_dict = {str(ts.static_covariates.index[0]): ts for ts in covariate_series_list}
                else:
                    cov_dict = {}

                # --- APLICAÇÃO DO PIPELINE ---
                final_series_input = []
                final_covariates_input = []

                has_target_scaler = hasattr(self.pipeline, 'target_pipeline')
                has_static_encoder = hasattr(self.pipeline, 'static_pipeline')
                has_cov_scaler = hasattr(self.pipeline, 'covariate_pipeline')

                for ts_target in target_series_list:
                    store_id = str(ts_target.static_covariates.index[0])
                    
                    # 1. Transforma Target + Estáticas
                    ts_proc = ts_target
                    if has_target_scaler:
                        ts_proc = self.pipeline.target_pipeline.transform(ts_proc)
                    if has_static_encoder:
                        ts_proc = self.pipeline.static_pipeline.transform(ts_proc)
                    final_series_input.append(ts_proc)

                    # 2. Transforma Covariáveis (Se existirem para esta loja)
                    if store_id in cov_dict:
                        ts_cov = cov_dict[store_id]
                        if has_cov_scaler:
                            ts_cov = self.pipeline.covariate_pipeline.transform(ts_cov)
                        final_covariates_input.append(ts_cov)
                
                # Injeta no predict
                predict_kwargs['series'] = final_series_input
                if final_covariates_input:
                    predict_kwargs['future_covariates'] = final_covariates_input

            # 3. Predição
            pred_series_list = self.model.predict(**predict_kwargs)
            if not isinstance(pred_series_list, list):
                pred_series_list = [pred_series_list]
            
            # 4. Inverse Transform (Target)
            final_df_list = []
            for pred_series in pred_series_list:
                pred_inverse = self.pipeline.inverse_transform(pred_series, partial=True)
                df = pred_inverse.pd_dataframe()
                
                # Resgata ID
                store_id = "UNKNOWN"
                if pred_inverse.static_covariates is not None:
                     store_id = str(pred_inverse.static_covariates.index[0])
                df['CODIGO_LOJA'] = store_id
                
                # Renomeia
                col_val = [c for c in df.columns if c not in ['CODIGO_LOJA', 'DATA']][0]
                df.rename(columns={col_val: 'PREVISAO_VENDA'}, inplace=True)
                final_df_list.append(df)
                
            return pd.concat(final_df_list).reset_index().rename(columns={'DATA': 'DATA_PREVISAO'})

        except Exception as e:
            print(f"⚠️ [UnifiedForecaster] Erro: {str(e)}")
            # Retorna dummy para não quebrar pipeline em lote
            dates = pd.date_range(start="2025-01-01", periods=n, freq="D")
            return pd.DataFrame({'dummy': np.zeros(n)}, index=dates)
