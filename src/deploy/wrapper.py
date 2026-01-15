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
        """
        model_input: DataFrame Pandas contendo 'DATA', 'CODIGO_LOJA', 'VALOR' (opcional para inferencia pura, mas necessário para o pipeline)
                     Deve conter o histórico recente (ex: últimos 60 dias) para que o modelo possa gerar lags.
        """
        from darts import TimeSeries # Importação local para evitar erros de carga se não necessário antes
        
        # 1. Definição do Horizonte (n)
        n = 1
        if isinstance(model_input, pd.DataFrame):
            if 'n' in model_input.columns:
                try:
                    n = int(model_input.iloc[0]['n'])
                except:
                    pass
        
        predict_kwargs = {"n": n}
        
        # Injeta covariáveis futuras conhecidas (Feriados, etc) se disponíveis no artefato
        if self.model.supports_future_covariates and self.future_covariates is not None:
             predict_kwargs["future_covariates"] = self.future_covariates

        try:
            # 2. Tratamento do Input (Contexto)
            # Se o dataframe tiver mais de uma linha, assumimos que é o contexto histórico para "warm start"
            if isinstance(model_input, pd.DataFrame) and len(model_input) > 1:
                # Reconstrói TimeSeries do Darts a partir do DataFrame bruto
                # Necessário garantir que as colunas existam. O usuário do notebook deve garantir isso.
                if 'DATA' in model_input.columns and 'CODIGO_LOJA' in model_input.columns and 'TARGET_VENDAS' in model_input.columns:
                    
                    # Identifica colunas estáticas presentes
                    possible_static = ["CLUSTER_LOJA", "SIGLA_UF", "TIPO_LOJA", "MODELO_LOJA"]
                    static_cols = [c for c in possible_static if c in model_input.columns]

                    # Converte para lista de TimeSeries
                    input_series_list = TimeSeries.from_group_dataframe(
                        model_input,
                        group_cols="CODIGO_LOJA",
                        time_col="DATA",
                        value_cols="TARGET_VENDAS",
                        static_cols=static_cols,
                        freq='D',
                        fill_missing_dates=True,
                        fillna_value=0.0
                    )
                    
                    # Aplica o Pipeline (Scaler) na série de entrada
                    # IMPORTANTE: O pipeline.transform espera uma lista de séries e (opcionalmente) covariáveis
                    # Aqui simplificamos passando apenas a série target para escalar, assumindo que as covariáveis globais
                    # já estão resolvidas ou não são transformadas dinamicamente neste ponto (limitação comum)
                    # Se o modelo precisasse de past_covariates transformadas, teríamos que passar aqui também.
                    
                    # Transformando apenas o Target (Contexto)
                    transformed_context_list = []
                    
                    # O ProjectPipeline customizado tem métodos fit/transform que aceitam (target, covs)
                    # Vamos usar o target_pipeline interno diretamente para transformar só o target
                    if hasattr(self.pipeline, 'target_pipeline'):
                         for ts in input_series_list:
                                transformed_context_list.append(self.pipeline.target_pipeline.transform(ts))
                    else:
                        # Fallback se a estrutura do pipeline for diferente
                         transformed_context_list = input_series_list
                    
                    predict_kwargs['series'] = transformed_context_list
                    
                else:
                    print("⚠️ [UnifiedForecaster] Input não possui colunas 'DATA', 'CODIGO_LOJA', 'TARGET_VENDAS'. Usando estado interno.")

            # 3. Predição
            pred_series_list = self.model.predict(**predict_kwargs)
            if not isinstance(pred_series_list, list):
                pred_series_list = [pred_series_list]
            
            # 4. Pós-processamento (Inverse Transform)
            final_df_list = []
            for pred_series in pred_series_list:
                pred_inverse = self.pipeline.inverse_transform(pred_series, partial=True)
                df = pred_inverse.pd_dataframe()
                
                # Resgata o ID da loja das covariáveis estáticas ou do nome da componente se disponível
                store_id = "UNKNOWN"
                if pred_inverse.static_covariates is not None:
                     store_id = str(pred_inverse.static_covariates.index[0])
                
                df['CODIGO_LOJA'] = store_id
                # Renomeia coluna de valor (geralmente vira o nome da coluna target ou '0')
                if 'TARGET_VENDAS' in df.columns:
                    df.rename(columns={'TARGET_VENDAS': 'PREVISAO_VENDA'}, inplace=True)
                else:
                    # Se o nome for diferente, pega a primeira coluna numérica
                    num_cols = df.select_dtypes(include=np.number).columns
                    if len(num_cols) > 0:
                         df.rename(columns={num_cols[0]: 'PREVISAO_VENDA'}, inplace=True)

                final_df_list.append(df)
                
            return pd.concat(final_df_list).reset_index().rename(columns={'DATA': 'DATA_PREVISAO'})

        except Exception as e:
            print(f"⚠️ [UnifiedForecaster] Falha crítica na predição: {str(e)}")
            import traceback
            traceback.print_exc()
            dates = pd.date_range(start="2025-01-01", periods=n, freq="D")
            return pd.DataFrame(np.zeros((n, 1)), index=dates, columns=["dummy_prediction"])
