import mlflow
import pandas as pd
import numpy as np
from darts import TimeSeries
from typing import List, Optional

class UnifiedForecaster(mlflow.pyfunc.PythonModel):
    """
    Wrapper unificado para inferência usando Darts com suporte a 
    múltiplas séries, covariáveis e saneamento atômico de dados.
    """
    
    def load_context(self, context):
        """Carrega o modelo e o pipeline salvos como artefatos."""
        import joblib
        self.model = joblib.load(context.artifacts["model"])
        self.pipeline = joblib.load(context.artifacts["pipeline"])

    def _sanitize_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica saneamento atômico para garantir séries 1D e tipos corretos."""
        clean_dict = {}
        for col in df.columns.unique():
            col_name = str(col).strip()
            series_data = df[col]
            # Se houver colunas duplicadas, seleciona a primeira
            if isinstance(series_data, pd.DataFrame):
                series_data = series_data.iloc[:, 0]
            clean_dict[col_name] = series_data.values.flatten()
        
        df_clean = pd.DataFrame(clean_dict)
        # Proteção para garantir que o código_loja seja string limpa
        if 'codigo_loja' in df_clean.columns:
            df_clean['codigo_loja'] = df_clean['codigo_loja'].astype(str).str.replace(r'\.0$', '', regex=True)
        if 'data' in df_clean.columns:
            df_clean['data'] = pd.to_datetime(df_clean['data'])
            
        return df_clean

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Executa a predição para todas as lojas presentes no model_input.
        """
        # 1. Saneamento inicial
        df = self._sanitize_input(model_input)
        
        # 2. Identificação de Features
        # Assume-se que colunas que não são metadados ou target são covariáveis
        exclude_cols = ['data', 'codigo_loja', 'target_vendas']
        static_cols = ["cluster_loja", "sigla_uf", "tipo_loja", "modelo_loja"]
        covariate_cols = [c for c in df.columns if c not in exclude_cols + static_cols]
        
        # 3. Construção dos Objetos Darts por Loja
        target_series_list = []
        covariates_list = []
        store_ids = []

        for store_id, group_df in df.groupby("codigo_loja"):
            group_df = group_df.sort_values("data")
            
            # Separar histórico (target) e futuro (covariates)
            df_history = group_df.dropna(subset=['target_vendas'])
            
            if df_history.empty:
                continue

            # Criar série Target com identidade no componente
            ts_target = TimeSeries.from_dataframe(
                df_history.rename(columns={"target_vendas": store_id}),
                time_col="data",
                value_cols=[store_id],
                freq='D',
                fill_missing_dates=True,
                fillna_value=0.0
            )

            # Adicionar Covariáveis Estáticas (com nome de índice correto para o Scaler)
            available_static = [c for c in static_cols if c in group_df.columns]
            static_df = group_df[available_static].iloc[0:1].copy()
            static_df.index = pd.Index([store_id], name="codigo_loja")
            ts_target = ts_target.with_static_covariates(static_df)

            # Criar série de Covariáveis Futuras
            ts_cov = TimeSeries.from_dataframe(
                group_df,
                time_col="data",
                value_cols=covariate_cols,
                freq='D',
                fill_missing_dates=True,
                fillna_value=0.0
            )

            target_series_list.append(ts_target)
            covariates_list.append(ts_cov)
            store_ids.append(store_id)

        if not target_series_list:
            return pd.DataFrame()

        # 4. Transformação via Pipeline (Scaling)
        # O pipeline já deve estar com global_fit=True para suportar novas lojas
        ts_target_scaled, ts_cov_scaled = self.pipeline.transform(target_series_list, covariates_list)

        # 5. Predição
        # O horizonte n é determinado pelo tamanho das covariáveis enviadas além do histórico
        n_forecast = len(covariates_list[0]) - len(target_series_list[0])
        
        preds_scaled = self.model.predict(
            n=n_forecast,
            series=ts_target_scaled,
            future_covariates=ts_cov_scaled
        )

        # 6. Inverse Transform e Formatação Final
        preds_original = self.pipeline.inverse_transform(preds_scaled, partial=True)
        
        all_results = []
        for i, store_id in enumerate(store_ids):
            df_pred = preds_original[i].pd_dataframe().reset_index()
            df_pred.columns = ['data', 'previsao_vendas']
            df_pred['codigo_loja'] = store_id
            all_results.append(df_pred)

        return pd.concat(all_results, ignore_index=True)