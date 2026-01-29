from typing import List, Tuple
from darts import TimeSeries
from darts.dataprocessing.pipeline import Pipeline
from darts.dataprocessing.transformers import (
    Scaler,
    StaticCovariatesTransformer,
    MissingValuesFiller
)
from sklearn.preprocessing import OrdinalEncoder

class ProjectPipeline:
    """
    Pipeline unificado de pré-processamento para Targets e Covariáveis do Darts.
    """
    def __init__(self):
        self.target_pipeline = Pipeline([
            MissingValuesFiller(verbose=False),
            # global_fit=True permite que o scaler aprenda uma escala única para todas as lojas
            # e aceite um número diferente de séries no transform futuramente.
            Scaler(name="target_scaler", global_fit=True) 
        ])
        
        self.static_pipeline = Pipeline([
            StaticCovariatesTransformer(
                verbose=False,
                transformer_cat=OrdinalEncoder(
                    handle_unknown='use_encoded_value', 
                    unknown_value=-1
                )
            )
        ])

        self.covariate_pipeline = Pipeline([
            MissingValuesFiller(verbose=False),
            Scaler(name="covar_scaler", global_fit=True) # Também global para as covariáveis
        ])

    def fit(self, target_series: TimeSeries, covariates: TimeSeries) -> "ProjectPipeline":
        """
        Ajusta os escaladores nos dados de treino.

        Args:
            target_series: Série temporal alvo.
            covariates: Covariáveis.

        Returns:
            self
        """
        self.target_pipeline.fit(target_series)
        self.static_pipeline.fit(target_series) 
        self.covariate_pipeline.fit(covariates)
        return self

    def transform(self, target_series: List[TimeSeries], covariates: List[TimeSeries]) -> Tuple[List[TimeSeries], List[TimeSeries]]:
        """
        Aplica as transformações e garante que o nome do índice 'codigo_loja' seja preservado.
        """
        ts_scaled = self.target_pipeline.transform(target_series)
        ts_scaled = self.static_pipeline.transform(ts_scaled)
        
        # --- Garantia de Identidade: Restaura o nome do índice se resetado pelo Darts ---
        fixed_ts_scaled = []
        for ts in ts_scaled:
            if ts.static_covariates is not None and ts.static_covariates.index.name != "codigo_loja":
                new_static = ts.static_covariates.copy()
                new_static.index.name = "codigo_loja"
                ts = ts.with_static_covariates(new_static)
            fixed_ts_scaled.append(ts)
            
        cov_scaled = self.covariate_pipeline.transform(covariates)
        return fixed_ts_scaled, cov_scaled

    def inverse_transform(self, target_series: TimeSeries, partial: bool = False) -> TimeSeries:
        """
        Reverte a transformação do target (útil para predições).

        Args:
            target_series: Série temporal escalada.
            partial (bool): Se True, inverte parcialmente.

        Returns:
            TimeSeries: Série temporal na escala original.
        """
        return self.target_pipeline.inverse_transform(target_series, partial=partial)
