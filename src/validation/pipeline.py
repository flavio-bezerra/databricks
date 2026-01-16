from darts.dataprocessing.pipeline import Pipeline
from darts.dataprocessing.transformers import (
    Scaler,
    StaticCovariatesTransformer,
    MissingValuesFiller
)
from sklearn.preprocessing import OrdinalEncoder

class ProjectPipeline:
    def __init__(self):
        self.target_pipeline = Pipeline([
            MissingValuesFiller(verbose=False),
            Scaler(name="target_scaler")
        ])
        
        # --- ALTERAÇÃO DE SEGURANÇA ---
        # Usa OrdinalEncoder para lidar com categorias desconhecidas (lojas novas/erros)
        # unknown_value=-1 garante que o modelo rode mesmo com dados sujos
        self.static_pipeline = Pipeline([
            StaticCovariatesTransformer(
                verbose=False,
                transformer_cat=OrdinalEncoder(
                    handle_unknown='use_encoded_value', 
                    unknown_value=-1
                )
            )
        ])
        # -----------------------------

        self.covariate_pipeline = Pipeline([
            MissingValuesFiller(verbose=False),
            Scaler(name="covar_scaler")
        ])

    def fit(self, target_series, covariates):
        self.target_pipeline.fit(target_series)
        self.static_pipeline.fit(target_series) 
        self.covariate_pipeline.fit(covariates)
        return self

    def transform(self, target_series, covariates):
        ts_scaled = self.target_pipeline.transform(target_series)
        ts_scaled = self.static_pipeline.transform(ts_scaled)
        cov_scaled = self.covariate_pipeline.transform(covariates)
        return ts_scaled, cov_scaled

    def inverse_transform(self, target_series, partial=False):
        return self.target_pipeline.inverse_transform(target_series, partial=partial)