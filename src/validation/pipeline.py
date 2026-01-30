"""
Módulo de Pipeline de Transformação (Validation).

Define pipelines reutilizáveis para pré-processamento de séries temporais.
Para redes neurais e muitos modelos de ML, dados normalizados (escala 0-1 ou Z-score) são essenciais
para a convergência do treinamento. Este módulo encapsula essas regras.

Classes:
- ProjectPipeline: Classe que agrupa scalers para target, features estáticas e covariáveis.
"""

from typing import Tuple
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
    Pipeline unificado de pré-processamento.
    Mantém o estado (fit) dos scalers para garantir que a transformação inversa (inverse_transform)
    possa trazer as predições de volta para a escala real (R$).
    """
    def __init__(self):
        # Pipeline para a variável alvo (Vendas)
        self.target_pipeline = Pipeline([
            MissingValuesFiller(verbose=False), # Garante sem buracos
            Scaler(name="target_scaler")        # Normaliza (MinMax por padrão)
        ])
        
        # Pipeline para Covariáveis Estáticas (Características da Loja: UF, Cluster)
        # --- ESTRATÉGIA DE CODIFICAÇÃO ---
        # Covariáveis estáticas são strings categóricas. Precisamos convertê-las para números.
        # Usamos OrdinalEncoder pois muitos modelos de Deep Learning aceitam índices inteiros para Embeddings.
        # Configuração 'handle_unknown' é vital: se aparecer uma loja com "UF Desconhecida", virá -1 e não quebrará.
        self.static_pipeline = Pipeline([
            StaticCovariatesTransformer(
                verbose=False,
                transformer_cat=OrdinalEncoder(
                    handle_unknown='use_encoded_value', 
                    unknown_value=-1
                )
            )
        ])

        # Pipeline para Covariáveis Dinâmicas (Feriados, Indicadores, Calendário)
        self.covariate_pipeline = Pipeline([
            MissingValuesFiller(verbose=False),
            Scaler(name="covar_scaler") # Normaliza para mesma escala do target
        ])

    def fit(self, target_series: TimeSeries, covariates: TimeSeries) -> "ProjectPipeline":
        """
        Calcula as estatísticas (Mínimo, Máximo, Média) necessárias para o escalonamento,
        usando os dados de treino.
        
        Args:
            target_series: Série de vendas de treino.
            covariates: Features de treino.
            
        Returns:
            self: Retorna o próprio objeto treinado.
        """
        self.target_pipeline.fit(target_series)
        self.static_pipeline.fit(target_series) # Fit nas estáticas (associa categorias a números)
        self.covariate_pipeline.fit(covariates)
        return self

    def transform(self, target_series: TimeSeries, covariates: TimeSeries) -> Tuple[TimeSeries, TimeSeries]:
        """
        Aplica as transformações aprendidas no 'fit' aos dados.
        Tranforma dados reais -> dados normalizados (0-1).
        
        Args:
            target_series: Série temporal (treino ou teste).
            covariates: Covariáveis.

        Returns:
            Tuple: Séries transformadas prontas para o modelo.
        """
        ts_scaled = self.target_pipeline.transform(target_series)
        ts_scaled = self.static_pipeline.transform(ts_scaled)
        cov_scaled = self.covariate_pipeline.transform(covariates)
        return ts_scaled, cov_scaled

    def inverse_transform(self, target_series: TimeSeries, partial: bool = False) -> TimeSeries:
        """
        Reverte a normalização. Transforma predição normalizada (0.5) -> valor real (R$ 5000).
        Essencial para obter o resultado final e calcular métricas de negócio.
        
        Args:
            target_series: Série temporal escalada (que saiu do output do modelo).
            partial (bool): Se True, permite inverter apenas um subconjunto das séries (otimização).

        Returns:
            TimeSeries: Série na escala original.
        """
        return self.target_pipeline.inverse_transform(target_series, partial=partial)
