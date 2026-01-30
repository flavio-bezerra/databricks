"""
Módulo de Definição de Modelos (Validation).

Este módulo define wrappers personalizados para integrar modelos do Darts com o ecossistema MLflow.
Como o Darts opera com objetos `TimeSeries` (não nativos do MLflow/Pandas), precisamos de uma camada
de adaptação para salvar, carregar e servir esses modelos.

Classes:
- DartsWrapper: Classe customizada (mlflow.pyfunc.PythonModel) que encapsula o modelo Darts.
"""

import mlflow
import pickle
import pandas as pd
import numpy as np
from typing import Any, Optional

class DartsWrapper(mlflow.pyfunc.PythonModel):
    """
    Wrapper que permite salvar modelos de previsão Darts no registro do MLflow.
    
    Esta classe resolve dois problemas principais:
    1. ARTEFATOS: Carrega o modelo pickle e quaisquer covariáveis futuras necessárias (ex: feriados futuros) no `load_context`.
    2. INTERFACE: Implementa um método `predict` que aceita DataFrames (padrão MLflow), converte inputs se necessário,
       e retorna DataFrames, abstraindo a complexidade de TimeSeries do usuário final.
    """
    def load_context(self, context: Any) -> None:
        """
        Carrega os artefatos do modelo quando ele é baixado do registro (MLflow).
        É executado automaticamente ao iniciar o modelo para inferência.
        """
        # Carrega o objeto modelo de treino (pickle)
        with open(context.artifacts["darts_model"], "rb") as f:
            self.model = pickle.load(f)
        
        # Tenta carregar covariáveis futuras (se o modelo usar, ex: Prophet, LinearRegression)
        self.future_covariates: Optional[Any] = None
        if "future_covariates" in context.artifacts:
            try:
                with open(context.artifacts["future_covariates"], "rb") as f:
                    self.future_covariates = pickle.load(f)
            except Exception as e:
                print(f"⚠️ [Wrapper] Erro ao carregar covariáveis: {e}")

    def predict(self, context: Any, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Executa a inferência.
        
        Args:
            context: Contexto MLflow (não usado explicitamente aqui, mas exigido pela assinatura).
            model_input (pd.DataFrame): DataFrame de entrada. Pode conter uma coluna 'n' indicando o horizonte.
            
        Returns:
            pd.DataFrame: DataFrame com as predições.
        """
        # Determina o horizonte de previsão 'n' (padrão = 1 dia, ou lido do input)
        n = 1
        if isinstance(model_input, pd.DataFrame):
            if 'n' in model_input.columns:
                try:
                    n = int(model_input.iloc[0]['n'])
                except Exception:
                    pass
        
        predict_kwargs = {"n": n}
        
        # Se o modelo precisa de features futuras (feriados, etc) e as temos carregadas, passamos aqui.
        if self.model.supports_future_covariates and self.future_covariates is not None:
             predict_kwargs["future_covariates"] = self.future_covariates

        try:
            # Chama o método original .predict() do objeto Darts
            pred = self.model.predict(**predict_kwargs)
            
            # Converte a saída (TimeSeries ou List[TimeSeries]) de volta para DataFrame Pandas
            if isinstance(pred, list):
                 return pd.concat([p.pd_dataframe() for p in pred])
            return pred.pd_dataframe()
            
        except Exception as e:
            # Fallback robusto: Se o modelo falhar, retorna zeros para não quebrar pipelines em batch.
            print(f"⚠️ [Wrapper] Falha na predição: {str(e)}. Retornando Dummy.")
            dates = pd.date_range(start="2025-01-01", periods=n, freq="D")
            return pd.DataFrame(np.zeros((n, 1)), index=dates, columns=["dummy_prediction"])
