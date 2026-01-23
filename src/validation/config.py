import os
import pytz
from typing import Optional, List
from datetime import datetime
from pyspark.sql import SparkSession

class Config:
    """
    Configuração centralizada para validação e treinamento.
    Gerencia parâmetros via Databricks Widgets ou defaults.
    """
    def __init__(self, spark_session: Optional[SparkSession] = None):
        self.spark_session: SparkSession = spark_session
        
        # Tenta pegar widgets do Databricks
        try:
            from pyspark.dbutils import DBUtils
            # Se a sessão spark não for passada, tenta obter globalmente
            if not self.spark_session:
                 self.spark_session = SparkSession.builder.getOrCreate()
                 
            dbutils = DBUtils(self.spark_session)
        except ImportError:
            dbutils = None

        try:
            # Setup Widgets se não existirem
            try:
                if dbutils:
                    dbutils.widgets.text("data_inicio_treino", "2019-01-01", "1. Início Treino")
                    dbutils.widgets.text("data_fim_treino", "2025-01-01", "2. Fim Treino (Corte)")
                    dbutils.widgets.text("data_fim_validacao", "2025-12-31", "3. Fim Validação (Ground Truth)")
                    dbutils.widgets.text("catalog", "ds_dev", "4. Catálogo")
                    dbutils.widgets.text("forecast_horizon", "35", "5. Horizonte (Dias)")
                    dbutils.widgets.text("n_epochs", "20", "6. Épocas (DL Models)")
                    dbutils.widgets.text("lags", "5", "7. Lags")
            except Exception:
                pass # Widgets já criados ou erro ao criar

            if dbutils:
                self.CATALOG: str = dbutils.widgets.get("catalog")
                self.DATA_START: str = dbutils.widgets.get("data_inicio_treino")
                # TRAIN_END_DATE: Data limite para o modelo APRENDER (Fit).
                self.TRAIN_END_DATE: str = dbutils.widgets.get("data_fim_treino")
                # INGESTION_END: Data limite para carregar dados (precisa incluir 2025 para validar)
                self.INGESTION_END: str = dbutils.widgets.get("data_fim_validacao")
                self.VAL_START_DATE: str = self.TRAIN_END_DATE                               # Validação começa onde treino termina
                self.FORECAST_HORIZON: int = int(dbutils.widgets.get("forecast_horizon"))
                self.N_EPOCHS: int = int(dbutils.widgets.get("n_epochs"))
                self.LAGS: int = int(dbutils.widgets.get("lags"))
        except (NameError, Exception):
            print('⚠️ Célula rodando fora do contexto de Widgets do Databricks/dbutils.')
            # Define defaults para evitar crash
            self.CATALOG = "ds_dev"
            self.FORECAST_HORIZON = 35
            self.N_EPOCHS = 20
            self.LAGS = 5

        self.SCHEMA: str = "cvc_val"
        # Unity Catalog Volumes Path
        self.VOLUME_ROOT: str = f"/Volumes/{self.CATALOG}/{self.SCHEMA}/experiments/artefacts/loja"
        self.PATH_SCALERS: str = f"{self.VOLUME_ROOT}/scalar/validation"
        self.PATH_MODELS: str = f"{self.VOLUME_ROOT}/models/validation"
        
        # DEFINIÇÃO DO EXPERIMENTO MLFLOW
        self.EXPERIMENT_NAME: str = "/Workspace/Shared/data_science/projetos/cvc_curva_de_vendas_por_canal/experiments/Model_Validation_CVC_Loja"
        self.MLFLOW_REGISTRY_URI: str = "databricks-uc"
        self.LAGS_FUTURE: List[int] = [0, -1, -2, -3]
        self.VERSION: str = datetime.now(pytz.timezone('America/Sao_Paulo')).strftime('%Y_%m_%d_%H_%M')

        # Garante estrutura de diretórios
        try:
             for path in [self.PATH_SCALERS, self.PATH_MODELS]:
                 os.makedirs(path, exist_ok=True)
        except Exception:
             print("⚠️ Não foi possível criar diretórios locais (pode ser erro de permissão no Volume).")
