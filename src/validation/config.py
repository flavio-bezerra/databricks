"""
Módulo de Configuração (Validation).

Este módulo centraliza todos os parâmetros e constantes utilizados no pipeline de validação e treinamento.
Sua principal função é permitir a parametrização dinâmica via Databricks Widgets, facilitando a execução
de experimentos com diferentes janelas de tempo, catálogos ou hiperparâmetros sem alterar o código.

Classes:
- Config: Classe Singleton-like que carrega e armazena as configurações.
"""

import os
import pytz
from typing import Optional, List
from datetime import datetime
from pyspark.sql import SparkSession

class Config:
    """
    Gerenciador de Configurações para o Pipeline de Validação.
    
    Esta classe tenta ler os parâmetros definidos na interface do Databricks (Widgets).
    Se não estiver rodando no Databricks (ex: teste local), define valores padrão seguros.
    """
    def __init__(self, spark_session: Optional[SparkSession] = None):
        self.spark_session: SparkSession = spark_session
        
        # Inicialização do dbutils para acesso aos Widgets
        try:
            from pyspark.dbutils import DBUtils
            # Garante que temos uma sessão Spark ativa
            if not self.spark_session:
                 self.spark_session = SparkSession.builder.getOrCreate()
                 
            dbutils = DBUtils(self.spark_session)
        except ImportError:
            # Caso esteja rodando fora do Databricks env
            dbutils = None

        try:
            # --- DEFINIÇÃO DOS WIDGETS ---
            # Widgets permitem input do usuário na UI do Notebook.
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
                pass # Ignora se widgets já estiverem criados

            # --- LEITURA DOS PARÂMETROS ---
            if dbutils:
                self.CATALOG: str = dbutils.widgets.get("catalog")
                self.DATA_START: str = dbutils.widgets.get("data_inicio_treino")
                
                # DATA DE CORTE PARA TREINO (TRAIN_END_DATE)
                # O modelo só "enxerga" dados até esta data. Tudo depois disso é futuro desconhecido para ele.
                self.TRAIN_END_DATE: str = dbutils.widgets.get("data_fim_treino")
                
                # DATA DE FIM DA VALIDAÇÃO (INGESTION_END)
                # Precisamos carregar dados até essa data para ter o "Gabarito" (Ground Truth) 
                # e calcular o erro das previsões.
                self.INGESTION_END: str = dbutils.widgets.get("data_fim_validacao")
                
                # O início da validação é logicamente o fim do treino.
                self.VAL_START_DATE: str = self.TRAIN_END_DATE
                
                # Hiperparâmetros do modelo
                self.FORECAST_HORIZON: int = int(dbutils.widgets.get("forecast_horizon"))
                self.N_EPOCHS: int = int(dbutils.widgets.get("n_epochs"))
                self.LAGS: int = int(dbutils.widgets.get("lags"))
        except (NameError, Exception):
            print('⚠️ Célula rodando fora do contexto de Widgets do Databricks/dbutils.')
            # Define valores padrão (Defaults) para desenvolvimento/teste local
            self.CATALOG = "ds_dev"
            self.FORECAST_HORIZON = 35
            self.N_EPOCHS = 20
            self.LAGS = 5

        self.SCHEMA: str = "cvc_val"
        
        # --- CONSTANTES DE DIRETÓRIO (Unity Catalog Volumes) ---
        # Caminhos onde artefatos temporários e modelos serão salvos.
        self.VOLUME_ROOT: str = f"/Volumes/{self.CATALOG}/{self.SCHEMA}/experiments/artefacts/loja"
        self.PATH_SCALERS: str = f"{self.VOLUME_ROOT}/scalar/validation"
        self.PATH_MODELS: str = f"{self.VOLUME_ROOT}/models/validation"
        
        # --- CONFIGURAÇÃO MLFLOW ---
        self.EXPERIMENT_NAME: str = "/Workspace/Shared/data_science/projetos/cvc_curva_de_vendas_por_canal/experiments/Model_Validation_CVC_Loja"
        self.MLFLOW_REGISTRY_URI: str = "databricks-uc" # Usa Unity Catalog como registro de modelos
        
        # Definição de lags para modelos de regressão (ex: [0, -1, -2] significa t, t-1, t-2)
        self.LAGS_FUTURE: List[int] = [0, -1, -2, -3]
        
        # Versionamento por timestamp
        self.VERSION: str = datetime.now(pytz.timezone('America/Sao_Paulo')).strftime('%Y_%m_%d_%H_%M')

        # Criação automática dos diretórios se não existirem
        try:
             for path in [self.PATH_SCALERS, self.PATH_MODELS]:
                 os.makedirs(path, exist_ok=True)
        except Exception:
             print("⚠️ Não foi possível criar diretórios locais (pode ser erro de permissão no Volume).")
