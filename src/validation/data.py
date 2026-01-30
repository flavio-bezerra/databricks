"""
Módulo de Preparação de Dados (Validation).

Este módulo é responsável por buscar os dados brutos no Feature Store e transformá-los
no formato exigido pela biblioteca Darts (TimeSeries). Ele lida com a complexidade de:
1. Unir tabelas de features (Join Point-in-time).
2. Materializar dados do Spark para Pandas (necessário para o Darts).
3. Criar objetos TimeSeries para Targets (Vendas) e Covariáveis (Feriados, Indicadores).
4. Alinhar temporalmente séries globais e locais.

Classes:
- DataIngestion: Orquestra todo o fluxo de dados.
"""

from typing import List, Tuple, Optional, Any
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
import pyspark.sql.functions as F
from pyspark.sql import SparkSession, DataFrame
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.utils.timeseries_generation import datetime_attribute_timeseries

class DataIngestion:
    """
    Controlador de Ingestão e Transformação de Dados.
    """
    def __init__(self, spark_session: SparkSession, config: Any):
        self.spark: SparkSession = spark_session
        self.config = config
        self.fe = FeatureEngineeringClient()

    def create_training_set(self) -> DataFrame:
        """
        Constrói o Dataset de Treinamento unindo a tabela alvo com suas features.
        
        Utiliza o `FeatureEngineeringClient` para realizar um "Point-in-Time Join" correto,
        garantindo que para cada venda em data T, apenas as features conhecidas em T (ou antes)
        sejam associadas, evitando vazamento de dados do futuro (Data Leakage).
        
        Returns:
            DataFrame: DataFrame Spark contendo todas as colunas de dados e features unificadas.
        """
        print("🛒 Construindo Training Set via Feature Store (Spark Native)...")
        target_table = f"{self.config.CATALOG}.{self.config.SCHEMA}.historico_targuet_loja"
        
        # Define a "Espinha Dorsal" (Spine) do dataset: Quem (Loja) e Quando (Data) queremos prever.
        df_spine = (self.spark.table(target_table)
                    .filter(F.col("data").between(self.config.DATA_START, self.config.INGESTION_END))
                    .select("codigo_loja", "data", "valor")
                    .withColumnRenamed("valor", "target_vendas")
                    .withColumn("codigo_loja", F.col("codigo_loja").cast("string"))
                   )

        # Configura os lookups para buscar features adicionais baseadas na chave (codigo_loja)
        # Feature 1: Características estáticas da loja (Cluster, UF, Tipo)
        # Feature 2: Dados históricos de feriados (Time-series)
        feature_lookups = [
            FeatureLookup(
                table_name=f"{self.config.CATALOG}.{self.config.SCHEMA}.lojas_fs",
                lookup_key=["codigo_loja"],
                feature_names=["cluster_loja", "sigla_uf", "tipo_loja", "modelo_loja"]
            ),
            FeatureLookup(
                table_name=f"{self.config.CATALOG}.{self.config.SCHEMA}.historico_feriados_loja",
                lookup_key=["codigo_loja"],
                timestamp_lookup_key="data", # Importante: Join considerando o tempo
                feature_names=["valor"], 
                rename_outputs={"valor": "is_feriado"}
            )
        ]

        # Executa o join inteligente do Feature Store
        training_set = self.fe.create_training_set(
            df=df_spine,
            feature_lookups=feature_lookups,
            label="target_vendas",
            exclude_columns=[]
        )

        df_spark = training_set.load_df()

        print("   ⚡ Executando limpeza e tratamento no Spark Cluster...")
        # Preenche valores nulos que podem ter surgido do Join (ex: loja sem feriado na data)
        df_spark = df_spark.na.fill({
            "is_feriado": 0.0, 
            "target_vendas": 0.0,
            "cluster_loja": "DESCONHECIDO",
            "sigla_uf": "DESCONHECIDO",
            "tipo_loja": "DESCONHECIDO",
            "modelo_loja": "DESCONHECIDO"
        })

        # Garante tipagem correta da data
        df_spark = df_spark.withColumn("data", F.to_timestamp("data"))
        return df_spark

    def get_global_support(self) -> pd.DataFrame:
        """
        Carrega séries temporais globais (não específicas de loja), como indicadores econômicos agregados.
        Estas séries ajudam o modelo a entender tendências macro.
        
        Returns:
            pd.DataFrame: DataFrame Pandas indexado por dia, preenchido para dias faltantes.
        """
        table_name = "historico_suporte_loja"
        print(f"🌍 Carregando suporte global (Spark Aggregation)...")
        # Agrega metricas globais por dia
        df_spark = (self.spark.table(f"{self.config.CATALOG}.{self.config.SCHEMA}.{table_name}")
            .filter(F.col("DATA").between(self.config.DATA_START, self.config.INGESTION_END))
            .groupBy("data")
            .pivot("metricas")
            .agg(F.sum("valor"))
            .na.fill(0.0)
        )
        pdf = df_spark.toPandas()
        pdf['data'] = pd.to_datetime(pdf['data'])
        # Garante frequência diária, preenchendo buracos com 0
        return pdf.set_index('data').asfreq('D').fillna(0.0)

    def build_darts_objects(
        self, 
        df_spark_wide: DataFrame, 
        df_global_support: pd.DataFrame, 
        df_market_indicators: Optional[pd.DataFrame] = None
    ) -> Tuple[List[TimeSeries], List[TimeSeries]]:
        """
        Converte os dados tabulares (Spark DataFrame) para a estrutura de objetos do Darts.
        
        O Darts exige objetos `TimeSeries`. Para modelos globais (treinar 1 modelo para N lojas),
        precisamos de uma lista de TimeSeries, uma para cada loja.
        
        Args:
            df_spark_wide: DataFrame principal com Vendas e Atributos da loja.
            df_global_support: Dados agregados globais.
            df_market_indicators: (Opcional) Outros indicadores de mercado externos.
            
        Returns:
            Tuple: (Lista de Séries Alvo [Vendas], Lista de Séries de Covariáveis [Feriados + Globais])
        """
        print("⚙️ Materializando dados do Spark para Pandas (Driver)...")
        # CUIDADO: Trazendo dados para a memória do Driver.
        # Para datasets massivos, considerar processamento distribuído (Fugue/PandasUDF), 
        # mas para séries agregadas por loja, geralmente cabe na memória.
        df_wide = df_spark_wide.toPandas()
        
        print(f"   DEBUG: Columns before dedupe: {list(df_wide.columns)}")
        # Remove colunas duplicadas (pode acontecer se o join trouxe chaves repetidas)
        df_wide = df_wide.loc[:, ~df_wide.columns.duplicated()]
        print(f"   DEBUG: Columns after dedupe: {list(df_wide.columns)}")
        
        # Verificação crítica de tipagem da coluna loja
        if "codigo_loja" in df_wide.columns:
             col_obj = df_wide["codigo_loja"]
             # Se por algum motivo 'codigo_loja' ainda for um DataFrame (duplicidade extrema), forçamos uma correção.
             if isinstance(col_obj, pd.DataFrame):
                  print("   ⚠️ CRITICAL: 'codigo_loja' is still a DataFrame (duplicate columns)!")
                  df_wide = df_wide.loc[:, ~df_wide.columns.duplicated(keep='first')]

        if df_wide.empty:
            print("⚠️ AVISO: DataFrame df_wide está vazio! Verifique os filtros de data e dados.")
            return [], []

        df_wide['data'] = pd.to_datetime(df_wide['data'])
        
        # Definição das colunas que são estáticas (não mudam com o tempo para uma mesma loja)
        possible_static = ["cluster_loja", "sigla_uf", "tipo_loja", "modelo_loja"]
        static_cols = [c for c in possible_static if c in df_wide.columns]

        # --- CRIAÇÃO DAS TARGET SERIES (O que queremos prever) ---
        print("   Build: Criando Target Series (Vetorizado)...")
        try:
            # from_group_dataframe é o método mais eficiente para criar múltiplas séries de um DF longo
            target_series_list = TimeSeries.from_group_dataframe(
                df_wide,
                group_cols="codigo_loja",
                time_col="data",
                value_cols="target_vendas",
                static_cols=static_cols, # Associa features de loja como covariáveis estáticas
                freq='D',
                fill_missing_dates=True,
                fillna_value=0.0
            )
        except Exception as e:
            print(f"❌ Erro crítico no from_group_dataframe (Target): {e}")
            raise e

        # Mapa para acesso rápido (ID Loja -> TimeSeries)
        target_dict = {}
        for ts in target_series_list:
            if ts.static_covariates is not None and not ts.static_covariates.empty:
                # Ajuste técnico: Garantir que o índice da covariável estática tenha nome
                if ts.static_covariates.index.name == "target_vendas":
                     ts.with_static_covariates(ts.static_covariates.rename_axis("codigo_loja"))
                
                # Extrai ID da loja da covariável estática (Darts coloca o ID do grupo lá)
                key_val = str(ts.static_covariates.index[0]).replace(".0", "")
                target_dict[key_val] = ts
        
        valid_stores = list(target_dict.keys())
        print(f"   ℹ️ Lojas identificadas: {len(valid_stores)}")

        # --- CRIAÇÃO DAS COVARIÁVEIS LOCAIS (Feriados) ---
        print("   Build: Criando Covariáveis Locais...")
        try:
            # Covariáveis passadas/futuras conhecidas específicas por loja
            feriado_series_list = TimeSeries.from_group_dataframe(
                df_wide,
                group_cols="codigo_loja",
                time_col="data",
                value_cols="is_feriado",
                static_cols=static_cols,
                freq='D',
                fill_missing_dates=True,
                fillna_value=0.0
            )
        except Exception as e:
            print(f"❌ Erro crítico no from_group_dataframe (Feriado): {e}")
            raise e

        feriado_dict = {
            str(ts.static_covariates.index[0]).replace(".0", ""): ts 
            for ts in feriado_series_list
            if ts.static_covariates is not None and not ts.static_covariates.empty
        }

        # --- PREPARAÇÃO DE COVARIÁVEIS GLOBAIS ---
        print("   Build: Preparando Covariáveis Globais...")
        ts_support = TimeSeries.from_dataframe(df_global_support, fill_missing_dates=True, freq='D', fillna_value=0.0)
        
        # Combina (stack) suporte global com indicadores de mercado (se houver)
        if df_market_indicators is not None:
             ts_market = TimeSeries.from_dataframe(df_market_indicators, fill_missing_dates=True, freq='D', fillna_value=0.0)
             global_covariates = ts_support.stack(ts_market)
        else:
             global_covariates = ts_support

        # Features de Calendário (Sazonalidade)
        # Adiciona dia da semana, trimestre, semana do ano, etc.
        ts_time = datetime_attribute_timeseries(df_global_support.index, attribute="dayofweek", cyclic=True)
        ts_time = ts_time.stack(datetime_attribute_timeseries(df_global_support.index, attribute="quarter", one_hot=True))
        ts_time = ts_time.stack(datetime_attribute_timeseries(df_global_support.index, attribute="week", cyclic=True))
        
        # Stack Final Global: Suporte + Mercado + Calendário
        global_covariates = global_covariates.stack(ts_time)

        final_target_list = []
        full_covariates_list = []

        print("   Build: Stacking Final (Otimizado)...")
        # Combina Covariáveis GLOBAIS + LOCAIS para cada loja
        for loja in valid_stores:
            ts_target = target_dict[loja]
            final_target_list.append(ts_target)
            
            ts_local = feriado_dict.get(loja)
            
            # Se não existir feriado para a loja (caso raro), cria zerado
            if ts_local is None:
                ts_local = TimeSeries.from_times_and_values(
                    ts_target.time_index, 
                    np.zeros((len(ts_target), 1)), 
                    freq='D',
                    columns=["is_feriado"]
                )
            else:
                # Garante sincronia temporal (interseção)
                if ts_local.start_time() != ts_target.start_time() or ts_local.end_time() != ts_target.end_time():
                    ts_local = ts_local.slice_intersect(ts_target)

            # Sincroniza globais com o período da loja
            if global_covariates.start_time() != ts_target.start_time() or global_covariates.end_time() != ts_target.end_time():
                 ts_global_cut = global_covariates.slice_intersect(ts_target)
            else:
                 ts_global_cut = global_covariates

            # STACK: Une Global + Local em uma única série multivalorada de features
            full_covariates_list.append(ts_global_cut.stack(ts_local))

        print(f"✅ Objetos Darts Prontos: {len(final_target_list)} lojas processadas.")
        return final_target_list, full_covariates_list