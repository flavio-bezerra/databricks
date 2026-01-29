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
    Responsável pela ingestão de dados, criação de Feature Sets e preparação para o Darts.
    """
    def __init__(self, spark_session: SparkSession, config: Any):
        self.spark: SparkSession = spark_session
        self.config = config
        self.fe = FeatureEngineeringClient()

    def create_training_set(self) -> DataFrame:
        """
        Cria o conjunto de treinamento usando Feature Store.
        
        Returns:
            DataFrame: DataFrame Spark pronto para uso.
        """
        print("🛒 Construindo Training Set via Feature Store (Spark Native)...")
        target_table = f"{self.config.CATALOG}.{self.config.SCHEMA}.historico_targuet_loja"
        
        df_spine = (self.spark.table(target_table)
                    .filter(F.col("data").between(self.config.DATA_START, self.config.INGESTION_END))
                    .select("codigo_loja", "data", "valor")
                    .withColumnRenamed("valor", "target_vendas")
                    .withColumn("codigo_loja", F.col("codigo_loja").cast("string"))
                   )

        feature_lookups = [
            FeatureLookup(
                table_name=f"{self.config.CATALOG}.{self.config.SCHEMA}.lojas_fs",
                lookup_key=["codigo_loja"],
                feature_names=["cluster_loja", "sigla_uf", "tipo_loja", "modelo_loja"]
            ),
            FeatureLookup(
                table_name=f"{self.config.CATALOG}.{self.config.SCHEMA}.historico_feriados_loja",
                lookup_key=["codigo_loja"],
                timestamp_lookup_key="data",
                feature_names=["valor"], 
                rename_outputs={"valor": "is_feriado"}
            )
        ]

        training_set = self.fe.create_training_set(
            df=df_spine,
            feature_lookups=feature_lookups,
            label="target_vendas",
            exclude_columns=[]
        )

        df_spark = training_set.load_df()

        print("   ⚡ Executando limpeza e tratamento no Spark Cluster...")
        df_spark = df_spark.na.fill({
            "is_feriado": 0.0, 
            "target_vendas": 0.0,
            "cluster_loja": "DESCONHECIDO",
            "sigla_uf": "DESCONHECIDO",
            "tipo_loja": "DESCONHECIDO",
            "modelo_loja": "DESCONHECIDO"
        })

        df_spark = df_spark.withColumn("data", F.to_timestamp("data"))
        return df_spark

    def get_global_support(self) -> pd.DataFrame:
        """
        Calcula e retorna dados de suporte global agregados com extensão futura.
        """
        table_name = "historico_suporte_loja"
        print(f"🌍 Carregando suporte global (Spark Aggregation)...")

        # Atribuição correta à variável df_spark
        df_spark = (self.spark.table(f"{self.config.CATALOG}.{self.config.SCHEMA}.{table_name}")
            .filter(F.col("DATA").between(self.config.DATA_START, self.config.INGESTION_END))
            .groupBy("data")
            .pivot("metricas")
            .agg(F.sum("valor"))
            .na.fill(0.0)
        )

        # Conversão para Pandas usando a variável definida acima
        pdf = df_spark.toPandas() 
        pdf['data'] = pd.to_datetime(pdf['data'])
        pdf = pdf.set_index('data').asfreq('D').fillna(0.0)

        # --- Lógica de Extensão para o Horizonte de Predição ---
        # Garante que as covariáveis cubram o treino + horizonte de predição (n)
        full_range = pd.date_range(
            start=pdf.index.min(), 
            periods=len(pdf) + self.config.FORECAST_HORIZON, 
            freq='D'
        )
        return pdf.reindex(full_range).ffill().fillna(0.0)

    def build_darts_objects(
        self, 
        df_spark_wide: DataFrame, 
        df_global_support: pd.DataFrame, 
        df_market_indicators: Optional[pd.DataFrame] = None
    ) -> Tuple[List[TimeSeries], List[TimeSeries]]:
        """
        Converte DataFrames para objetos TimeSeries do Darts com Reindexação Universal.
        Garante que todas as séries comecem em DATA_START, evitando erros de drop_after.
        """
        print(f"⚙️ Materializando dados e aplicando Reindexação Universal (Início: {self.config.DATA_START})...")
        df_pd_raw = df_spark_wide.toPandas()

        # --- Saneamento Atômico: Resolve erro de 1-dimensional ---
        clean_dict = {}
        for col in df_pd_raw.columns.unique():
            col_name = str(col).strip()
            series_data = df_pd_raw[col]
            if isinstance(series_data, pd.DataFrame):
                series_data = series_data.iloc[:, 0]
            clean_dict[col_name] = series_data.values.flatten()
        
        df_pd = pd.DataFrame(clean_dict)
        if df_pd.empty:
            return [], []

        # Formatação de tipos
        df_pd['codigo_loja'] = df_pd['codigo_loja'].astype(str).str.replace(r'\.0$', '', regex=True)
        df_pd['data'] = pd.to_datetime(df_pd['data'])
        
        # Range total do projeto para uniformizar as séries
        full_project_range = pd.date_range(
            start=self.config.DATA_START, 
            end=df_pd['data'].max(), 
            freq='D'
        )

        static_features = ["cluster_loja", "sigla_uf", "tipo_loja", "modelo_loja"]
        available_features = [c for c in static_features if c in df_pd.columns]

        target_series_list = []
        target_dict = {}
        feriado_dict = {}

        print("   Build: Processando lojas com preenchimento de histórico faltante...")
        for store_id, group_df in df_pd.groupby("codigo_loja"):
            # 1. Uniformização Temporal (Reindex)
            # Garante que lojas novas tenham histórico zerado desde o início do projeto
            temp_df = (group_df.set_index('data')
                       .reindex(full_project_range)
                       .reset_index()
                       .rename(columns={'index': 'data'}))
            
            # Preenche valores faltantes (0 para métricas, ffill/bfill para metadados da loja)
            temp_df['target_vendas'] = temp_df['target_vendas'].fillna(0.0)
            temp_df['is_feriado'] = temp_df['is_feriado'].fillna(0.0)
            temp_df[available_features] = temp_df[available_features].ffill().bfill()
            
            # 2. Criar Target Series (Renomeando para ID da Loja)
            ts_target = TimeSeries.from_dataframe(
                temp_df.rename(columns={"target_vendas": store_id}), 
                time_col="data", value_cols=[store_id], 
                freq='D', fill_missing_dates=True, fillna_value=0.0
            )
            
            # Metadados estáticos com nome do índice 'codigo_loja'
            static_df = temp_df[available_features].iloc[0:1].copy()
            static_df.index = pd.Index([store_id], name="codigo_loja")
            ts_target = ts_target.with_static_covariates(static_df)
            
            target_dict[store_id] = ts_target
            target_series_list.append(ts_target)

            # 3. Criar Feriado Series
            ts_f = TimeSeries.from_dataframe(
                temp_df.rename(columns={"is_feriado": f"feriado_{store_id}"}), 
                time_col="data", value_cols=[f"feriado_{store_id}"],
                freq='D', fill_missing_dates=True, fillna_value=0.0
            )
            feriado_dict[store_id] = ts_f

        # --- Processamento de Covariáveis Globais ---
        print("   Build: Preparando Covariáveis Globais e Calendário...")
        ts_support = TimeSeries.from_dataframe(df_global_support, fill_missing_dates=True, freq='D', fillna_value=0.0)
        
        if df_market_indicators is not None:
             ts_market = TimeSeries.from_dataframe(df_market_indicators, fill_missing_dates=True, freq='D', fillna_value=0.0)
             global_covariates = ts_support.stack(ts_market)
        else:
             global_covariates = ts_support

        # Calendário baseado no range do suporte (já estendido para o futuro)
        ts_time = datetime_attribute_timeseries(df_global_support.index, attribute="dayofweek", cyclic=True)
        ts_time = ts_time.stack(datetime_attribute_timeseries(df_global_support.index, attribute="quarter", one_hot=True))
        ts_time = ts_time.stack(datetime_attribute_timeseries(df_global_support.index, attribute="week", cyclic=True))
        global_covariates = global_covariates.stack(ts_time)

        final_target_list = []
        full_covariates_list = []

        print("   Build: Stacking Final com Alinhamento Temporal...")
        for loja in target_dict.keys():
            ts_target = target_dict[loja]
            final_target_list.append(ts_target)
            
            # Sincroniza suporte global (Início igual ao target, Fim estendido para o futuro)
            ts_global_cut = global_covariates.drop_before(ts_target.start_time())
            
            # Sincroniza e estende Feriados para casar com o suporte global
            ts_local = feriado_dict[loja]
            if ts_local.end_time() < ts_global_cut.end_time():
                df_feriado = ts_local.pd_dataframe().reindex(ts_global_cut.time_index).fillna(0.0)
                ts_local = TimeSeries.from_dataframe(df_feriado, freq='D')
            else:
                ts_local = ts_local.slice_intersect(ts_global_cut)

            full_covariates_list.append(ts_global_cut.stack(ts_local))

        print(f"✅ Objetos Darts Prontos: {len(final_target_list)} lojas uniformizadas.")
        return final_target_list, full_covariates_list