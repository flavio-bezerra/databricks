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
    Ajustado para garantir cobertura de covariáveis (3 meses de margem).
    """
    def __init__(self, spark_session: SparkSession, config: Any):
        self.spark: SparkSession = spark_session
        self.config = config
        self.fe = FeatureEngineeringClient()

    def create_training_set(self) -> DataFrame:
        """
        Cria o conjunto de treinamento usando Feature Store.
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

        df_spark = (self.spark.table(f"{self.config.CATALOG}.{self.config.SCHEMA}.{table_name}")
            .filter(F.col("DATA").between(self.config.DATA_START, self.config.INGESTION_END))
            .groupBy("data")
            .pivot("metricas")
            .agg(F.sum("valor"))
            .na.fill(0.0)
        )

        pdf = df_spark.toPandas() 
        pdf['data'] = pd.to_datetime(pdf['data'])
        pdf = pdf.set_index('data').asfreq('D').fillna(0.0)

        # Extensão base para garantir o horizonte de previsão
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
        Converte DataFrames para objetos TimeSeries do Darts.
        Aplica margem de 3 meses nas covariáveis para evitar erros de índice no fit/predict.
        """
        print(f"⚙️ Materializando dados e aplicando margem de segurança temporal...")
        df_pd_raw = df_spark_wide.toPandas()

        # Saneamento de colunas
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

        df_pd['codigo_loja'] = df_pd['codigo_loja'].astype(str).str.replace(r'\.0$', '', regex=True)
        df_pd['data'] = pd.to_datetime(df_pd['data'])
        
        # Range original do projeto para as Targets
        full_project_range = pd.date_range(
            start=self.config.DATA_START, 
            end=df_pd['data'].max(), 
            freq='D'
        )

        # --- DEFINIÇÃO DO RANGE EXPANDIDO PARA COVARIÁVEIS (±3 Meses) ---
        covariates_range = pd.date_range(
            start=pd.to_datetime(self.config.DATA_START) - pd.DateOffset(months=3),
            end=df_global_support.index.max() + pd.DateOffset(months=3),
            freq='D'
        )

        static_features = ["cluster_loja", "sigla_uf", "tipo_loja", "modelo_loja"]
        available_features = [c for c in static_features if c in df_pd.columns]

        target_series_list = []
        target_dict = {}
        feriado_dict = {}

        print("   Build: Processando lojas e feriados locais...")
        for store_id, group_df in df_pd.groupby("codigo_loja"):
            # Target usa o range normal
            temp_df_target = (group_df.set_index('data')
                              .reindex(full_project_range)
                              .reset_index()
                              .rename(columns={'index': 'data'}))
            
            temp_df_target['target_vendas'] = temp_df_target['target_vendas'].fillna(0.0)
            
            ts_target = TimeSeries.from_dataframe(
                temp_df_target.rename(columns={"target_vendas": str(store_id)}), 
                time_col="data", 
                value_cols=[str(store_id)], 
                freq='D', 
                fill_missing_dates=True, 
                fillna_value=0.0
            )
            
            static_df = group_df[available_features].iloc[0:1].copy()
            static_df.index = pd.Index([str(store_id)], name="codigo_loja")
            ts_target = ts_target.with_static_covariates(static_df)
            
            target_dict[store_id] = ts_target
            target_series_list.append(ts_target)

            # Feriado (Covariável Local) usa o range expandido
            temp_df_cov = (group_df.set_index('data')
                           .reindex(covariates_range)
                           .fillna(0.0)
                           .reset_index()
                           .rename(columns={'index': 'data'}))

            ts_f = TimeSeries.from_dataframe(
                temp_df_cov.rename(columns={"is_feriado": f"feriado_{store_id}"}), 
                time_col="data", value_cols=[f"feriado_{store_id}"],
                freq='D', fill_missing_dates=True, fillna_value=0.0
            )
            feriado_dict[store_id] = ts_f

        # --- PREPARAÇÃO DAS COVARIÁVEIS GLOBAIS (Expandidas) ---
        df_global_ext = df_global_support.reindex(covariates_range).ffill().bfill().fillna(0.0)
        ts_support = TimeSeries.from_dataframe(df_global_ext, fill_missing_dates=True, freq='D', fillna_value=0.0)
        
        if df_market_indicators is not None:
             df_market_ext = df_market_indicators.reindex(covariates_range).ffill().bfill().fillna(0.0)
             ts_market = TimeSeries.from_dataframe(df_market_ext, fill_missing_dates=True, freq='D', fillna_value=0.0)
             global_covariates = ts_support.stack(ts_market)
        else:
             global_covariates = ts_support

        # Features de tempo baseadas no range expandido
        ts_time = datetime_attribute_timeseries(covariates_range, attribute="dayofweek", cyclic=True)
        ts_time = ts_time.stack(datetime_attribute_timeseries(covariates_range, attribute="quarter", one_hot=True))
        ts_time = ts_time.stack(datetime_attribute_timeseries(covariates_range, attribute="week", cyclic=True))
        global_covariates = global_covariates.stack(ts_time)

        final_target_list = []
        full_covariates_list = []

        print("   Build: Stacking Final com Margem de Segurança...")
        for loja in target_dict.keys():
            final_target_list.append(target_dict[loja])
            # Stack da global expandida com a local (feriados) expandida
            full_covariates_list.append(global_covariates.stack(feriado_dict[loja]))

        print(f"✅ Objetos Darts Prontos com margem temporal: {len(final_target_list)} lojas.")
        return final_target_list, full_covariates_list