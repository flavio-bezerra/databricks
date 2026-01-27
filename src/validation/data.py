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
        Calcula e retorna dados de suporte global agregados.
        
        Returns:
            pd.DataFrame: DataFrame Pandas indexado por data.
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
        return pdf.set_index('data').asfreq('D').fillna(0.0)

    def build_darts_objects(
        self, 
        df_spark_wide: DataFrame, 
        df_global_support: pd.DataFrame, 
        df_market_indicators: Optional[pd.DataFrame] = None
    ) -> Tuple[List[TimeSeries], List[TimeSeries]]:
        """
        Converte DataFrames para objetos TimeSeries do Darts.

        Args:
            df_spark_wide (DataFrame): DataFrame Spark com dados das lojas.
            df_global_support (pd.DataFrame): DataFrame Pandas de suporte global.
            df_market_indicators (Optional[pd.DataFrame]): Indicadores de mercado opcionais.

        Returns:
            Tuple[List[TimeSeries], List[TimeSeries]]: Listas de Target Series e Covariates.
        """
        print("⚙️ Materializando dados do Spark para Pandas (Driver)...")
        df_wide = df_spark_wide.toPandas()
        df_wide['data'] = pd.to_datetime(df_wide['data'])
        
        possible_static = ["codigo_loja", "cluster_loja", "sigla_uf", "tipo_loja", "modelo_loja"]
        static_cols = [c for c in possible_static if c in df_wide.columns]

        print("   Build: Criando Target Series (Vetorizado)...")
        target_series_list = TimeSeries.from_group_dataframe(
            df_wide,
            group_cols="codigo_loja",
            time_col="data",
            value_cols="target_vendas",
            static_cols=static_cols,
            freq='D',
            fill_missing_dates=True,
            fillna_value=0.0
        )
        
        target_dict = {str(ts.static_covariates["codigo_loja"].iloc[0]): ts for ts in target_series_list}
        valid_stores = list(target_dict.keys())

        print("   Build: Criando Covariáveis Locais...")
        feriado_series_list = TimeSeries.from_group_dataframe(
            df_wide,
            group_cols="codigo_loja",
            time_col="data",
            value_cols="is_feriado",
            freq='D',
            fill_missing_dates=True,
            fillna_value=0.0
        )
        feriado_dict = {str(ts.static_covariates["codigo_loja"].iloc[0]): ts for ts in feriado_series_list}

        # --- Stack Global ---
        ts_support = TimeSeries.from_dataframe(df_global_support, fill_missing_dates=True, freq='D', fillna_value=0.0)
        
        if df_market_indicators is not None:
             ts_market = TimeSeries.from_dataframe(df_market_indicators, fill_missing_dates=True, freq='D', fillna_value=0.0)
             global_covariates = ts_support.stack(ts_market)
        else:
             global_covariates = ts_support

        # Features de Calendário (Reproduzindo lógica de Treino)
        ts_time = datetime_attribute_timeseries(df_global_support.index, attribute="dayofweek", cyclic=True)
        ts_time = ts_time.stack(datetime_attribute_timeseries(df_global_support.index, attribute="quarter", one_hot=True))
        ts_time = ts_time.stack(datetime_attribute_timeseries(df_global_support.index, attribute="week", cyclic=True))
        global_covariates = global_covariates.stack(ts_time)

        final_target_list = []
        full_covariates_list = []

        print("   Build: Stacking Final (Otimizado)...")
        for loja in valid_stores:
            ts_target = target_dict[loja]
            final_target_list.append(ts_target)
            ts_local = feriado_dict.get(loja)
            if ts_local is None:
                ts_local = TimeSeries.from_times_and_values(
                    ts_target.time_index, 
                    np.zeros((len(ts_target), 1)), 
                    freq='D',
                    columns=["is_feriado"]
                )
            else:
                if ts_local.start_time() != ts_target.start_time() or ts_local.end_time() != ts_target.end_time():
                    ts_local = ts_local.slice_intersect(ts_target)

            if global_covariates.start_time() != ts_target.start_time() or global_covariates.end_time() != ts_target.end_time():
                 ts_global_cut = global_covariates.slice_intersect(ts_target)
            else:
                 ts_global_cut = global_covariates

            full_covariates_list.append(ts_global_cut.stack(ts_local))

        print(f"✅ Objetos Darts Prontos: {len(final_target_list)} lojas processadas.")
        return final_target_list, full_covariates_list
