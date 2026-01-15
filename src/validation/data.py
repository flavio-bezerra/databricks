from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
import pyspark.sql.functions as F
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.utils.timeseries_generation import datetime_attribute_timeseries

class DataIngestion:
    def __init__(self, spark_session, config):
        self.spark = spark_session
        self.config = config
        self.fe = FeatureEngineeringClient()

    def create_training_set(self):
        """
        Gera o dataset completo via Feature Store e realiza ETL nativo no Spark.
        Retorna: DataFrame PySpark (Lazy Evaluation)
        """
        print("🛒 Construindo Training Set via Feature Store (Spark Native)...")

        # 1. Definir a 'Spine' (Target)
        target_table = f"{self.config.CATALOG}.{self.config.SCHEMA}.bip_vhistorico_targuet_loja"
        
        df_spine = (self.spark.table(target_table)
                    .filter(F.col("DATA").between(self.config.DATA_START, self.config.INGESTION_END))
                    .select("CODIGO_LOJA", "DATA", "VALOR")
                    .withColumnRenamed("VALOR", "TARGET_VENDAS")
                    .withColumn("CODIGO_LOJA", F.col("CODIGO_LOJA").cast("string"))
                   )

        # 2. Configurar Lookups
        feature_lookups = [
            FeatureLookup(
                table_name=f"{self.config.CATALOG}.{self.config.SCHEMA}.cmc_alojas",
                lookup_key=["CODIGO_LOJA"],
                feature_names=["CLUSTER_LOJA", "SIGLA_UF", "TIPO_LOJA", "MODELO_LOJA"]
            ),
            FeatureLookup(
                table_name=f"{self.config.CATALOG}.{self.config.SCHEMA}.bip_vhistorico_feriados_loja",
                lookup_key=["CODIGO_LOJA"],
                timestamp_lookup_key="DATA",
                feature_names=["VALOR"], 
                output_name="IS_FERIADO"
            )
        ]

        # 3. Criar Training Set (Retorna objeto FeatureEngineeringTrainingSet)
        training_set = self.fe.create_training_set(
            df=df_spine,
            feature_lookups=feature_lookups,
            label="TARGET_VENDAS",
            exclude_columns=[]
        )

        # 4. Carregar como DataFrame Spark (SEM toPandas aqui)
        df_spark = training_set.load_df()

        # --- ETL NATIVO NO SPARK (Distribuído) ---
        print("   ⚡ Executando limpeza e tratamento no Spark Cluster...")
        
        # A. Tratamento de Nulos (Left Joins geram nulos)
        # Sintaxe Spark: fill(valor, subset=[colunas]) ou fill({col: val})
        df_spark = df_spark.na.fill({
            "IS_FERIADO": 0.0, 
            "TARGET_VENDAS": 0.0,
            "CLUSTER_LOJA": "DESCONHECIDO",
            "SIGLA_UF": "DESCONHECIDO",
            "TIPO_LOJA": "DESCONHECIDO",
            "MODELO_LOJA": "DESCONHECIDO"
        })

        # B. Garantia de Tipos (Casting)
        df_spark = df_spark.withColumn("DATA", F.to_timestamp("DATA"))
        
        return df_spark

    def get_global_support(self):
        """
        Carrega suporte global mantendo processamento no Spark até o final.
        """
        table_name = "bip_vhistorico_suporte_canal_loja"
        print(f"🌍 Carregando suporte global (Spark Aggregation)...")
        
        # Toda a agregação ocorre no cluster
        df_spark = (self.spark.table(f"{self.config.CATALOG}.{self.config.SCHEMA}.{table_name}")
            .filter(F.col("DATA").between(self.config.DATA_START, self.config.INGESTION_END))
            .groupBy("DATA")
            .pivot("METRICAS")
            .agg(F.sum("VALOR"))
            .na.fill(0.0) # Preenche nulos do pivot no Spark
        )
        
        # Só converte o resultado final (pequeno) para Pandas
        pdf = df_spark.toPandas()
        pdf['DATA'] = pd.to_datetime(pdf['DATA'])
        return pdf.set_index('DATA').asfreq('D').fillna(0.0)

    def build_darts_objects(self, df_spark_wide, df_global_support):
        """
        Recebe Spark DataFrame -> Converte para Pandas -> Cria objetos Darts
        """
        print("⚙️ Materializando dados do Spark para Pandas (Driver)...")
        
        # AQUI acontece a transferência de dados Cluster -> Driver
        # Como já filtramos e limpamos no Spark, o dado vem menor e mais limpo.
        df_wide = df_spark_wide.toPandas()
        
        # Garante tipos Pandas compatíveis com Darts
        df_wide['DATA'] = pd.to_datetime(df_wide['DATA'])
        
        # Identificação dinâmica de colunas estáticas
        possible_static = ["CLUSTER_LOJA", "SIGLA_UF", "TIPO_LOJA", "MODELO_LOJA"]
        static_cols = [c for c in possible_static if c in df_wide.columns]

        print("   Build: Criando Target Series...")
        target_series_list = TimeSeries.from_group_dataframe(
            df_wide,
            group_cols="CODIGO_LOJA",
            time_col="DATA",
            value_cols="TARGET_VENDAS",
            static_cols=static_cols,
            freq='D',
            fill_missing_dates=True,
            fillna_value=0.0
        )
        
        target_dict = {str(ts.static_covariates.index[0]): ts for ts in target_series_list}
        valid_stores = list(target_dict.keys())

        print("   Build: Criando Covariáveis Locais...")
        feriado_series_list = TimeSeries.from_group_dataframe(
            df_wide,
            group_cols="CODIGO_LOJA",
            time_col="DATA",
            value_cols=["IS_FERIADO"], 
            freq='D',
            fill_missing_dates=True,
            fillna_value=0.0
        )
        feriado_dict = {str(ts.static_covariates["CODIGO_LOJA"].iloc[0]): ts for ts in feriado_series_list}

        # Globais (já vieram prontas do método get_global_support)
        ts_support = TimeSeries.from_dataframe(
            df_global_support, 
            fill_missing_dates=True, 
            freq='D',
            fillna_value=0.0
        )
        
        ts_time = datetime_attribute_timeseries(df_global_support.index, attribute="dayofweek", cyclic=True)
        ts_time = ts_time.stack(datetime_attribute_timeseries(df_global_support.index, attribute="quarter", one_hot=True))
        ts_time = ts_time.stack(datetime_attribute_timeseries(df_global_support.index, attribute="week", cyclic=True))
        
        global_covariates = ts_support.stack(ts_time)

        final_target_list = []
        full_covariates_list = []

        print("   Build: Stacking Final...")
        for loja in valid_stores:
            ts_target = target_dict[loja]
            final_target_list.append(ts_target)
            
            ts_local = feriado_dict.get(loja)
            if ts_local:
                 ts_local = ts_local.slice_intersect(ts_target)
            else:
                 ts_local = TimeSeries.from_times_and_values(ts_target.time_index, np.zeros(len(ts_target)), freq='D')
            
            ts_global = global_covariates.slice_intersect(ts_target)
            full_covariates_list.append(ts_global.stack(ts_local))

        print(f"✅ Objetos Darts Prontos: {len(final_target_list)} lojas.")
        return final_target_list, full_covariates_list
