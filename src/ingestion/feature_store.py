from typing import List, Union, Optional
from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import SparkSession, DataFrame

def salvar_feature_table(
    df: DataFrame, 
    table_name_full: str, 
    pk_columns: Union[str, List[str]], 
    timestamp_col: Optional[str] = None, 
    spark: Optional[SparkSession] = None
) -> None:
    """
    Salva ou atualiza uma tabela no Feature Store com melhores práticas (Liquid Clustering/Optimize).

    Args:
        df (DataFrame): O DataFrame PySpark contendo os features.
        table_name_full (str): Nome completo da tabela (catalog.schema.table).
        pk_columns (Union[str, List[str]]): Coluna(s) chave primária.
        timestamp_col (Optional[str]): Coluna de timestamp para Point-in-time lookup.
        spark (Optional[SparkSession]): Sessão Spark ativa.

    Returns:
        None
    """
    if spark is None:
        spark = SparkSession.builder.getOrCreate()
        
    fe = FeatureEngineeringClient()

    # 1. Normalização de Inputs
    if isinstance(pk_columns, str):
        pk_columns = [pk_columns]

    # Cria lista de verificação (PKs + Timestamp se houver)
    check_keys = pk_columns.copy()
    
    # Regra do Feature Store: Se tem timestamp, ele DEVE estar na lista de PKs
    if timestamp_col:
        if timestamp_col not in pk_columns:
            pk_columns.append(timestamp_col)
        if timestamp_col not in check_keys:
            check_keys.append(timestamp_col)

    # --- CORREÇÃO 1: REMOVER NULOS ---
    # Chaves Primárias no Feature Store NÃO podem ser nulas.
    # O erro "NOT NULL constraint violated" acontece aqui se não limparmos.
    print(f"   🧹 Removendo Nulos nas chaves: {check_keys}...")
    df = df.dropna(subset=check_keys)

    # --- CORREÇÃO 2: REMOVER DUPLICATAS ---
    # Garante unicidade
    print(f"   🧹 Removendo duplicatas nas chaves: {check_keys}...")
    df = df.dropDuplicates(check_keys)

    # 2. Tentativa de Atualização ou Criação Limpa
    try:
        # Tenta carregar a tabela como Feature Table
        fe.get_table(name=table_name_full)
        print(f"🔄 [UPDATE] Tabela encontrada no Feature Store: {table_name_full}")
        
        fe.write_table(
            name=table_name_full,
            df=df,
            mode="merge"
        )
        
        # --- BEST PRACTICE: OTIMIZAÇÃO CONTÍNUA ---
        print(f"   ⚡ Otimizando a tabela (Liquid/Z-Order + Compactação)...")
        spark.sql(f"OPTIMIZE {table_name_full}")
        spark.sql(f"VACUUM {table_name_full} RETAIN 168 HOURS") # Limpa arquivos antigos (>7 dias)
        
    except Exception:
        # Se cair aqui, verifica se tabela existe como Delta comum e remove
        if spark.catalog.tableExists(table_name_full):
            print(f"⚠️ [CLEANUP] Tabela existe mas sem restrições de Feature Store. Removendo: {table_name_full}")
            spark.sql(f"DROP TABLE IF EXISTS {table_name_full}")
            
        print(f"🆕 [CREATE] Criando nova Feature Table: {table_name_full}")
        print(f"   🔑 PKs: {pk_columns} | 🕒 Time: {timestamp_col}")
        
        # --- BEST PRACTICE: LIQUID CLUSTERING (Recomendado pela Databricks ao invés de Z-Order/Partition) ---
        fe.create_table(
            name=table_name_full,
            primary_keys=pk_columns,
            timestamp_keys=timestamp_col,
            df=df,
            schema=df.schema,
            description="Ingested via JDBC for Feature Store"
            # table_properties={"delta.enableLiquidClustering": "true"} # REMOVED: SDK compatibility issue
        )
        
        # Garante otimização inicial
        print(f"   ⚡ Otimizando layout inicial...")
        spark.sql(f"OPTIMIZE {table_name_full}")
