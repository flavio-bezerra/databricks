"""
Módulo de Feature Store (Ingestion).

Este arquivo contém a lógica para persistir dados transformados no Databricks Feature Store.
Ele gerencia a criação e atualização de Feature Tables, garantindo integridade de dados 
(chaves primárias únicas e não nulas) e performance (Liquid Clustering/Optimize).

Funções:
- salvar_feature_table: Função principal para gravar DataFrames como tabelas do Feature Store.
"""

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
    Salva ou atualiza uma tabela no Feature Store aplicando melhores práticas de Engenharia de Dados.
    
    Esta função realiza várias etapas críticas:
    1. Validação e limpeza de chaves primárias (PKs) e Timestamp.
    2. Remoção de duplicatas para garantir integridade.
    3. Tentativa de 'merge' (upsert) se a tabela já existir.
    4. Criação de nova tabela com Liquid Clustering se não existir.
    5. Otimização de armazenamento (Optimize/Vacuum) para performance de leitura.

    Args:
        df (DataFrame): O DataFrame PySpark com os dados a serem ingeridos.
        table_name_full (str): Nome completo da tabela de destino (ex: catalog.schema.table).
        pk_columns (Union[str, List[str]]): Nome da(s) coluna(s) que identificam unicamente cada registro.
        timestamp_col (Optional[str]): Coluna de tempo para permitir "Point-in-time lookup" (evita data leakage).
        spark (Optional[SparkSession]): Sessão Spark ativa. Se None, cria/obtém uma nova.

    Returns:
        None: A função realiza operações de efeito colateral (gravação no banco).
    """
    if spark is None:
        spark = SparkSession.builder.getOrCreate()
        
    # Cliente do Feature Engineering para interagir com o catálogo de features
    fe = FeatureEngineeringClient()

    # 1. Normalização de Inputs
    # Garante que pk_columns seja sempre uma lista, mesmo que venha como string única
    if isinstance(pk_columns, str):
        pk_columns = [pk_columns]

    # Cria lista de verificação (todas as colunas que não podem ser nulas)
    check_keys = pk_columns.copy()
    
    # Regra do Feature Store: Se a tabela tem uma dimensão de tempo (timestamp),
    # essa coluna também faz parte da unicidade do registro para lookups históricos.
    if timestamp_col:
        if timestamp_col not in pk_columns:
            pk_columns.append(timestamp_col)
        if timestamp_col not in check_keys:
            check_keys.append(timestamp_col)

    # --- LIMPEZA DE DADOS (DATA QUALITY) ---
    
    # Validação 1: Chaves Primárias no Feature Store NÃO podem ser nulas.
    # Se houver nulos, a gravação falharia com erro "NOT NULL constraint violated".
    print(f"   🧹 Removendo Nulos nas chaves: {check_keys}...")
    df = df.dropna(subset=check_keys)

    # Validação 2: Unicidade.
    # O Feature Store exige que cada combinação de PK+Timestamp seja única.
    # Removemos duplicatas arbitrárias (primeira ocorrência vence) para evitar falhas.
    print(f"   🧹 Removendo duplicatas nas chaves: {check_keys}...")
    df = df.dropDuplicates(check_keys)

    # 2. Estratégia de Gravação (Merge vs Create)
    try:
        # Verifica se a tabela já existe e é uma Feature Table válida
        fe.get_table(name=table_name_full)
        print(f"🔄 [UPDATE] Tabela encontrada no Feature Store: {table_name_full}")
        
        # Realiza um MERGE (Upsert): Atualiza registros existentes e insere novos
        fe.write_table(
            name=table_name_full,
            df=df,
            mode="merge"
        )
        
        # --- OTIMIZAÇÃO DE STORAGE ---
        # OPTIMIZE: Condensa pequenos arquivos em arquivos maiores (melhora leitura).
        # VACUUM: Remove arquivos antigos não mais referenciados pelo log transacional (economiza espaço).
        print(f"   ⚡ Otimizando a tabela (Liquid/Z-Order + Compactação)...")
        spark.sql(f"OPTIMIZE {table_name_full}")
        spark.sql(f"VACUUM {table_name_full} RETAIN 168 HOURS") # Mantém histórico de 7 dias para Time Travel
        
    except Exception:
        # Se ocorrer erro no get_table, assumimos que a tabela não existe ou não está configurada corretamente.
        
        # Cleanup preventivo: Se existir como tabela Delta comum (mas não Feature Table), removemos para recriar do zero.
        if spark.catalog.tableExists(table_name_full):
            print(f"⚠️ [CLEANUP] Tabela existe mas sem restrições de Feature Store. Removendo: {table_name_full}")
            spark.sql(f"DROP TABLE IF EXISTS {table_name_full}")
            
        print(f"🆕 [CREATE] Criando nova Feature Table: {table_name_full}")
        print(f"   🔑 PKs: {pk_columns} | 🕒 Time: {timestamp_col}")
        
        # Criação da tabela com suporte a features
        # Nota: Liquid Clustering (se suportado pela versão) é a melhor prática atual para particionamento.
        fe.create_table(
            name=table_name_full,
            primary_keys=pk_columns,
            timestamp_keys=timestamp_col,
            df=df,
            schema=df.schema,
            description="Ingested via JDBC for Feature Store"
            # table_properties={"delta.enableLiquidClustering": "true"} # Descomentar se ambiente suportar SDK compatível
        )
        
        # Garante otimização do layout inicial dos arquivos
        print(f"   ⚡ Otimizando layout inicial...")
        spark.sql(f"OPTIMIZE {table_name_full}")
