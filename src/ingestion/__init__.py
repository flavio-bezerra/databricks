"""
Módulo de Ingestão de Dados.

Este módulo é responsável por conectar a fontes de dados externas (como Azure SQL Database)
e gerenciar a persistência de dados no Databricks Feature Store.

Principais Componentes:
- connectors.py: Funções utilitárias para estabelecer conexões JDBC seguras.
- feature_store.py: Lógica para salvar e otimizar tabelas delta no Feature Store,
  aplicando melhores práticas como Liquid Clustering e limpeza de dados.
"""
