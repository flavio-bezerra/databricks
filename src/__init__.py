"""
Pacote principal do projeto de Previsão de Vendas (Databricks).

Este pacote contém todos os módulos necessários para o ciclo de vida do modelo de Machine Learning,
incluindo ingestão de dados, engenharia de recursos (Feature Store), validação de modelos
(treinamento e avaliação) e implantação (deploy).

Estrutura:
- ingestion: Conectores de banco de dados e funções para salvar no Feature Store.
- validation: Pipeline de treinamento, definição de modelos e configuração de experimentos.
- deploy: Wrappers para inferência e registro de modelos no MLflow/Unity Catalog.
"""
