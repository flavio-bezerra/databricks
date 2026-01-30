"""
Módulo de Validação e Treinamento.

Este é o núcleo do pipeline de Machine Learning. Contém a lógica para:
1. Configuração do ambiente (Config).
2. Preparação de dados para séries temporais (DataIngestion).
3. Definição do pipeline de transformação (Scalers, Encoders).
4. Treinamento e Validação Walk-Forward (ModelTrainer).
5. Definição de Wrappers para compatibilidade com MLflow (DartsWrapper).

Fluxo Típico:
Config -> DataIngestion -> ProjectPipeline -> ModelTrainer -> MLflow
"""
