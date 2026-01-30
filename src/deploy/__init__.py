"""
Módulo de Implantação (Deploy).

Este módulo contém classes e funções necessárias para empacotar o modelo final
e prepará-lo para inferência em produção (Batch ou Real-time).

Principais Componentes:
- wrapper.py: Contém o `UnifiedForecaster`, uma classe personalizada do MLflow (PythonModel)
  que encapsula o modelo treinado (Darts) e o pipeline de pré-processamento, garantindo
  que a inferência receba os dados crus e aplique as mesmas transformações do treino.
"""
