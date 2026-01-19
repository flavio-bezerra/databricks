# 📈 CVC Lojas - Forecasting MLOps Pipeline

Este projeto implementa um pipeline de **MLOps ponta a ponta** para previsão de vendas das lojas físicas da CVC. A arquitetura utiliza **Databricks**, **Unity Catalog**, **Feature Store**, e modelos de Séries Temporais (**Darts**) orquestrados via **MLflow**.



## 🏗️ Arquitetura e Estrutura

O projeto adota uma estrutura modular, separando a lógica de negócio (pacote `src`) da execução (Notebooks).

```text
databricks/
├── src/                            # 📦 Core Package (Lógica Modularizada)
│   ├── ingestion/                  # Conectores JDBC & Feature Store (Liquid Clustering)
│   ├── validation/                 # Pipelines de Treino, Walk-Forward & Configs
│   └── deploy/                     # Wrapper "All-in-One" para Inferência Produtiva
│
├── 1_ingestao_features.ipynb       # ETL: SQL Server -> Databricks Feature Store
├── 2_validacao_modelos.ipynb       # Experimentação: Backtesting (Walk-Forward)
├── 3_treino_final_deploy.ipynb     # Deploy: Treino Final (2021-2025) -> Unity Catalog
└── 4_inferencia_recorrente.ipynb   # Produção: Scoring Semanal/Mensal
```

---

## 🚀 Fluxos de Trabalho (Workflows)

### 1. Ingestão de Dados (`src.ingestion`)
Responsável por trazer dados transacionais do Azure SQL para o **Feature Store** no Unity Catalog.
* **Destaques:** Utiliza *Liquid Clustering* e remoção de duplicatas baseada em PKs para garantir qualidade na entrada.
* **Artefato:** Tabelas Delta otimizadas em `ds_dev.cvc_val.*`.

### 2. Validação de Modelos (`src.validation`)
Executa uma validação rigorosa para escolher o melhor algoritmo.
* **Metodologia:** *Strict Walk-Forward Validation*. O modelo é treinado e testado mês a mês no passado, sem vazamento de dados futuros.
* **Modelos Avaliados:** LightGBM, XGBoost, TFT (Temporal Fusion Transformer), N-BEATS.
* **Segurança:** Utiliza `OrdinalEncoder` com tratamento para categorias desconhecidas (novas lojas).

### 3. Treino e Deploy (`src.deploy`)
Treina a versão final do modelo com dados recentes (Pós-Pandemia: 2021-2025) para evitar *Concept Drift*.
* **Wrapper "UnifiedForecaster":** O modelo é encapsulado em uma classe Python customizada que contém:
    * O modelo treinado (ex: LightGBM).
    * O pipeline de transformação (`Scalers`, `Encoders`).
    * Lógica automática de geração de datas futuras e feriados.
* **Registro:** O modelo é salvo no Unity Catalog e promovido via Alias (`@Champion`).

### 4. Inferência Recorrente
Pipeline agendado que consome o modelo `@Champion`.
* **Resiliência:** O sistema detecta automaticamente se precisa gerar o esqueleto de datas futuras (Forecast Horizon) ou se ele já foi fornecido.
* **Fallback:** Em caso de falha crítica, retorna um schema vazio válido para não quebrar jobs Spark dependentes.

---

## 🛠️ Tecnologias e Bibliotecas

* **Plataforma:** Databricks (Runtime ML)
* **Governança:** Unity Catalog (Features & Models)
* **Frameworks:**
    * `Darts` (Time Series)
    * `PySpark` & `Delta Lake`
    * `MLflow` (Tracking & Registry)
    * `Scikit-Learn` (Pipelines)

## 📋 Como Executar

### Pré-requisitos
Certifique-se de que a pasta `src` esteja no diretório de trabalho ou instalada como biblioteca.

### Passo a Passo
1.  **Ingestão:** Execute `cvc_ingestao_features_validacao.ipynb` para atualizar as tabelas do Feature Store.
2.  **Validação (Opcional):** Execute `cvc_validacao_modelos_lojas.ipynb` se desejar testar novas arquiteturas de modelo.
3.  **Deploy:** Execute `cvc_treino_final_deploy.ipynb`. Este notebook irá:
    * Treinar o modelo com dados até `2025-12-31`.
    * Registrar o modelo no Unity Catalog.
    * Atribuir a tag **@Champion** à nova versão.
4.  **Inferência:** Execute `cvc_inferencia_recorrente.ipynb`. Ele carregará automaticamente a versão `@Champion` e salvará as previsões na tabela de resultados.

---

## 🛡️ Robustez e Tratamento de Erros

* **Safe ID Extraction:** O sistema blinda os IDs das lojas (`CODIGO_LOJA`) para evitar que transformações numéricas corrompam identificadores (ex: "Loja 001" virar "1.0").
* **Future Skeleton:** O Wrapper é capaz de autocompletar datas futuras caso o input contenha apenas dados históricos.
* **Schema Enforcement:** Retornos de erro padronizados garantem que o Spark não falhe por incompatibilidade de tipos.