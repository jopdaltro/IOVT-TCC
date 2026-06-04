# ROADMAP (Atualizado)

## Fase 1 - Concluida

- Implementacao de MLP, XGBoost, SVM e Logistic Regression em scripts reproduziveis.
- Inclusao de metricas IDS no resumo por modelo.
- Inclusao de eficiencia de treino e inferencia.
- Script agregador de comparacao direta por metrica (sem score composto).

## Fase 2 - Em andamento

- Executar Logistic Regression e gerar artifacts em results/metrics/logistic_regression.
- Reexecutar os modelos para popular efficiency com tempos medidos em todos os summaries.
- Gerar learning curves para os 4 modelos e diagnostico de overfitting.
- Montar pacote final em results/tcc_package com graficos, tabelas e relatorios.

## Fase 3 - Producao

- Validacao de latencia no hardware alvo.
- Ajuste de threshold para reduzir falso positivo em cenario real.
- Testes de robustez cross-dataset e com ataques nao vistos.
