# Resultados dos Modelos de ML (IDS IoVT)

## Objetivo desta versao

A comparacao agora foca em criterio de IDS real:

- Recall de ataque
- Precision de ataque
- F1 macro
- False Positive Rate (FPR)
- Eficiencia relativa de treino e inferencia

## Modelos avaliados

- MLP
- XGBoost
- SVM
- Logistic Regression (script implementado; resultados dependem de execucao)

## Artefatos usados na comparacao

- results/metrics/mlp/mlp_summary.json
- results/metrics/xgboost/xgboost_summary.json
- results/metrics/svm/svm_summary.json
- results/metrics/logistic_regression/logistic_regression_summary.json
- results/metrics/model_comparison_summary.json
- results/metrics/model_comparison_report.md

## Ranking atual (com dados disponiveis)

Apos executar o agregador, o ranking atual esta em:

- results/metrics/model_comparison_report.md

Observacao: no estado atual, o ranking foi gerado com os summaries existentes. Para comparacao completa dos 4 modelos com eficiencia medida, execute tambem o treino da Logistic Regression.

## Definicoes de metrica

### FPR multiclasse

- One-vs-Rest por classe em fpr_per_class_ovr
- Media macro em fpr_macro_ovr

### Visao binaria IDS (ataque vs benign)

- attack_precision
- attack_recall
- attack_f1
- attack_fpr
- attack_fnr

## Eficiencia

Cada summary novo inclui bloco efficiency com:

- training_time_seconds
- inference_time_seconds
- inference_time_ms_per_sample
- inference_throughput_samples_per_second
- model_size_mb

Quando um summary antigo nao possui tempos medidos, o agregador usa prior relativa por tipo de algoritmo apenas para classificacao qualitativa de velocidade.

## Recomendacao operacional

- Para decisao de deploy, priorize ataque_recall alto e attack_fpr baixo.
- Use acuracia como metrica secundaria.
- Valide latencia no hardware alvo antes de producao.
