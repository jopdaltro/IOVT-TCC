# Metodologia (IDS orientado a operacao real)

## Pipeline

1. Carregamento do dataset balanceado.
2. Split estratificado treino/teste.
3. GridSearchCV por modelo.
4. Predicao no conjunto de teste.
5. Calculo de metricas de IDS + eficiencia.
6. Persistencia de artefatos em results/.

## Modelos

- MLP: models/scripts/mlp_classifier.py
- XGBoost: models/scripts/xgboost_classifier.py
- SVM: models/scripts/svm_classifier.py
- Logistic Regression: models/scripts/logistic_regression_classifier.py

## Avaliacao

Metricas salvas por modelo:

- test_accuracy
- precision_macro, recall_macro, f1_macro
- precision_weighted, recall_weighted, f1_weighted
- fpr_per_class_ovr, fpr_macro_ovr
- attack_detection.* (visao binaria IDS)
- efficiency.* (tempo e throughput)

Implementacao compartilhada de metricas:

- models/scripts/evaluation_utils.py

## Comparacao e ranking

Agregador:

- models/scripts/build_model_comparison.py

Saidas:

- results/metrics/model_comparison_summary.json
- results/metrics/model_comparison_report.md

Score composto para decisao real:

- 35% attack_recall
- 30% (1 - attack_fpr)
- 15% f1_macro
- 10% attack_precision
- 6% eficiencia de inferencia
- 4% eficiencia de treino

## Observacao de reproducibilidade

A comparacao de eficiencia e relativa entre modelos. Para comparar ambientes diferentes, repita os treinos no mesmo hardware e mesma configuracao de paralelismo.
