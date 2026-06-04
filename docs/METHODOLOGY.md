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

## Comparacao direta por metrica

Agregador:

- models/scripts/build_model_comparison.py

Saidas:

- results/tcc_package/tables/model_comparison_no_score.json
- results/tcc_package/reports/model_comparison_report.md

Metricas usadas para comparacao:

- attack_recall
- attack_precision
- attack_fpr
- f1_macro
- training_time_seconds
- inference_time_ms_per_sample

## Analise de overfitting

Script:

- models/scripts/build_learning_curves.py

Saidas:

- results/tcc_package/figures/learning_curves/*.png
- results/tcc_package/tables/*_learning_curve_f1_macro.csv
- results/tcc_package/reports/learning_curve_analysis.md

Abordagem teorica aplicada:

- Curva de aprendizado com treino e validacao em funcao do tamanho de treino.
- Overfitting: treino alto, validacao menor e gap persistente.
- Underfitting: treino e validacao baixos e proximos.
- Ajuste equilibrado: curvas proximas em nivel alto e tendencia de estabilizacao.

## Observacao de reproducibilidade

A comparacao de eficiencia e relativa entre modelos. Para comparar ambientes diferentes, repita os treinos no mesmo hardware e mesma configuracao de paralelismo.
