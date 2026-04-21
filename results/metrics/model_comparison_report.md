# Comparativo de Modelos (IDS)

Criterio de ranking: Recall de ataque e FPR de ataque com maior peso, depois F1, Precision e eficiencia relativa.

| Rank | Modelo | Recall ataque | Precision ataque | FPR ataque | F1 macro | Velocidade treino | Velocidade inferencia | Score |
|---|---|---:|---:|---:|---:|---|---|---:|
| 1 | xgboost | 0.999978 | 0.984828 | 0.037915 | 0.990000 | fast | fast | 0.970564 |
| 2 | mlp | 0.999730 | 0.984280 | 0.039295 | 0.990000 | very_fast | medium | 0.965046 |
| 3 | svm | 0.998381 | 0.981774 | 0.045615 | 0.990000 | medium | slow | 0.922300 |
| 4 | logistic_regression | 0.942145 | 0.954001 | 0.111800 | 0.930706 | slow | very_fast | 0.891217 |