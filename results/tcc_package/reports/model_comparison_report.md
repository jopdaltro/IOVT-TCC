# Comparativo de Modelos IDS (Sem Score Composto)

Este relatorio apresenta comparacao direta por metrica, sem score composto e sem AUC/ROC.

| Modelo | Accuracy teste | Recall ataque | Precision ataque | FPR ataque | F1 macro | Tempo treino (s) | Inferencia (ms/amostra) |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.381167 | 0.992755 | 0.998223 | 0.029326 | 0.467762 | 0.347067 | 0.000264 |
| mlp | 0.633757 | 1.000000 | 1.000000 | 0.000000 | 0.671468 | 279.431139 | 0.002130 |
| svm | 0.408000 | 0.995936 | 1.000000 | 0.000000 | 0.558754 | 20.782388 | 0.318360 |
| xgboost | 0.876986 | 1.000000 | 1.000000 | 0.000000 | 0.897139 | 129.678387 | 0.005529 |

## Destaques por metrica

- Maior recall de ataque: mlp (1.000000)
- Menor FPR de ataque: mlp (0.000000)
- Maior F1 macro: xgboost (0.897139)
- Menor tempo de inferencia: logistic_regression (0.000264 ms/amostra)
- Menor tempo de treino: logistic_regression (0.347067 s)