# Relatório Final – IDS IoVT (train_final.py)

| Modelo | Accuracy | F1 macro | F1 weighted | Attack Recall | Attack FPR | Treino (s) | Inferência (ms/amostra) |
|--------|---:|---:|---:|---:|---:|---:|---:|
| xgboost | 0.8771 | 0.8964 | 0.8790 | 1.0000 | 0.0000 | 129.8 | 0.0067 |
| two_stage | 0.8774 | 0.8967 | 0.8794 | 1.0000 | 0.0000 | 85.6 | 0.0479 |
| mlp | 0.6191 | 0.6742 | 0.6196 | 1.0000 | 0.0000 | 192.9 | 0.0047 |
| logistic_regression | 0.5139 | 0.5826 | 0.5134 | 1.0000 | 0.0003 | 41.5 | 0.0003 |
| svm | 0.5163 | 0.5809 | 0.5138 | 0.9987 | 0.0003 | 9.1 | 0.0003 |

## Melhor modelo multiclasse

Two-Stage combina detecção binária perfeita com discriminação entre tipos de ataque.
XGBoost direto com sample_weight balanceado é alternativa mais simples.
