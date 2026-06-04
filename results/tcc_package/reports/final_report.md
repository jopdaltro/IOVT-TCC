# Relatório Final – IDS IoVT (train_final.py v2)

| Modelo | Accuracy | F1 macro | Attack Recall | Attack FPR | Treino (s) | Inf. (ms/smp) |
|--------|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.5139 | 0.5826 | 1.0000 | 0.0003 | 60.1 | 0.0004 |
| svm | 0.5163 | 0.5809 | 0.9987 | 0.0003 | 13.9 | 0.0004 |
| xgboost | 0.8771 | 0.8964 | 1.0000 | 0.0000 | 336.1 | 0.0395 |
| two_stage | 0.8774 | 0.8967 | 1.0000 | 0.0000 | 241.4 | 0.1700 |
| mlp | 0.6191 | 0.6742 | 1.0000 | 0.0000 | 1247.4 | 0.0116 |