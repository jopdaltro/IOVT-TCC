# Comparativo: Com ID vs Sem ID (bytes only)

| Modelo | Features | Accuracy | F1 macro | Attack Recall | Attack FPR |
|--------|----------|---:|---:|---:|---:|
| XGBoost | com id+ts+dlc | 0.8771 | 0.8964 | 1.0000 | 0.0000 |
| XGBoost | **bytes only** | 0.6059 | 0.6529 | 0.9290 | 0.0075 |
| Two-Stage | com id+ts+dlc | 0.8774 | 0.8967 | 1.0000 | 0.0000 |
| Two-Stage | **bytes only** | 0.6061 | 0.6532 | 0.9290 | 0.0075 |