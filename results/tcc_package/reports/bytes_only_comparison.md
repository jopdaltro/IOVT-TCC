# Comparativo: Com ID vs Sem ID (bytes only)

| Modelo | Features | Accuracy | F1 macro | Attack Recall | Attack FPR |
|--------|----------|---:|---:|---:|---:|
| XGBoost | com id+ts+dlc | 0.8771 | 0.8964 | 1.0000 | 0.0000 |
| XGBoost | **bytes only** | 0.5957 | 0.5855 | 0.9272 | 0.0082 |
| Two-Stage | com id+ts+dlc | 0.8774 | 0.8967 | 1.0000 | 0.0000 |
| Two-Stage | **bytes only** | 0.5958 | 0.5858 | 0.9272 | 0.0082 |