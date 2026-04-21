# IOVT-TCC

Deteccao de intrusoes em redes CAN para IoVT (Internet of Vehicles) com foco em desempenho de IDS real: qualidade de deteccao e eficiencia operacional.

## 1. Contexto

Em IoVT, a rede CAN conecta ECUs criticas. Em ambiente real, nao basta alta acuracia: o IDS precisa manter baixa taxa de falso positivo e responder rapido para nao degradar operacao.

## 2. Objetivo

- Treinar e comparar quatro modelos: MLP, XGBoost, SVM e Logistic Regression.
- Avaliar nao apenas acuracia, mas metricas centrais para IDS.
- Gerar um ranking final para deploy considerando qualidade + eficiencia.

## 3. Dados

Fontes:

- CARDt
- CICIoV2024

Documentacao:

- [docs/DATASETS.md](docs/DATASETS.md)
- [data/processed/metadata.md](data/processed/metadata.md)

Arquivo principal de treino:

- [data/processed/all_datasets_aligned_balanced.csv](data/processed/all_datasets_aligned_balanced.csv)

## 4. Modelos e scripts

- [models/scripts/mlp_classifier.py](models/scripts/mlp_classifier.py)
- [models/scripts/xgboost_classifier.py](models/scripts/xgboost_classifier.py)
- [models/scripts/svm_classifier.py](models/scripts/svm_classifier.py)
- [models/scripts/logistic_regression_classifier.py](models/scripts/logistic_regression_classifier.py)

Todos os scripts usam split estratificado e GridSearchCV, salvando resultados em [results/metrics](results/metrics) e modelos em [results/models](results/models).

## 5. Metricas de avaliacao IDS

As metricas principais desta versao sao:

- Recall (detectar ataques)
- Precision (evitar falso positivo)
- F1-score
- False Positive Rate (FPR)

Definicoes aplicadas:

- FPR multiclasse: One-vs-Rest (por classe) e media macro em `fpr_macro_ovr`.
- Visao IDS binaria (ataque vs benign): `attack_precision`, `attack_recall`, `attack_f1`, `attack_fpr`, `attack_fnr`.

## 6. Eficiencia (velocidade)

Cada summary passa a incluir:

- `training_time_seconds`
- `inference_time_seconds`
- `inference_time_ms_per_sample`
- `inference_throughput_samples_per_second`
- `model_size_mb`

Esses campos ficam no bloco `efficiency` dos arquivos `*_summary.json`.

## 7. Comparacao e ranking para uso real

Script agregador:

- [models/scripts/build_model_comparison.py](models/scripts/build_model_comparison.py)

Saidas:

- [results/metrics/model_comparison_summary.json](results/metrics/model_comparison_summary.json)
- [results/metrics/model_comparison_report.md](results/metrics/model_comparison_report.md)

Criterio de score composto (deploy IDS):

- 35% recall de ataque
- 30% baixo FPR de ataque
- 15% F1 macro
- 10% precision de ataque
- 6% velocidade relativa de inferencia
- 4% velocidade relativa de treino

## 8. Como executar

### Requisitos

- Python 3.10+
- Dependencias em [requirements.txt](requirements.txt)

### Instalacao

```bash
pip install -r requirements.txt
```

### Treinar os 4 modelos

```bash
python models/scripts/mlp_classifier.py
python models/scripts/xgboost_classifier.py
python models/scripts/svm_classifier.py
python models/scripts/logistic_regression_classifier.py
```

### Gerar comparativo final

```bash
python models/scripts/build_model_comparison.py
```

## 9. Artefatos

Summaries:

- [results/metrics/mlp/mlp_summary.json](results/metrics/mlp/mlp_summary.json)
- [results/metrics/xgboost/xgboost_summary.json](results/metrics/xgboost/xgboost_summary.json)
- [results/metrics/svm/svm_summary.json](results/metrics/svm/svm_summary.json)
- [results/metrics/logistic_regression/logistic_regression_summary.json](results/metrics/logistic_regression/logistic_regression_summary.json)

Modelos:

- [results/models/mlp_best_model.joblib](results/models/mlp_best_model.joblib)
- [results/models/xgboost_best_model.joblib](results/models/xgboost_best_model.joblib)
- [results/models/svm_best_model.joblib](results/models/svm_best_model.joblib)
- [results/models/logistic_regression_best_model.joblib](results/models/logistic_regression_best_model.joblib)

## 10. Conclusao

O projeto agora esta estruturado para responder a pergunta correta de IDS em IoVT: qual modelo entrega melhor equilibrio entre deteccao de ataques, baixo falso positivo e eficiencia operacional. O ranking final de producao deve ser lido em [results/metrics/model_comparison_report.md](results/metrics/model_comparison_report.md) apos executar os quatro treinamentos.
