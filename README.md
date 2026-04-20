# IOVT-TCC

Deteccao de intrusoes em redes CAN para IoVT (Internet of Vehicles) usando Machine Learning.

## 1. Contexto do problema

Em IoVT, veiculos conectados trocam dados constantemente com ECUs, gateways e servicos externos. A rede CAN e critica para funcoes do veiculo, e ataques como DoS e spoofing podem afetar seguranca e operacao.

Este projeto implementa um IDS (Intrusion Detection System) baseado em Machine Learning para classificar trafego CAN em classes de comportamento normal e ataque.

## 2. Objetivo do projeto

- Construir um pipeline reproduzivel para treino e avaliacao de modelos.
- Comparar desempenho entre MLP, XGBoost e SVM.
- Registrar metricas e artefatos para analise tecnica no TCC.

## 3. Dados utilizados

### Fontes

- CARDt (dados CAN com ataques e trafego legitimo)
- CICIoV2024 (dataset publico de seguranca veicular)

Documentacao detalhada das fontes e estrutura:

- [docs/DATASETS.md](docs/DATASETS.md)
- [data/processed/metadata.md](data/processed/metadata.md)

### Pipeline de dados (resumo)

1. Alinhamento de colunas entre datasets.
2. Limpeza (drop de nulos/duplicados quando aplicavel).
3. Balanceamento para treinamento.
4. Split estratificado treino/teste.

Arquivo principal de treino:

- [data/processed/all_datasets_aligned_balanced.csv](data/processed/all_datasets_aligned_balanced.csv)

Observacao: os CSVs grandes nao sao versionados no GitHub para respeitar o limite de 100 MB por arquivo.

## 4. Metodologia

Implementacoes em scripts Python com GridSearchCV e validacao estratificada:

- [models/scripts/mlp_classifier.py](models/scripts/mlp_classifier.py)
- [models/scripts/xgboost_classifier.py](models/scripts/xgboost_classifier.py)
- [models/scripts/svm_classifier.py](models/scripts/svm_classifier.py)

Principais pontos:

- MLP e SVM com pipeline de escalonamento (StandardScaler).
- XGBoost com busca de hiperparametros para arvores boosted.
- Metricas salvas por modelo em JSON, relatorio de classificacao e matriz de confusao.

## 5. Comparacao de resultados

### 5.1 Resumo geral (teste)

Todos os resultados abaixo foram extraidos dos arquivos em [results/metrics](results/metrics).

| Modelo | Accuracy (CV) | Accuracy (Teste) | Acertos no teste | Erros no teste | Amostras de teste |
|---|---:|---:|---:|---:|---:|
| MLP | 0.988755 | 0.988452 | 684222 | 7994 | 692216 |
| XGBoost | 0.988986 | 0.989028 | 684621 | 7595 | 692216 |
| SVM | 0.985810 | 0.985659 | 682289 | 9927 | 692216 |

Leitura rapida:

- Melhor desempenho global: XGBoost (0.989028 no teste).
- Segundo lugar: MLP, com diferenca muito pequena para XGBoost.
- SVM teve o menor desempenho entre os tres, mas ainda com accuracy elevada.

### 5.2 Diferenca relativa entre os modelos

- XGBoost vs MLP: +0.000576 de accuracy (57.6 bps).
- XGBoost vs SVM: +0.003369 de accuracy (336.9 bps).
- MLP vs SVM: +0.002793 de accuracy (279.3 bps).

### 5.3 Relatorios de classificacao

Arquivos completos:

- [results/metrics/mlp/mlp_classification_report.txt](results/metrics/mlp/mlp_classification_report.txt)
- [results/metrics/xgboost/xgboost_classification_report.txt](results/metrics/xgboost/xgboost_classification_report.txt)
- [results/metrics/svm/svm_classification_report.txt](results/metrics/svm/svm_classification_report.txt)

Resumo observado nos tres modelos:

- Macro avg de precision/recall/F1 em torno de 0.99.
- Classes codificadas como 0-4 com desempenho alto em geral.
- Maior variacao aparece na classe 0 (recall inferior as demais), padrao comum tambem em MLP e SVM.

### 5.4 Matrizes de confusao

- [results/metrics/mlp/mlp_confusion_matrix.csv](results/metrics/mlp/mlp_confusion_matrix.csv)
- [results/metrics/xgboost/xgboost_confusion_matrix.csv](results/metrics/xgboost/xgboost_confusion_matrix.csv)
- [results/metrics/svm/svm_confusion_matrix.csv](results/metrics/svm/svm_confusion_matrix.csv)

## 6. Artefatos salvos

Modelos treinados:

- [results/models/mlp_best_model.joblib](results/models/mlp_best_model.joblib)
- [results/xgboost_best_model.joblib](results/xgboost_best_model.joblib)
- [results/models/svm_best_model.joblib](results/models/svm_best_model.joblib)

Summaries com hiperparametros e metricas:

- [results/metrics/mlp/mlp_summary.json](results/metrics/mlp/mlp_summary.json)
- [results/metrics/xgboost/xgboost_summary.json](results/metrics/xgboost/xgboost_summary.json)
- [results/metrics/svm/svm_summary.json](results/metrics/svm/svm_summary.json)

Visualizacao ROC:

- [results/visualizations/roc_curve_interativo.html](results/visualizations/roc_curve_interativo.html)

## 7. Como executar

### Requisitos

- Python 3.10+
- Dependencias em [requirements.txt](requirements.txt)

### Instalacao

```bash
pip install -r requirements.txt
```

### Treino dos modelos

```bash
python models/scripts/mlp_classifier.py
python models/scripts/xgboost_classifier.py
python models/scripts/svm_classifier.py
```

## 8. Estrutura do repositorio

- [docs](docs)
- [models/scripts](models/scripts)
- [notebooks](notebooks)
- [results](results)
- [data](data)

## 9. Conclusao

Os tres modelos apresentaram desempenho muito alto para deteccao de intrusao em dados CAN processados. No conjunto atual, XGBoost obteve o melhor resultado geral, seguido de perto por MLP. Isso indica que ambos sao candidatos fortes para um IDS em IoVT, com escolha final dependente de requisitos de latencia, custo computacional e interpretabilidade no ambiente de deploy.
