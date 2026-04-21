# Documentacao do Projeto

## Arquivos principais

- docs/DATASETS.md: fontes e estrutura dos datasets
- docs/METHODOLOGY.md: metodologia atual de treino e avaliacao IDS
- docs/RESULTS.md: resultados e leitura operacional
- docs/RECOMMENDATIONS.md: recomendacoes de melhoria e proximos passos

## Scripts de modelo

- models/scripts/mlp_classifier.py
- models/scripts/xgboost_classifier.py
- models/scripts/svm_classifier.py
- models/scripts/logistic_regression_classifier.py
- models/scripts/evaluation_utils.py
- models/scripts/build_model_comparison.py

## Fluxo recomendado

1. Treinar os quatro modelos.
2. Executar o agregador de comparacao.
3. Ler o ranking final em results/metrics/model_comparison_report.md.
