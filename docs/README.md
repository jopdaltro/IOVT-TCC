# Documentacao do Projeto

## Arquivos principais

- docs/DATASETS.md: fontes e estrutura dos datasets
- docs/METHODOLOGY.md: metodologia atual de treino e avaliacao IDS
- docs/RESULTS.md: resultados e leitura operacional
- docs/RECOMMENDATIONS.md: recomendacoes de melhoria e proximos passos
- docs/TCC_WRITING_BASE.md: base para redacao do TCC (titulo, estrutura e texto guia)

## Scripts de modelo

- models/scripts/mlp_classifier.py
- models/scripts/xgboost_classifier.py
- models/scripts/svm_classifier.py
- models/scripts/logistic_regression_classifier.py
- models/scripts/evaluation_utils.py
- models/scripts/build_model_comparison.py
- models/scripts/build_learning_curves.py
- models/scripts/build_tcc_package.py

## Fluxo recomendado

1. Treinar os quatro modelos.
2. Executar o pacote final com `python models/scripts/build_tcc_package.py`.
3. Ler a comparacao em results/tcc_package/reports/model_comparison_report.md.
4. Ler a analise de overfitting em results/tcc_package/reports/learning_curve_analysis.md.
