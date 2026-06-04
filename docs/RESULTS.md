# Resultados dos Modelos de ML (IDS IoVT)

## Objetivo desta versao

A comparacao agora foca em criterio de IDS real sem score composto:

- Recall de ataque
- Precision de ataque
- F1 macro
- False Positive Rate (FPR)
- Eficiencia de treino e inferencia

## Modelos avaliados

- MLP
- XGBoost
- SVM
- Logistic Regression (script implementado; resultados dependem de execucao)

## Artefatos usados na comparacao

- results/metrics/mlp/mlp_summary.json
- results/metrics/xgboost/xgboost_summary.json
- results/metrics/svm/svm_summary.json
- results/metrics/logistic_regression/logistic_regression_summary.json
- results/tcc_package/tables/model_comparison_no_score.json
- results/tcc_package/reports/model_comparison_report.md
- results/tcc_package/reports/learning_curve_analysis.md

## Comparacao atual (com dados disponiveis)

Apos executar o pacote de resultados, a comparacao atual esta em:

- results/tcc_package/reports/model_comparison_report.md

Observacao: para comparacao completa dos 4 modelos com eficiencia medida, execute os quatro scripts de treino antes do pacote final.

## Definicoes de metrica

### FPR multiclasse

- One-vs-Rest por classe em fpr_per_class_ovr
- Media macro em fpr_macro_ovr

### Visao binaria IDS (ataque vs benign)

- attack_precision
- attack_recall
- attack_f1
- attack_fpr
- attack_fnr

## Eficiencia

Cada summary novo inclui bloco efficiency com:

- training_time_seconds
- inference_time_seconds
- inference_time_ms_per_sample
- inference_throughput_samples_per_second
- model_size_mb

Quando um summary nao possui tempos medidos, a metrica fica como n/a em vez de usar estimativa por prior.

## Recomendacao operacional

- Para decisao de deploy, priorize ataque_recall alto e attack_fpr baixo.
- Use acuracia como metrica secundaria.
- Valide latencia no hardware alvo antes de producao.

## Analise de overfitting (sem AUC/ROC)

- Learning curves sao geradas com F1 macro em treino e validacao.
- O comportamento de overfitting/underfitting e descrito em linguagem direta no relatorio.
- Arquivos:
	- results/tcc_package/figures/learning_curves
	- results/tcc_package/reports/learning_curve_analysis.md
