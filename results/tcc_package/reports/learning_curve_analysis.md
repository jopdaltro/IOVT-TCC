# Analise de Overfitting com Learning Curves

A versao atual adota learning curves com F1 macro para avaliar overfitting/underfitting sem uso de AUC/ROC.

## Como interpretar

- Overfitting: treino alto, validacao menor e gap persistente.
- Underfitting: treino e validacao baixos e proximos.
- Ajuste equilibrado: curvas proximas em nivel alto e com tendencia de estabilizacao.

## Execucao

Execute:

```bash
python models/scripts/build_learning_curves.py
```

Arquivos gerados:

- results/tcc_package/figures/learning_curves/*_learning_curve_f1_macro.png
- results/tcc_package/tables/*_learning_curve_f1_macro.csv
- results/tcc_package/tables/learning_curve_summary.json

## Base teorica

A analise segue a interpretacao classica de learning curves para identificar vies (underfitting) e variancia (overfitting), usando discrepancia treino-validacao ao longo do aumento do conjunto de treino.
