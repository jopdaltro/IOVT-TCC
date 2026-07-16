# Base Consolidada de Escrita (Dias 1-2)

Este documento consolida a base textual e numerica para redacao do novo TCC em LATEX/Monografia_Joao_Pedro_2.

Observacao importante: os numeros canonicos desta versao passam a vir dos diretorios `results/metrics/*_final_clean`, que substituem o comparativo antigo usado na primeira versao deste guia.

## 1) Fontes consolidadas (fonte de verdade)

### Base textual
1. IOVT-TCC/docs/TCC_WRITING_BASE.md
2. IOVT-TCC/docs/METHODOLOGY.md
3. IOVT-TCC/docs/RESULTS.md
4. IOVT-TCC/docs/DATASETS.md
5. Miniartigo: IOVT-TCC/conference_latex_template_10_17_19__1_.pdf
6. Literatura auxiliar: IOVT-TCC/artigos auxiliares/

### Base de resultados
1. IOVT-TCC/results/metrics/logistic_regression_final_clean/run_summary.json
2. IOVT-TCC/results/metrics/logistic_regression_final_clean/classification_report.txt
3. IOVT-TCC/results/metrics/mlp_final_clean/run_summary.json
4. IOVT-TCC/results/metrics/mlp_final_clean/classification_report.txt
5. IOVT-TCC/results/metrics/svm_final_clean/run_summary.json
6. IOVT-TCC/results/metrics/svm_final_clean/classification_report.txt
7. IOVT-TCC/results/metrics/xgboost_final_clean/run_summary.json
8. IOVT-TCC/results/metrics/xgboost_final_clean/classification_report.txt
9. IOVT-TCC/results/tcc_package/reports/learning_curve_analysis.md

## 2) Regra de consistencia para numeros

Para evitar contradicoes no texto final:
1. Comparativo principal (4 modelos) usa os valores dos arquivos `classification_report.txt` e `run_summary.json` dos diretorios `*_final_clean`.
2. `run_summary.json` deve ser usado para contexto experimental, tamanho de dados e acuracia final registrada pelo script.
3. `classification_report.txt` deve ser usado para accuracy do teste, precision, recall e F1 por classe, alem de medias macro e ponderada.
4. Relatorios antigos em `results/tcc_package/reports/` passam a ser apenas complemento interpretativo, nao fonte canonica de resultados finais.

## 3) Numeros canonicamente adotados (comparativo principal)

Fonte: results/metrics/*_final_clean/classification_report.txt e run_summary.json

## Dataset final de referencia

- Arquivo: `data/processed/carhacking_ciciov_benign_dos_aligned.csv`
- Classes finais: BENIGN, DoS, Fuzzy, GEAR e RPM
- Filtro aplicado: remocao de 200665 amostras CARDt com `DLC < 8`

- logistic_regression
  - Train shape: 200000 x 9
  - Test shape: 80000 x 9
  - Accuracy: 0.97252
  - Precision macro: 0.97587
  - Recall macro: 0.96940
  - F1 macro: 0.97235
  - Pior classe por recall: Fuzzy (0.89292)

- mlp
  - Train shape: 2724943 x 9
  - Test shape: 681236 x 9
  - Accuracy: 0.99908
  - Precision macro: 0.99937
  - Recall macro: 0.99873
  - F1 macro: 0.99905
  - Pior classe por recall: Fuzzy (0.99368)

- svm
  - Train shape: 150000 x 9
  - Test shape: 60000 x 9
  - Accuracy: 0.94992
  - Precision macro: 0.94689
  - Recall macro: 0.95387
  - F1 macro: 0.95000
  - Pior classe por recall: Fuzzy (0.88677)

- xgboost
  - Train shape: 2724943 x 9
  - Test shape: 681236 x 9
  - Accuracy: 0.99998
  - Precision macro: 0.99999
  - Recall macro: 0.99998
  - F1 macro: 0.99998
  - Pior classe por recall: Fuzzy (0.99990)

## 4) Sintese analitica pronta para escrita

1. Melhor desempenho global: XGBoost, com accuracy de 0.99998 e F1 macro de 0.99998.
2. Segundo melhor desempenho global: MLP, tambem com resultado muito elevado e leve perda relativa na classe Fuzzy.
3. Logistic Regression apresentou desempenho forte e consistente, mas abaixo de MLP e XGBoost sobretudo na classe Fuzzy.
4. SVM teve desempenho global inferior aos dois melhores modelos e mostrou maior dificuldade relativa para classificar Fuzzy e trafego benigno.
5. A classe Fuzzy foi a mais desafiadora entre os quatro modelos, sendo um bom ponto de discussao em resultados e conclusao.
6. As classes GEAR e RPM apresentaram separacao quase perfeita na maior parte dos modelos.

## 5) Blocos de texto reutilizaveis por capitulo

### Capitulo 1 (Introducao)
Problema central: a deteccao de intrusoes em barramento CAN exige classificacao confiavel de trafego benigno e diferentes tipos de ataque, com interpretacao cuidadosa das metricas globais e por classe.

### Capitulo 2 (Fundamentacao)
Explicar CAN, ataques (DoS, Fuzzy, GEAR e RPM), IDS supervisionado multiclasse e definicao das metricas centrais (accuracy, precision macro, recall macro, F1 macro e desempenho por classe).

### Capitulo 3 (Relacionados)
Comparar estudos por: dataset, tipo de ataque, algoritmo, metricas e foco experimental. A pasta `artigos auxiliares/` deve apoiar o enquadramento de benchmarking, arquiteturas de ML e formulacao dos objetivos.

### Capitulo 4 (Metodologia)
Pipeline reproducivel: dados alinhados e limpos, remocao de amostras com `DLC < 8`, split treino/teste, treino dos 4 modelos e geracao de relatorios de classificacao por modelo.

### Capitulo 5 (Resultados)
Comparacao direta por metrica, tabela principal com 4 modelos e discussao das diferencas de desempenho global e por classe, com destaque para a classe Fuzzy.

### Capitulo 6 (Conclusoes)
Resposta objetiva a pergunta de pesquisa: XGBoost foi o modelo com melhor desempenho multiclasse no conjunto final limpo, seguido de perto pelo MLP.

## 6) Learning curves: texto-base pronto

A analise de learning curves pode ser empregada como apoio interpretativo para verificar sinais de overfitting e underfitting. Entretanto, os resultados finais canonicos desta versao devem ser citados a partir dos relatorios `*_final_clean`, que concentram a base oficial de comparacao entre modelos.

## 7) Checklist de uso durante a escrita

1. Antes de escrever cada secao, verificar se as metricas citadas pertencem ao relatorio canonico.
2. Nao misturar resultados antigos de `tcc_package/reports/` com os numeros finais limpos.
3. Usar a literatura de `artigos auxiliares/` para justificar objetivos, comparacoes e relevancia do problema, nunca como fonte de resultado experimental proprio.
4. Manter coerencia terminologica: IDS, IoVT, CAN, accuracy, precision macro, recall macro e F1 macro.
5. Ao final de cada capitulo, validar se o texto responde ao objetivo geral e aos objetivos especificos.
