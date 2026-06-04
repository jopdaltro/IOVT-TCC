# Base de Escrita do TCC (Sem Score Composto e Sem AUC/ROC)

## 1. Sugestoes de titulo

1. Deteccao de Intrusoes em Redes CAN para IoVT com Comparacao Direta de Modelos Supervisionados
2. Analise Comparativa de MLP, XGBoost, SVM e Regressao Logistica para IDS em Barramento CAN
3. IDS para IoVT em Rede CAN: Avaliacao por Recall, FPR, F1 e Eficiencia Operacional

## 2. Problema e justificativa

Use este texto-base:

"Em ambientes IoVT, falhas de deteccao e alarmes falsos em IDS podem comprometer seguranca e operacao. Por isso, o problema desta pesquisa nao e apenas atingir alta acuracia global, mas equilibrar deteccao de ataques, baixa taxa de falso positivo e viabilidade computacional para uso real."

## 3. Objetivo geral

"Comparar quatro modelos de classificacao (MLP, XGBoost, SVM e Regressao Logistica) para IDS em rede CAN, com foco em metricas operacionais interpretaveis e analise de overfitting por learning curves."

## 4. Objetivos especificos

- Implementar pipeline reproduzivel com split estratificado e ajuste de hiperparametros.
- Avaliar modelos por recall de ataque, precision de ataque, F1 macro e FPR de ataque.
- Comparar eficiencia por tempo de treino e inferencia por amostra.
- Diagnosticar comportamento de ajuste (overfitting/underfitting) com learning curves.
- Consolidar graficos e tabelas em uma pasta unica para analise e escrita.

## 5. Estrutura recomendada de capitulos

1. Introducao
2. Fundamentacao teorica
3. Metodologia
4. Resultados e discussao
5. Conclusoes e trabalhos futuros

## 6. O que escrever em cada capitulo

## Introducao

- Contexto de IoVT e rede CAN.
- Riscos de ataque no barramento CAN.
- Limites de usar apenas acuracia como criterio.
- Pergunta de pesquisa.
- Objetivos e organizacao do texto.

## Fundamentacao teorica

- Rede CAN e tipos de ataque (DoS, spoofing etc.).
- IDS supervisionado para classificacao multiclasses.
- Metricas usadas:
  - Recall de ataque (sensibilidade para ataques)
  - Precision de ataque (qualidade dos alertas)
  - FPR de ataque (alarme falso em benign)
  - F1 macro (equilibrio entre classes)
  - Tempo de treino e inferencia (viabilidade)
- Learning curves para diagnostico de ajuste:
  - Overfitting: treino alto e validacao baixa, gap grande.
  - Underfitting: treino e validacao baixos e proximos.
  - Ajuste equilibrado: curvas proximas com boa performance.

## Metodologia

- Origem dos dados e preprocessamento.
- Split treino/teste estratificado.
- Configuracao dos 4 modelos.
- Procedimento de comparacao por metrica (sem score agregado).
- Geracao de learning curves com F1 macro.
- Organizacao dos artefatos em results/tcc_package.

## Resultados e discussao

- Tabela comparativa por metrica principal.
- Melhor modelo por criterio (ex.: menor FPR, maior recall, menor latencia).
- Interpretacao das learning curves por modelo.
- Trade-offs encontrados entre qualidade e eficiencia.
- Riscos de generalizacao e limites do estudo.

## Conclusoes e trabalhos futuros

- Sintese dos principais achados por metrica.
- Escolha do modelo recomendada por cenario de uso.
- Limites (dataset, ambiente, ausencia de validacao em hardware final).
- Futuro: validacao cross-dataset, testes com ataques nao vistos, ajuste de threshold.

## 7. Pergunta de pesquisa (modelo pronto)

"Qual dos modelos MLP, XGBoost, SVM e Regressao Logistica oferece melhor desempenho para IDS em rede CAN no contexto IoVT quando comparado por recall de ataque, FPR de ataque, F1 macro e eficiencia computacional, e qual o comportamento de ajuste identificado por learning curves?"

## 8. Hipoteses (opcional)

- H1: Modelos nao lineares (MLP, XGBoost, SVM-RBF) tendem a maior recall de ataque que Regressao Logistica.
- H2: Existe trade-off entre menor FPR e menor tempo de inferencia.
- H3: Learning curves permitem identificar diferencas de overfitting entre os modelos mesmo quando metricas finais sao proximas.

## 9. Como citar a ideia de learning curve na escrita

Texto-base sugerido:

"A analise de learning curves foi utilizada para apoiar o diagnostico de overfitting e underfitting. Seguindo abordagem amplamente difundida na literatura aplicada, observou-se a evolucao do desempenho de treino e validacao em funcao do tamanho do conjunto de treino. Gaps persistentes entre treino e validacao foram interpretados como indicio de alta variancia (overfitting), enquanto desempenho baixo e convergente em ambas as curvas foi interpretado como alta tendencia a vies (underfitting)."

## 10. Checklist final para capitulo de resultados

- Inserir tabela de comparacao sem score composto.
- Inserir learning curve de cada modelo.
- Explicar cada grafico em 1 paragrafo objetivo.
- Declarar claramente qual modelo vence em cada metrica.
- Justificar recomendacao final por cenario de aplicacao.
