# Narrativa Central do TCC (Dia 1)

## Tema
Deteccao de Intrusoes em Redes CAN de Sistemas Veiculares: uma abordagem baseada em Machine Learning.

## Titulo Provisorio (recomendado)
Deteccao de Intrusoes em Redes CAN de Sistemas Veiculares: Comparacao de Modelos Supervisionados para Classificacao Multiclasse.

## Problema de Pesquisa
Em redes CAN de sistemas veiculares conectados (IoVT), a deteccao de intrusoes exige modelos capazes de distinguir com alta confiabilidade o trafego legitimo de diferentes padroes de ataque. O problema desta pesquisa consiste em identificar qual modelo supervisionado apresenta melhor desempenho multiclasse no conjunto final alinhado e limpo, evitando conclusoes baseadas apenas em uma metrica isolada.

## Pergunta de Pesquisa
Qual dos modelos MLP, XGBoost, SVM e Regressao Logistica apresenta melhor desempenho para deteccao de intrusoes em rede CAN, quando comparado por accuracy, precision macro, recall macro, F1 macro e desempenho por classe no conjunto final limpo?

## Objetivo Geral
Comparar quatro modelos supervisionados (MLP, XGBoost, SVM e Regressao Logistica) para deteccao de intrusoes em rede CAN no contexto IoVT, utilizando o conjunto final alinhado e limpo e priorizando metricas multiclasse consistentes.

## Objetivos Especificos
1. Consolidar e preparar o conjunto de dados final alinhado entre Car-Hacking e CICIoV, com foco nas classes BENIGN, DoS, Fuzzy, GEAR e RPM.
2. Treinar e avaliar os modelos MLP, XGBoost, SVM e Regressao Logistica com procedimento reproduzivel.
3. Comparar os modelos por accuracy, precision macro, recall macro e F1 macro.
4. Analisar o desempenho por classe, destacando a capacidade de separacao entre trafego benigno e ataques.
5. Identificar a classe mais desafiadora para os modelos e discutir os principais padroes de erro.
6. Indicar o modelo com melhor compromisso global para classificacao multiclasse no cenario avaliado.

## Contribuicoes Declaradas
1. Proposta de comparacao multiclasse entre quatro modelos supervisionados aplicada a IDS em barramento CAN.
2. Consolidacao de uma base final limpa de resultados com relatorios de classificacao reutilizaveis para escrita academica.
3. Analise comparativa por metricas globais e por classe, evitando interpretacoes baseadas em uma unica medida.
4. Organizacao de material de suporte para escrita do TCC a partir de resultados experimentais e literatura auxiliar.

## Escopo Congelado para este TCC
- Modelos no corpo principal: MLP, XGBoost, SVM e Regressao Logistica.
- Dataset final canonico: carhacking_ciciov_benign_dos_aligned.csv.
- Classes analisadas: BENIGN, DoS, Fuzzy, GEAR e RPM.
- Metricas centrais: accuracy, precision macro, recall macro, F1 macro e relatorio por classe.
- Sem score composto no comparativo principal e sem Two-Stage no corpo principal.
- Estrutura do texto: 6 capitulos (template ABNT herdado).

## Frase-guia para abertura da Introducao
Este trabalho investiga a deteccao de intrusoes em redes CAN veiculares no contexto IoVT por meio da comparacao de modelos supervisionados capazes de classificar trafego benigno e diferentes padroes de ataque.

## Frase-guia para fechamento da Introducao
A contribuicao central desta pesquisa e uma comparacao direta entre quatro modelos supervisionados em um conjunto final limpo, com foco em desempenho multiclasse e interpretacao objetiva dos resultados por metrica e por classe.
