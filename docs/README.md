# IOVT IDS - Intrusion Detection System para Redes Veiculares

## 📋 Visão Geral

Este projeto implementa e compara **múltiplos algoritmos de Machine Learning** para detecção de intrusões em redes CAN (Controller Area Network) de veículos conectados. O objetivo é identificar ataques de **spoofing** e **Denial of Service (DoS)** em dados telemetricos de sensores automotivos.

## 🎯 Objetivos da Pesquisa

- Avaliar efetividade de diferentes algoritmos de ML (MLP, XGBoost, SVM) em detecção de intrusões veiculares
- Comparar performance entre dois datasets independentes (CARDt e CICIoV2024)
- Identificar melhores práticas de preprocessamento e validação para dados de CAN bus
- Fornecer benchmark comparativo para futuros trabalhos em IoV Security

## 📁 Estrutura do Projeto

```
├── data/                          # Dados brutos e processados
│   ├── raw/                       # Datasets originais
│   │   ├── CARDt/                 # Dataset CARDt (~5 classes)
│   │   └── CICIoV2024/            # Dataset CICIoV2024 público
│   └── processed/                 # Dados combinados e balanceados
│       ├── all_datasets_aligned.csv
│       └── all_datasets_aligned_balanced.csv
│
├── notebooks/                     # Análises e treinamento
│   ├── 02_mlp_classifier.ipynb    # Rede Neural Artificial (MLP)
│   └── 03_xgboost_classifier.ipynb # XGBoost com GridSearch
│
├── models/                        # Scripts e modelos treinados
│   ├── scripts/
│   │   └── svm_classifier.py      # SVM (a implementar)
│   └── saved_models/              # Modelos treinados (.pkl, .joblib)
│
├── results/                       # Resultados e visualizações
│   ├── metrics/                   # Matrizes de confusão, relatórios
│   └── visualizations/            # Gráficos, ROC curves, HTML
│
└── docs/                          # Documentação do projeto
    ├── README.md (este arquivo)
    ├── DATASETS.md
    ├── METHODOLOGY.md
    └── RESULTS.md
```

## 🔍 Tipos de Ataques Detectados

### CARDt Dataset
- **DoS (Denial of Service)**: Sobrecarregar a rede CAN
- **Fuzzy**: Random bit-flipping em mensagens CAN
- **RPM Spoofing**: Altera dados simulados de rotações do motor
- **Gear Spoofing**: Altera dados da marcha selecionada
- **Benign (Normal)**: Operação legítima

### CICIoV2024 Dataset
- **Gas Pedal Spoofing**: Falsificação de dados do pedal de acelerador
- **RPM Spoofing**: Dados falsos de rotações do motor
- **Speed Spoofing**: Velocidade do veículo falsificada
- **Steering Wheel Spoofing**: Ângulo de direção falsificado
- **DoS Attack**: Flooding da rede CAN
- **Benign (Normal)**: Operação legítima

## 🤖 Modelos Implementados

| Modelo | Notebook | Status | AUC Reportado |
|--------|----------|--------|---------------|
| **MLP** (Redes Neurais) | 02_mlp_classifier.ipynb | ✅ Completo | 99.78% |
| **XGBoost** | 03_xgboost_classifier.ipynb | 🔄 GridSearch Pendente | TBD |
| **SVM** | models/scripts/svm_classifier.py | 🚧 A Implementar | TBD |

## 📊 Resultados Preliminares

### MLP Neural Network
- **Macro-average AUC**: 99.78%
- **Micro-average AUC**: 99.95%
- **Fuzzy Classification**: 100% AUC
- **RPM Classification**: 100% AUC
- **Gear Classification**: 100% AUC
- **DoS Classification**: 99.38% AUC
- **Benign Classification**: 99.53% AUC

### Dataset Utilizado
- **Tamanho**: ~691k amostras
- **Balanceamento**: BENIGN reduzido para ~1M, todos ataques preservados
- **Split**: 80% treino / 20% teste
- **Validação**: Stratified K-Fold (MLP), GridSearchCV (XGBoost)

## 🔧 Requisitos

- Python 3.8+
- scikit-learn
- xgboost
- pandas
- numpy
- matplotlib / seaborn
- plotly (para visualizações interativas)

## 📖 Como Usar

1. **Navegue aos notebooks em ordem:**
   ```
   notebooks/02_mlp_classifier.ipynb       # MLP training
   notebooks/03_xgboost_classifier.ipynb   # XGBoost + GridSearch
   ```

2. **Dados já estão processados e balanceados** em `data/processed/`

3. **Resultados serão salvos em:**
   - Métricas: `results/metrics/`
   - Visualizações: `results/visualizations/`

## 📝 Próximas Etapas

- [ ] Executar XGBoost GridSearch até conclusão
- [ ] Implementar SVM classifier
- [ ] Gerar relatório comparativo de modelos
- [ ] Documentar recomendações de melhoria
- [ ] Publicação em conference/journal

## 👤 Autor

TCC - Pesquisa em IoV Security

## 📚 Referências

- FORD2019.pdf - Referência em segurança veicular
- HCRK.pdf - Referência em detecção de intrusões CAN bus
- CICIoV2024 - Dataset público Canadian Institute for Cybersecurity

---

**Última atualização:** Abril 2026  
**Status:** Em reorganização e análise
