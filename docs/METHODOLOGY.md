# Metodologia de Pesquisa

## 📚 Estrutura Geral da Pesquisa

```
┌─────────────────────────────────────────────────────────┐
│                  DATASETS BRUTOS                        │
│              (CARDt + CICIoV2024)                      │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│            PREPROCESSAMENTO & LIMPEZA                   │
│  - Drop missings                                        │
│  - Normalizar tipos de dados                            │
│  - Extrair features CAN                                 │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│          ALINHAMENTO & COMBINAÇÃO                       │
│  - Normalizar nomes de colunas                          │
│  - Mapear classes (CARDt → CICIoV2024)                 │
│  - Concatenar datasets                                  │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│       BALANCEAMENTO DE CLASSES                          │
│  - Identificar desbalanceamento                         │
│  - Undersampling de BENIGN                              │
│  - Preservar minorias                                   │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│        SPLIT TREINO / TESTE / VALIDAÇÃO                 │
│  - 80% train / 20% test                                │
│  - Stratified (mantém proporção)                        │
│  - Shuffle aleatória                                    │
└────────────┬────────────────────────────────────────────┘
             │
   ┌─────────┼─────────────────────────┐
   ▼         ▼                         ▼
┌──────┐ ┌────────┐  ┌──────────────────────┐
│ MLP  │ │XGBoost │  │      SVM             │
└──────┘ └────────┘  └──────────────────────┘
   │         │                   │
   ▼         ▼                   ▼
┌──────────────────────────────────────────┐
│       AVALIAÇÃO & COMPARAÇÃO             │
│  - ConfusionMatrix                       │
│  - ROC/AUC Multiclass                    │
│  - Precision/Recall/F1                   │
│  - Tempo de treinamento/predição         │
└──────────────────────────────────────────┘
```

---

## 🔢 Features de Entrada

**Totalmente: 14 features numéricos**

```
1. Timestamp          [int64]    - Milissegundos desde início
2. CAN_ID_Part1       [integer]  - Parte 1 do ID CAN
3. CAN_ID_Part2       [integer]  - Parte 2 do ID CAN
4. DLC                [integer]  - Data Length Code (0-8)
5-12. data_0 to data_7 [integer] - 8 bytes de payload CAN (0-255 cada)
13. feature_engineered_1 [float]  - Statsística derivada (se aplicável)
14. feature_engineered_2 [float]  - Statsística derivada (se aplicável)
```

**Tipos de Features:**
- **Sintáticas**: Estrutura CAN (ID, DLC, bytes)
- **Semânticas**: Padrões de ataque (sequências, frequências)
- **Estatísticas**: Agregações de histograma/distribuição

### Pré-processamento de Features

```python
# Encoded Features (no notebook)
categorical_features = ['val3', 'val4', 'val6', 'val7']  # Para XGBoost
numerical_features = All others

# Scaling
StandardScaler ou MinMaxScaler (depende do modelo)
```

---

## 🎯 Objetivo (Target)

**Classificação Multiclasse (6 classes):**

```python
Classes = {
    0: BENIGN,
    1: DoS,
    2: Fuzzy,
    3: RPM_Spoofing,
    4: Gear/Steering_Spoofing,
    5: Gas/Speed_Spoofing
}

# Encoding: LabelEncoder do scikit-learn
y_encoded = [0, 1, 2, 3, 4, 5, 0, 1, ...]
```

---

## 🤖 Modelo 1: MLP (Multi-Layer Perceptron)

### Arquitetura
```
Input Layer (14 features)
    │
    ▼
┌─────────────────────┐
│ Hidden Layer 1: 30  │  ReLU activation
│ neurons             │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Hidden Layer 2: 30  │  ReLU activation
│ neurons             │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Hidden Layer 3: 30  │  ReLU activation
│ neurons             │
└─────────────────────┘
    │
    ▼
Output Layer (6 neurons - 6 classes)
    │
    ▼
Softmax activation → probabilidades
```

### Hiperparâmetros
```python
MLPClassifier(
    hidden_layer_sizes=(30, 30, 30),
    activation='relu',
    solver='adam',                    # Optimizer
    alpha=0.0001,                     # L2 regularization
    batch_size='auto',
    learning_rate='adaptive',         # Learning rate scheduler
    learning_rate_init=0.001,
    power_t=0.5,
    max_iter=25,                      # Epochs
    warm_start=True,                  # Incremental learning
    momentum=0.9,                     # SGD momentum
    early_stopping=True,              # Stop overfitting
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42
)
```

### Treinamento
```python
# Jupyter: 25 epochs iterativos
# Monitor: Loss (treino/test) e Accuracy (treino/test) por epoch
# Early Stopping: Se validation loss não melhorar em 10 epochs → para

for epoch in range(25):
    clf.partial_fit(X_train, y_train, classes=np.unique(y_train))
    train_loss = calculate_loss(predictions_train, y_train)
    test_loss = calculate_loss(predictions_test, y_test)
    train_acc = accuracy_score(y_train, predictions_train)
    test_acc = accuracy_score(y_test, predictions_test)
    log(epoch, train_loss, test_loss, train_acc, test_acc)
```

### Vantagens
- ✅ Interpretável para estruturas múltiplas camadas
- ✅ Treino rápido (< 5 min em GPU/CPU moderno)
- ✅ Bom baseline comparativo
- ✅ Generaliza bem com regularização

### Limitações
- ❌ Hiperparâmetros requerem ajuste manual
- ❌ Sensível a inicialização de pesos
- ❌ Pode overfit se não houver early stopping

### Resultados
```
              Precision    Recall  F1-Score   Support
      BENIGN       0.999    0.995      0.997    200000
        DoS        0.998    0.994      0.996    132500
       Fuzzy       0.998    0.999      0.998     98300
         RPM       0.999    0.999      0.999    142000
        GEAR       0.998    0.997      0.998    119500
      SPEED       0.996    0.998      0.997     85000

Macro-avg AUC: 99.78%
Micro-avg AUC: 99.95%
```

---

## 🚀 Modelo 2: XGBoost

### Algoritmo
**Extreme Gradient Boosting** - Ensemble de árvores de decisão com:
- Boosting sequencial (cada árvore corrige erros da anterior)
- Regularização L1/L2 integrada
- Suporte a features categóricas nativas

### Hiperparâmetros (GridSearch)
```python
GridSearchCV parameters (2x3x2x2x2 = 48 combinações):

n_estimators:       [100, 200]          # Num. de boosting rounds
max_depth:          [3, 5, 7]           # Profundidade das árvores
learning_rate:      [0.01, 0.1]         # Velocidade aprendizado
subsample:          [0.8, 1.0]          # Fração das amostras por árvore
colsample_bytree:   [0.8, 1.0]          # Fração das features por árvore

enable_categorical=True                 # Features categóricas
categorical_features=['val3','val4','val6','val7']
tree_method='hist'                      # GPU/CPU auto-optimize
```

### Validação
```python
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
# ⚠️ NOTA: n_splits=2 é baixo (ideal seria 5-10)

scoring = 'roc_auc_ovr'  # One-vs-Rest multiclass AUC

Best params found by grid search → train final model
```

### Procedimento
```python
# 1. GridSearch com 2-Fold CV
grid_search = GridSearchCV(
    XGBClassifier(n_jobs=-1, random_state=42),
    param_grid,
    cv=StratifiedKFold(2),
    scoring='roc_auc_ovr',
    n_jobs=-1
)

# 2. Fit na data de treino
grid_search.fit(X_train, y_train)

# 3. Melhor modelo com melhores params
best_clf = grid_search.best_estimator_

# 4. Prever e avaliar
y_pred = best_clf.predict(X_test)
y_pred_proba = best_clf.predict_proba(X_test)
```

### Vantagens
- ✅ Excelente performance em dados estruturados
- ✅ Automaticamente encontra interações entre features
- ✅ Feature importance "de graça"
- ✅ Regularização integrada previne overfitting
- ✅ Muito escalável (C++ backend)

### Limitações
- ❌ **Pendente**: GridSearch ainda não foi executado no notebook
- ❌ Requer mais tempo de tuning que MLP
- ❌ Resultados pode variar com random seed
- ⚠️ n_splits=2 no CV é **muito baixo** (recomendado: 5-10)

### Status Atual
```
[🔄] GridSearch código pronto mas não rodado
[❌] Métricas finais: TBD
[⚠️] Tempo de computação esperado: 30-60 min (dependendo CPU)
```

---

## 🎓 Modelo 3: SVM (Support Vector Machine)

### Status
```
[🚧] IMPLEMENTAÇÃO PENDENTE
[📄] Arquivo criado: models/scripts/svm_classifier.py
```

### Plano de Implementação

**Algoritmo:**
```python
from sklearn.svm import SVC

svm = SVC(
    kernel='rbf',                    # Radial Basis Function
    C=1.0,                           # Regularização
    gamma='scale',                   # Parâmetro kernel
    class_weight='balanced',         # Handle desbalanceamento
    probability=True,                # Output probabilidades (para ROC/AUC)
    random_state=42
)

# Com HyperparameterTuning similar ao XGBoost
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.0001, 0.001],
    'kernel': ['rbf', 'poly']  # Comparar kernels
}
```

**Timeline:**
1. [ ] Implementar SVM com função tunning
2. [ ] Rodar GridSearch (tempo: ~20-40 min)
3. [ ] Gerar confusion matrix
4. [ ] Calcular métricas (precision, recall, AUC)
5. [ ] Comparar com MLP e XGBoost

---

## 📊 Métricas de Avaliação

### Matriz de Confusão
```
Mostra: True Positives, False Positives, True Negatives, False Negatives
Uso: Entender erros de classificação por classe
```

### ROC Curve & AUC
```
ROC (Receiver Operating Characteristic):
- Eixo X: False Positive Rate (FPR)
- Eixo Y: True Positive Rate (TPR)
- Multiclass: One-vs-Rest (6 curvas)

AUC (Area Under Curve):
- 1.0 = Perfect classification
- 0.5 = Random classifier
- Interpretação:
  - 0.90-1.00: Excellent
  - 0.80-0.90: Good
  - 0.70-0.80: Fair
  - < 0.70: Poor
```

### Precision, Recall, F1-Score
```
Precision = TP / (TP + FP)          # Quando prediz ataque, acerta?
Recall = TP / (TP + FN)              # Detecta todos os ataques?
F1 = 2 * (Precision * Recall) / (Precision + Recall)

Macro-average: Média simples entre classes
Micro-average: Agregado de todos TP/FP/TN/FN
Weighted-average: Ponderado pelo suporte (num amostras)
```

### Tempo de Execução
```
- Treino: Quanto tempo leva para treinar?
- Predição: Latência por amostra na fase teste?
- Memória: Quanto espaço o modelo ocupa?
```

---

## 🔄 Validação Cruzada

### MLP
```
- Train/Test Split simples (80/20)
+ Early Stopping integrado
+ Validation set: 10% do train

⚠️ Poderia melhorar com K-Fold
```

### XGBoost
```
- StratifiedKFold com n_splits=2
+ Mantém proporção de classes

⚠️ CRÍTICO: n_splits=2 é MUITO BAIXO
   Recomendação: Elevar para 5-10 splits
```

### SVM
```
- A definir (será implementado later)
```

---

## ⚠️ Problemas Identificados & Soluções

| Problema | Impacto | Solução |
|----------|--------|--------|
| Desbalanceamento de classes | Modelo viesa para BENIGN | ✅ Undersampling implementado |
| XGBoost GridSearch não rodou | Sem resultado XGBoost | 🔄 Executar quando possível |
| SVM não implementado | Sem terceiro modelo | 🚧 Implementação pendente |
| n_splits=2 baixo demais | Validação fraca | 📈 Aumentar para 5-10 |
| Possível data leakage | Métricas artificialmente altas | 🔍 Verificar se CARDt/CICIoV têm overlap |
| Sem análise temporal | Ignora sequências | 💡 Considerar LSTM/GRU |

---

## 📈 Plano de Comparação Final

```
┌──────────────────────────────────────────────────────────┐
│              TABELA COMPARATIVA (3 MODELOS)              │
├─────────────┬──────────────┬─────────────┬───────────────┤
│ Métrica     │ MLP          │ XGBoost     │ SVM           │
├─────────────┼──────────────┼─────────────┼───────────────┤
│ AUC (macro) │ 99.78% ✅    │ TBD 🔄      │ TBD 🚧        │
│ Treino      │ < 5 min      │ 30-60 min   │ ~20-40 min    │
│ Predição    │ ~1 ms/amostra│ ~0.5 ms     │ ~5-10 ms      │
│ Memória     │ Small        │ Small       │ Medium        │
│ Escabilidade│ Good         │ Excellent   │ Poor          │
│ Interpretab.│ Medium       │ High        │ Low           │
│ Recommend.  │ ✅ Prod.     │ 🔄 Pending  │ 🚧 Research   │
└─────────────┴──────────────┴─────────────┴───────────────┘
```

---

**Última atualização:** Abril 2026  
**Status:** Metodologia definida, MLP completo, XGBoost/SVM pendentes
