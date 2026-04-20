# Recomendações para Melhoria da Pesquisa IOVT IDS

## 🎯 Resumo Executivo

O modelo MLP alcançou **99.998% de AUC** - um resultado excepcional, mas potencialmente indicador de:
1. ✅ Excelente capacidade de detecção em datasets conhecidos
2. ⚠️ Possível data leakage ou viés nos dados
3. 💡 Oportunidades para melhorias que aumentem robustez em produção

Este documento apresenta **10 recomendações prioritizadas** para aumentar a confiabilidade e aplicabilidade da pesquisa.

---

## 📋 Recomendações Prioritizadas

### 🔴 CRÍTICO (Antes de Publicação)

#### 1. **Validação Cross-Dataset**

**Problema:** Modelo treina em dados combinados (CARDt + CICIoV2024), mascarando possível overfitting específico.

**Solução Proposta:**

```python
# Teste 1: Treinar em CARDt → Testar em CICIoV2024
X_train_ardt = data[data['source'] == 'CARDt']
X_test_cicioiv = data[data['source'] == 'CICIoV2024']

mlp.fit(X_train_ardt, y_train_ardt)
y_pred_cicioiv = mlp.predict(X_test_cicioiv)
auc_cross = roc_auc_score(y_test_cicioiv, y_pred_cicioiv)

# Teste 2: Treinar em CICIoV2024 → Testar em CARDt
# Teste 3: Cross-validation por dataset (não por amostra)
```

**Métrica Esperada:**
- Se AUC cai para 95-98%: Indica overfitting → requer regularização
- Se AUC permanece 99%+: Modelo realmente genérico ✅
- Se AUC cai drasticamente (<90%): Data leakage sério

**Impacto:** CRÍTICO - determina se modelo é realmente robusto
**Tempo Estimado:** 1-2 horas
**Resultado Esperado:** Confiança aumentada nas métricas

---

#### 2. **Investigação de Data Leakage**

**Problema:** Performance tão alta pode indicar dados duplicados ou correlacionados entre datasets.

**Análise Proposta:**

```python
# Check 1: Features duplicadas entre datasets
from sklearn.metrics.pairwise import cosine_similarity

# Pegar amostras do CARDt e CICIoV
cardtSamples = data[data['source'] == 'CARDt'].iloc[:1000]
ciciovSamples = data[data['source'] == 'CICIoV2024'].iloc[:1000]

# Calcular similaridade
similarity = cosine_similarity(cardtSamples[features], ciciovSamples[features])
print(f"Mean similarity: {similarity.mean():.4f}")

# Check 2: IDs CAN comuns
ids_cardtSet = set(data[data['source'] == 'CARDt']['CAN_ID'].unique())
ids_cicioiv_set = set(data[data['source'] == 'CICIoV2024']['CAN_ID'].unique())
common_ids = ids_cardtSet & ids_cicioiv_set
print(f"Overlapping CAN IDs: {len(common_ids)} out of {len(ids_cardtSet)}")

# Check 3: Sequências temporais sobrepostas
# Se dois datasets têm mensagens idênticas com timestamps próximos...

# Check 4: Distribuição de features por dataset
"""Visualizar se distribuições são muito similares"""
for feature in ['data_0', 'data_1']:
    plt.hist(data[data['source']=='CARDt'][feature], alpha=0.5, label='CARDt')
    plt.hist(data[data['source']=='CICIoV2024'][feature], alpha=0.5, label='CICIoV')
    plt.legend()
    plt.show()
```

**Indicadores de Data Leakage:**
- Similaridade coseno > 0.95 entre datasets
- > 50% de IDs CAN comuns
- Distribuições de features idênticas
- Duplicatas exatas encontradas

**Impacto:** CRÍTICO - invalida toda a comparação se confirmado
**Tempo Estimado:** 2-3 horas
**Ação pós-resultado:**
- Se comprovado: Separar datasets, refazer análise
- Se negado: Validação cross-dataset ainda recomendada

---

#### 3. **Teste de Robustez em Ataques Não Vistos**

**Problema:** Modelo foi treinado em dados CARDt+CICIoV conhecidos. Como se comporta com novos padrões?

**Solução Proposta:**

```python
# Criar dados adversariais (ataques não vistos no treino)

# Teste 1: Synthetic Fuzzy (random bit-flipping desconhecida)
synthetic_fuzzy = X_test.copy()
for row in synthetic_fuzzy.iterrows():
    # Bit-flipping mais agressivo que dados de treino
    num_flips = random.randint(20, 50)  # vs avg 10-15 no treino
    flip_positions = random.sample(list(range(8)), min(num_flips, 8))
    for pos in flip_positions:
        synthetic_fuzzy.at[row[0], f'data_{pos}'] ^= 0xFF  # XOR com 0xFF

# Teste 2: Mixed attacks (combinação de DoS + Spoofing)
synthetic_mixed = X_test.copy()
for row in synthetic_mixed.iterrows():
    # Acelerar uma das features DoS + deslocar outra Spoofing
    synthetic_mixed.loc[row[0], 'data_0'] += 50  # DoS pattern
    synthetic_mixed.loc[row[0], 'data_1'] *= 1.5  # Spoofing pattern

# Teste 3: Temporal anomaly (sequências não vistas)
# Se houver timestamp: criar padrões de frequência anormais

# Avaliar no modelo
y_pred_synthetic = mlp.predict(synthetic_fuzzy)
y_proba_synthetic = mlp.predict_proba(synthetic_fuzzy)

# Métrica: Quantos foram classificados como BENIGN? (false negative perigoso)
false_negatives_ratio = (y_pred_synthetic == 0).sum() / len(synthetic_fuzzy)
print(f"False Negative Rate em ataques sintéticos: {false_negatives_ratio:.2%}")
```

**Critério de Sucesso:**
- False Negative Rate < 5% (ideal < 1%)
- Se > 10%: Modelo não é robusto para produção

**Impacto:** CRÍTICO - valida safety em campo
**Tempo Estimado:** 2-3 horas
**Resultado Esperado:** Confiança em robustez ou identificação de limitações

---

### 🟠 ALTO (Antes de Submissão)

#### 4. **Análise de Feature Importance**

**Problema:** Não sabemos quais features o modelo usa para decisão → black box.

**Solução Proposta:**

```python
# Abordagem 1: Permutation Feature Importance
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(
    mlp, X_test, y_test, n_repeats=10, random_state=42
)

# Plot feature importance
features_importance_df = pd.DataFrame({
    'feature': X_test.columns,
    'importance': perm_importance.importances_mean,
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(features_importance_df['feature'], features_importance_df['importance'])
plt.xlabel('Permutation Importance')
plt.title('Feature Importance for MLP IDS')
plt.tight_layout()
plt.savefig('results/visualizations/mlp_feature_importance.png')

# Abordagem 2: SHAP (SHapley Additive exPlanations)
import shap

explainer = shap.TreeExplainer(None)  # Para MLP, usar DeepExplainer
shap_values = explainer.shap_values(X_test[:1000])  # Subset para performance

shap.summary_plot(shap_values, X_test[:1000], show=False)
plt.savefig('results/visualizations/shap_summary.png')

# Abordagem 3: Para XGBoost (feature_importances_)
xgb_importance = xgb_model.feature_importances_
```

**Análise Esperada:**
- Quais 5 features mais importantes para detecção?
- Correlação com conhecimento de domínio (CAN IDs, DLC)?
- Bytes payload vs metadados (qual mais relevante)?

**Impacto:** ALTO - essencial para interpretabilidade
**Tempo Estimado:** 2-4 horas
**Output:** Gráficos de importância para incluir em paper

---

#### 5. **Detecção de Anomalias Complementar**

**Problema:** Detecção supervisionada falha com ataques desconhecidos. Anomalia não-supervisionada ajuda.

**Solução Proposta:**

```python
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

# Abordagem 1: Isolation Forest (Unsupervised)
iso_forest = IsolationForest(contamination=0.05, random_state=42)
anomaly_scores_if = iso_forest.fit_predict(X_train)  # -1 = anomalia, 1 = normal

# Combine com MLP: 
# - MLP detecta ataques conhecidos (supervisionado)
# - IF detecta anomalias desconhecidas (não supervisionado)
ensemble_predictions = []
for i in range(len(X_test)):
    mlp_pred = mlp.predict([X_test.iloc[i]])[0]
    if_pred = iso_forest.predict([X_test.iloc[i]])[0]
    
    if mlp_pred != 0:  # MLP detectou algo
        ensemble_predictions.append(mlp_pred)
    elif if_pred == -1:  # IF detectou anomalia
        ensemble_predictions.append('UNKNOWN_ANOMALY')
    else:
        ensemble_predictions.append(0)  # Normal

# Abordagem 2: One-Class SVM
from sklearn.svm import OneClassSVM

oc_svm = OneClassSVM(kernel='rbf', nu=0.05)
oc_svm.fit(X_train[y_train == 0])  # Treinar apenas em BENIGN
oc_anomalies = oc_svm.predict(X_test)

# Abordagem 3: Estatísticas por tipo de CAN ID
# Se CAN_ID nunca visto antes, é anômalo
seen_ids = set(X_train['CAN_ID'].unique())
unknown_id_flag = X_test['CAN_ID'].apply(lambda x: x not in seen_ids)
```

**Integração:**

```
Decisão = MLP_prediction se MLP confiante
        = Anomaly se IF ou OneClassSVM flags
        = Unknown_type caso contrário
```

**Impacto:** ALTO - aumenta capacidade real do sistema
**Tempo Estimado:** 4-6 horas
**Resultado:** Detector híbrido supervisionado + não-supervisionado

---

#### 6. **Feature Engineering Avançada**

**Problema:** Apenas 14 features brutos. Mais features "semânticas" podem melhorar.

**Propostas de Novas Features:**

```python
# Feature Engineering: Estatísticas de Sequência

def extract_advanced_features(df):
    new_features = pd.DataFrame()
    
    # 1. Entropía dos bytes payload
    payload_cols = [f'data_{i}' for i in range(8)]
    new_features['payload_entropy'] = df[payload_cols].apply(
        lambda row: entropy(np.histogram(row, bins=256)[0]), axis=1
    )
    
    # 2. Variância dos bytes
    new_features['payload_variance'] = df[payload_cols].var(axis=1)
    new_features['payload_std'] = df[payload_cols].std(axis=1)
    new_features['payload_mean'] = df[payload_cols].mean(axis=1)
    
    # 3. Padrões de simetria
    new_features['symmetry_score'] = df[payload_cols].apply(
        lambda row: sum([row[i] == row[7-i] for i in range(4)]) / 4, axis=1
    )
    
    # 4. Frequência de valores zero/máximo
    new_features['zero_count'] = (df[payload_cols] == 0).sum(axis=1)
    new_features['max_count'] = (df[payload_cols] == 255).sum(axis=1)
    
    # 5. Histograma de CAN_ID (por janela temporal)
    # Agregações por timestamp
    new_features['msg_rate_window'] = df.groupby('timestamp').size()
    
    # 6. Anomalias por CAN_ID
    can_id_stats = df.groupby('CAN_ID')['payload_entropy'].agg(['mean', 'std'])
    df = df.merge(can_id_stats, left_on='CAN_ID', right_index=True, 
                  how='left', suffixes=('', '_by_id'))
    new_features['entropy_zscore'] = (
        (df['payload_entropy'] - df['mean_by_id']) / df['std_by_id']
    )
    
    # 7. Mudanças consecutivas (diferenças entre mensagens)
    new_features['data_0_diff'] = df['data_0'].diff().fillna(0)
    new_features['dlc_change'] = df['DLC'].diff().fillna(0)
    
    return new_features.fillna(0)

# Usar features
X_enhanced = pd.concat([X_train, extract_advanced_features(X_train)], axis=1)
```

**Features Recomendadas (em ordem de prioridade):**
1. Entropia do payload (detecta randomness de Fuzzy)
2. Variância/Std dos bytes (detecta padrões anormais)
3. Taxa de mensagens por janela (detecta DoS/flooding)
4. Z-score por CAN_ID (encontra anomalias por tipo)
5. Simetria de payload (detecta padrões artificiais)

**Impacto:** ALTO - pode melhorar AUC de 99.99% para 99.995%+
**Tempo Estimado:** 4-6 horas (incluindo validação)
**Resultado:** Dataset melhorado com 30-40 features instead of 14

---

#### 7. **Técnicas de Balanceamento Avançado**

**Problema:** Undersampling atual pode descartar padrões importantes. SMOTE é alternativa.

**Solução Proposta:**

```python
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Problema: Classes muito desbalanceadas ainda (BENIGN 27%, Fuzzy 5%)
# Solução: Combinar técnicas

pipeline = Pipeline([
    # 1. Undersampling das classes maiores
    ('under', RandomUnderSampler(
        sampling_strategy={0: 100000},  # BENIGN para 100k
        random_state=42
    )),
    # 2. Oversampling das classes menores
    ('over', SMOTE(
        sampling_strategy={2: 50000, 4: 50000},  # Fuzzy e Gear
        random_state=42
    )),
])

X_balanced, y_balanced = pipeline.fit_resample(X_train, y_train)

print(f"Classes distribution before: {np.bincount(y_train)}")
print(f"Classes distribution after:  {np.bincount(y_balanced)}")

# Treinar em X_balanced, y_balanced
mlp_smote = MLPClassifier(...).fit(X_balanced, y_balanced)

# Avaliar se SMOTE fez diferença
y_pred_smote = mlp_smote.predict(X_test)
auc_smote = roc_auc_score(y_test, y_pred_smote, multi_class='ovr')
print(f"AUC com SMOTE: {auc_smote:.4f}")
```

**Comparação Esperada:**
- Undersampling puro: 99.998% (atual)
- SMOTE: 99.998-99.999% (similar ou melhor)
- Vantagem: Menos perda de informação

**Impacto:** MÉDIO - melhoria marginal esperada
**Tempo Estimado:** 2-3 horas
**Benefício:** Mais científico que undersampling simples

---

### 🟡 MÉDIO (Para Futuras Publicações)

#### 8. **Análise Temporal - Modelos RNN/LSTM**

**Problema:** Modelo atual trata cada mensagem independentemente. Ignora sequências temporais.

**Problema Identificado:** Se arquivo tem timestamps, RNN pode aproveitar sequências.

**Solução Proposta:**

```python
# Verificar se timestamps são significativos
df['time_diff'] = df['timestamp'].diff()
print(df['time_diff'].describe())

# Se timestamps monotônicos → podemos criar sequências
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense

# Preparar dados sequenciais
def create_sequences(X, y, seq_length=10):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])  # Predict próxima classe
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_train.values, y_train.values, seq_length=10)

# Modelo LSTM
model = keras.Sequential([
    LSTM(32, activation='relu', input_shape=(10, 14)),  # seq_length=10, 14 features
    LSTM(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(6, activation='softmax')  # 6 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_seq, keras.utils.to_categorical(y_seq, 6), epochs=10)

# Avaliar
y_pred_lstm = model.predict(X_seq_test)
```

**Quando Implementar:**
- Se timestamps revelados como temporalmente significativos
- Se análise temporal é crítica (ex: DoS em bursts)

**Impacto:** MÉDIO - possível melhoria de 1-2% AUC
**Tempo Estimado:** 8-12 horas (inclui experimentação)

---

#### 9. **Teste em Dados Real-World**

**Problema:** Dados ainda são de lab/simulação. Real-world pode ter padrões diferentes.

**Proposta de Validação:**

```markdown
Fase 1: Contato com fabricantes
- Mercedes, BMW, Tesla, Volkswagen
- Solicitar dados de telemetria real (anônimos)
- Assinar NDAs se necessário

Fase 2: Preparação
- Aligned datasets real-world com formato de treino
- Balanceamento similar (se possível)
- Validação cruzada: parte real + parte sintético

Fase 3: Teste
- Treinar em dados sintéticos (atual)
- Testar em dados real-world
- Medir degradação de performance

Fase 4: Retrainamento (se necessário)
- Fine-tuning do modelo com dados reais
- Transfer learning: usar pesos MLP + retrainer com real data
```

**Métrica de Sucesso:**
- AUC em dados real: > 95% (aceitável)
- AUC em dados real: > 98% (excelente)
- AUC em dados real: < 90% (modelo não é robusto)

**Impacto:** CRÍTICO para produção, MÉDIO para publicação
**Timeline:** Depende da receptividade de fabricantes

---

#### 10. **Benchmarking contra Baselines Conhecidos**

**Problema:** Não sabemos como nosso modelo compara a sistemas existentes.

**Solução Proposta:**

```python
# Comparar contra baselines

# 1. Modelo Trivial (Baseline)
from sklearn.tree import DecisionTreeClassifier
dt_baseline = DecisionTreeClassifier(max_depth=3)
dt_baseline.fit(X_train, y_train)
auc_dt = roc_auc_score(y_test, dt_baseline.predict(X_test), multi_class='ovr')
print(f"Decision Tree baseline AUC: {auc_dt:.4f}")

# 2. Random Forest (Strong baseline)
from sklearn.ensemble import RandomForestClassifier
rf_baseline = RandomForestClassifier(n_estimators=100, max_depth=10)
rf_baseline.fit(X_train, y_train)
auc_rf = roc_auc_score(y_test, rf_baseline.predict_proba(X_test), multi_class='ovr')
print(f"Random Forest baseline AUC: {auc_rf:.4f}")

# 3. KNN (Outro baseline)
from sklearn.neighbors import KNeighborsClassifier
knn_baseline = KNeighborsClassifier(n_neighbors=5)
knn_baseline.fit(X_train, y_train)
auc_knn = roc_auc_score(y_test, knn_baseline.predict_proba(X_test), multi_class='ovr')
print(f"KNN baseline AUC: {auc_knn:.4f}")

# 4. Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_baseline = LogisticRegression(max_iter=1000, multi_class='multinomial')
lr_baseline.fit(X_train, y_train)
auc_lr = roc_auc_score(y_test, lr_baseline.predict_proba(X_test), multi_class='ovr')
print(f"Logistic Regression baseline AUC: {auc_lr:.4f}")

# Tabela comparativa
comparison_df = pd.DataFrame({
    'Model': ['Decision Tree', 'Random Forest', 'KNN', 'Logistic Reg', 'MLP', 'XGBoost', 'SVM'],
    'AUC': [auc_dt, auc_rf, auc_knn, auc_lr, 0.99998, 'TBD', 'TBD'],
    'Speed': ['Fast', 'Medium', 'Slow', 'Fast', 'Medium', 'Slow', 'Slow'],
    'Interpretable': ['High', 'Medium', 'Low', 'High', 'Low', 'Medium', 'Low']
})

print(comparison_df)
comparison_df.to_csv('results/model_comparison.csv', index=False)
```

**Resultado Esperado:**

| Model | Baseline AUC | Observação |
|-------|--------------|-----------|
| Decision Tree | ~85% | Trivial |
| Random Forest | ~97% | Strong baseline |
| KNN | ~91% | Slow |
| Logistic Reg | ~88% | Linear |
| **MLP** | **99.998%** | **OURS - Excellent** |

**Impacto:** MÉDIO - essencial para contextualizar contribuição
**Tempo Estimado:** 3-4 horas

---

## 📊 Matriz de Priorização

| # | Recomendação | Criticidade | Impacto | Esforço | Deadline Sugerido |
|----|--------------|-----------|--------|--------|------------------|
| 1 | Cross-Dataset | 🔴 Críticо | Alto | 2h | **IMEDIATO** |
| 2 | Data Leakage | 🔴 Crítico | Alto | 3h | **IMEDIATO** |
| 3 | Robustez (Adversarial) | 🔴 Crítico | Alto | 3h | **IMEDIATO** |
| 4 | Feature Importance | 🟠 Alto | Alto | 3h | Esta semana |
| 5 | Anomaly Detection | 🟠 Alto | Médio | 5h | Esta semana |
| 6 | Feature Engineering | 🟠 Alto | Médio | 5h | Esta semana |
| 7 | SMOTE Balancing | 🟠 Alto | Baixo | 3h | Próxima semana |
| 8 | LSTM Temporal | 🟡 Médio | Médio | 10h | Mês que vem |
| 9 | Real-World Data | 🟡 Médio | Crítico | 40h+ | Depende de parceiros |
| 10 | Benchmarking | 🟡 Médio | Médio | 4h | Antes de submissão |

---

## 🎯 Roadmap Recomendado

### SEMANA 1 (Imediato)
```
[X] Reorganizar projeto ✅ COMPLETO
[ ] Cross-dataset validation (recomendação #1)
[ ] Data leakage investigation (recomendação #2)
[ ] Adversarial robustness test (recomendação #3)
[ ] Executar XGBoost GridSearch
[ ] Implementar SVM
```

**Status:** Se tudo passar → Modelo está pronto para submissão
**Status:** Se falhar → Limitações claras precisam ser documentadas

### SEMANA 2
```
[ ] Feature importance analysis (recomendação #4)
[ ] Anomaly detection complement (recomendação #5)
[ ] Advanced feature engineering (recomendação #6)
[ ] Gerar comparação final MLP vs XGBoost vs SVM
[ ] Documentar em formato paper
```

### SEMANA 3-4
```
[ ] SMOTE balancing experiments (recomendação #7)
[ ] Benchmarking vs baselines (recomendação #10)
[ ] LSTM temporal model (opcional, recomendação #8)
[ ] Refinamentos finais
[ ] Preparar para submissão/publicação
```

### FUTURO
```
[ ] Contatar fabricantes automotivos (recomendação #9)
[ ] Real-world data validation
[ ] Deployment em sistemas veiculares
[ ] Paper review e submissão
```

---

## 💡 Conclusão

O modelo MLP alcançou **performance excepcional (99.998% AUC)**, mas pesquisa não está completa. As **10 recomendações acima** transformam um resultado laboratório em contribuição científica sólida:

### ✅ O que Temos
- Modelo robusto em dados balanceados
- Rápido e eficiente (treina em 2-3 min)
- Pronto para produção básica

### ⚠️ O que Falta
- Validação cross-dataset (essencial)
- Teste em ataques não vistos
- Comparação com baseline
- Feature importance

### 🚀 Potencial
- Com recomendações CRÍTICAS: Publicável em conference Tier-2
- Com todas as recomendações: Publicável em conference Tier-1
- Com real-world data: Potencial produto viável

---

**Próximo Passo:** Implementar recomendações #1-#3 (CRÍTICAS) antes de qualquer submissão.

**Contato para Dúvidas:** Revirar este documento periodicamente conforme progresso.

---

**Última atualização:** Abril 2026  
**Versão:** 1.0 (Baseado em análise MLP 99.998% AUC)
