# 📋 Roadmap de Execução - Próximas Etapas

## 🎯 Status Atual (Abril 2026)

✅ **COMPLETO:**
- Reorganização de projeto em pastas lógicas
- Documentação de datasets
- Documentação de metodologia
- Análise de resultados MLP (99.998% AUC)
- Recomendações para melhoria

⚠️ **PENDENTE:**
- XGBoost GridSearch execution
- SVM implementation
- Cross-dataset validation
- Data leakage investigation

---

## 🚀 Fases de Execução

### FASE 1: Validação Crítica (ESTA SEMANA)

**Objetivos:** Confirmar robustez do modelo antes de publicação

#### 1.1 Executar XGBoost GridSearch
```markdown
📁 Notebook: notebooks/03_xgboost_classifier.ipynb
⏱️ Tempo estimado: 30-60 minutos
🔍 Checklist:
  - [ ] Abrir notebook
  - [ ] Executar todas as células sequencialmente
  - [ ] Aguardar conclusão do GridSearchCV
  - [ ] Documentar melhores hiperparâmetros
  - [ ] Salvar métricas finais
  - [ ] Comparar com MLP (ambos em test set)

📊 Output esperado:
  - Best params: {...}
  - Best CV score: ~98-99%
  - Test AUC: ~97-99%
  - Classification report
  - Confusion matrix
```

#### 1.2 Cross-Dataset Validation (Script Python)
```markdown
📁 Arquivo: models/scripts/cross_dataset_validation.py
⏱️ Tempo estimado: 1-2 horas
🔍 O que fazer:
  1. Carregar dados: CARDt vs CICIoV2024
  2. Adicionar coluna 'source' antes de combinar
  3. Treinar MLP apenas em CARDt
  4. Testar em CICIoV2024
  5. Treinar MLP apenas em CICIoV2024
  6. Testar em CARDt
  7. Comparar AUCs

📊 Decisão:
  - AUC > 95% em ambos: ✅ Modelo genérico OK
  - AUC > 98% em ambos: ✅ Excelente generalização
  - AUC < 90% em algum: ⚠️ Possível data leakage
```

#### 1.3 Data Leakage Investigation (Python)
```markdown
📁 Arquivo: models/scripts/data_leakage_check.py
⏱️ Tempo estimado: 1-2 horas
🔍 Verificações:
  1. Similaridade coseno entre datasets
  2. IDs CAN sobrepostos
  3. Distribuições de features
  4. Duplicatas exatas

✅ Se tudo OK: Documentar na seção VALIDATION
❌ Se problema encontrado: Refazer dataset sem overlap
```

#### 1.4 Adversarial Robustness Test (Python)
```markdown
📁 Arquivo: models/scripts/adversarial_test.py
⏱️ Tempo estimado: 2-3 horas
🔍 Testes:
  1. Synthetic Fuzzy (agressivo)
  2. Mixed attacks
  3. Temporal anomalies

📊 Métrica crítica: False Negative Rate
  - < 5%: ✅ OK para produção
  - 5-10%: ⚠️ Marginal
  - > 10%: ❌ Não recomendado
```

---

### FASE 2: Análise Profunda (PRÓXIMA SEMANA)

#### 2.1 Implementar SVM Classifier
```markdown
📁 Arquivo: models/scripts/svm_classifier.py
⏱️ Tempo estimado: 2-3 horas
🔍 Incluir:
  1. SVC com kernel RBF
  2. GridSearchCV com params [C, gamma, kernel]
  3. Treinamento
  4. Métricas (confusion_matrix, classification_report)
  5. ROC curves

📊 Output: Salvar em results/metrics/
```

#### 2.2 Feature Importance Analysis
```markdown
📁 Arquivo: models/scripts/feature_importance.py
⏱️ Tempo estimado: 2-3 horas
🔍 Incluir:
  1. Permutation importance (MLP)
  2. SHAP values (interpretabilidade)
  3. XGBoost feature_importances_
  4. Visualizações

📊 Output: results/visualizations/
```

#### 2.3 Advanced Feature Engineering
```markdown
📁 Arquivo: models/scripts/feature_engineering.py
⏱️ Tempo estimado: 4-5 horas
🔍 Novas features:
  1. Entropia do payload
  2. Variância/std dos bytes
  3. Taxa de mensagens (DoS detection)
  4. Z-score por CAN_ID
  5. Simetria de payload

📊 Treinar MLP+ com nome novo, comparar AUC
   Esperado: 99.998% → 99.999%+
```

#### 2.4 Anomaly Detection Integration
```markdown
📁 Arquivo: models/scripts/anomaly_detection.py
⏱️ Tempo estimado: 3-4 horas
🔍 Modelos:
  1. Isolation Forest
  2. One-Class SVM
  3. Ensemble decisão

📊 Output: models/saved_models/ensemble_ids.pkl
```

---

### FASE 3: Documentação & Paper (FINAL)

#### 3.1 Geração de Comparação Final
```markdown
📁 Arquivo: results/model_comparison_report.md
⏱️ Tempo estimado: 2 horas
🔍 Tabela:
  - MLP vs XGBoost vs SVM
  - Performance (AUC, F1, Recall)
  - Tempo treino/predição
  - Interpretabilidade
  - Recomendação

📊 Conclusão: Qual modelo usar?
```

#### 3.2 Benchmark vs Baselines
```markdown
📁 Arquivo: models/scripts/baseline_comparison.py
⏱️ Tempo estimado: 2-3 horas
🔍 Comparar contra:
  1. Decision Tree
  2. Random Forest
  3. KNN
  4. Logistic Regression

📊 Mostrar que MLP/XGBoost superiores
```

#### 3.3 Geração de Figuras para Paper
```markdown
📁 Outputs: results/visualizations/
🔍 Criar/organizar:
  1. Architecture diagrams
  2. ROC curves (todos modelos)
  3. Confusion matrices
  4. Feature importance plots
  5. Training curves
  6. Performance comparison bars

📊 Formato: High resolution PNG/PDF
```

#### 3.4 README Final
```markdown
📁 Arquivo: README.md (updated)
⏱️ Tempo estimado: 1 hora
🔍 Seções principais:
  1. Visão geral do projeto
  2. Como reproduzir resultados
  3. Datasets
  4. Modelos e performance
  5. Como usar o sistema
  6. Futuras melhorias
  7. Referências
```

---

## 📂 Arquivos a Criar/Modificar

### Scripts Python Novos

| Arquivo | Descrição | Prioridade |
|---------|-----------|-----------|
| models/scripts/cross_dataset_validation.py | Teste entre datasets | 🔴 CRÍTICO |
| models/scripts/data_leakage_check.py | Verificação de leakage | 🔴 CRÍTICO |
| models/scripts/adversarial_test.py | Teste adversarial | 🔴 CRÍTICO |
| models/scripts/svm_classifier.py | SVM classifier | 🟠 ALTO |
| models/scripts/feature_importance.py | Análise de features | 🟠 ALTO |
| models/scripts/feature_engineering.py | Novas features | 🟠 ALTO |
| models/scripts/anomaly_detection.py | Detecção de anomalias | 🟠 ALTO |
| models/scripts/baseline_comparison.py | Comparação com baselines | 🟡 MÉDIO |

### Novo Notebook

| Arquivo | Descrição |
|---------|-----------|
| notebooks/01_data_exploration.ipynb | Análise exploratória dos dados |

### Novos Resultados

| Arquivo | Descrição |
|---------|-----------|
| results/metrics/xgboost_gridearch_results.json | Melhor params XGBoost |
| results/metrics/cross_dataset_validation.csv | Resultados cross-dataset |
| results/visualizations/feature_importance_comparison.png | Feature importance |
| results/model_comparison_report.md | Comparação final modelos |

---

## ✅ Checklist de Execução

### HOJE
- [ ] Reorganização de pastas concluída
- [ ] Documentação criada
- [ ] Paths dos notebooks atualizados
- [ ] requirements.txt criado
- [ ] README.md atualizado

### AMANHÃ (Dia 2)
- [ ] Executar XGBoost GridSearch (1-2 horas)
- [ ] Iniciar cross-dataset validation (1-2 horas)
- [ ] Iniciar data leakage check (1-2 horas)

### SEMANA 1 (Dias 3-5)
- [ ] Completar validação crítica (3 scripts acima)
- [ ] Implementar SVM
- [ ] Feature importance análise
- [ ] Gerar draft de comparação

### SEMANA 2
- [ ] Feature engineering avançada
- [ ] Anomaly detection
- [ ] Benchmark vs baselines
- [ ] Resultados compilados

### SEMANA 3
- [ ] Documentação final
- [ ] Figuras para paper
- [ ] README completo
- [ ] Pronto para submissão/publicação

---

## 🚀 Como Executar

### Quick Start - Reproduzir Resultados Atuais
```bash
cd c:\Users\thelo\Documents\Estudo\IA\TCC

# 1. Instalar dependências
pip install -r requirements.txt

# 2. Rodar MLP (já completo)
jupyter notebook notebooks/02_mlp_classifier.ipynb

# 3. Rodar XGBoost (executar todas as células)
jupyter notebook notebooks/03_xgboost_classifier.ipynb

# 4. Resultados em
# results/visualizations/
# results/metrics/
```

### Ejecutar Scripts de Análise (Quando Prontos)
```bash
# Cross-dataset validation
python models/scripts/cross_dataset_validation.py

# Data leakage check
python models/scripts/data_leakage_check.py

# Adversarial test
python models/scripts/adversarial_test.py

# Feature importance
python models/scripts/feature_importance.py

# SVM classifier
python models/scripts/svm_classifier.py
```

---

## 📊 KPIs de Sucesso

| Métrica | Meta | Status |
|---------|------|--------|
| MLP AUC | > 99.5% | ✅ 99.998% |
| XGBoost AUC | > 98% | ⏳ Pendente |
| SVM AUC | > 97% | ⏳ Pendente |
| Cross-Dataset AUC | > 95% | ⏳ Pendente |
| False Negative Rate | < 5% | ⏳ Pendente |
| Feature Engineering Melhoria | +0.001% AUC | ⏳ Pendente |
| Benchmark vs Random Forest | MLP > RF | ⏳ Pendente |

---

## 📞 Troubleshooting

### Se XGBoost GridSearch for muito lento
```python
# Reduzir grid size
param_grid_small = {
    'n_estimators': [100],      # apenas 1
    'max_depth': [5],            # apenas 1
    'learning_rate': [0.1],      # apenas 1
    'subsample': [0.8],          # apenas 1
    'colsample_bytree': [0.8]    # apenas 1
}
# Depois expandir se necessário
```

### Se cross-dataset AUC cair muito
```python
# Possíveis causas:
# 1. Data leakage → investigar
# 2. Datasets muito diferentes → aplicar domain adaptation
# 3. Classe distribuição muito diferente → usar class_weight
```

### Se feature engineering não melhorar
```python
# Tentar:
# 1. Mais features
# 2. Normalizar features
# 3. Feature selection (manter top 10)
# 4. PCA dimensionality reduction
```

---

## 📚 Referências para Próximas Fases

### Artigos Recomendados
- SMOTE: Chawla et al. 2002 - "SMOTE: Synthetic Minority Over-sampling"
- Feature Importance: Breiman 2001 - "Random Forests"
- SHAP: Lundberg & Lee 2017 - "A Unified Approach to Interpreting Model Predictions"
- Adversarial Testing: Goodfellow et al. 2015 - "Explaining and Harnessing Adversarial Examples"

### Datasets Públicos Relacionados
- NSL-KDD (network intrusion)
- CICIDS2018 (network security)
- UNSW-NB15 (cyber attacks)

### Frameworks Úteis
- SHAP: Model interpretability
- Optuna: Hyperparameter optimization
- ONNX: Model deployment
- Ray Tune: Parallel hyperparameter search

---

**Última atualização:** Abril 2026  
**Status:** Projeto reorganizado e pronto para próximas fases  
**Próximo Milestone:** Executar XGBoost GridSearch (ETA: Hoje/Amanhã)
