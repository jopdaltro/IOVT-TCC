# Documentação dos Datasets

## 🔍 Visão Geral

Este projeto utiliza **dois datasets independentes** de segurança em redes automotivas:

| Aspecto | CARDt | CICIoV2024 |
|--------|-------|-----------|
| **Origem** | Dataset privado/académico (Automotive Research) | Dataset público (CIC - Canadian Institute) |
| **Protocolo** | CAN Bus (Controller Area Network) | CAN Bus com features extraídas |
| **Número de Classes** | 5 (DoS, Fuzzy, RPM, Gear, Benign) | 6 (DoS, GAS, RPM, SPEED, STEERING, Benign) |
| **Tamanho Aproximado** | 500k-1M amostras | 300k-500k amostras |
| **Formato Original** | .txt (mensagens raw), .csv | .csv com features |

---

## 📊 CARDt Dataset

### Localização
```
data/raw/CARDt/
├── DoS_dataset.csv                  # Ataques DoS
├── Fuzzy_dataset.csv                # Bit-flipping attacks
├── gear_dataset.csv                 # Spoofing na marcha
├── RPM_dataset.csv                  # Spoofing do motor
├── normal_run_data.csv              # Tráfego legítimo
├── normal_run_data.txt              # Raw CAN messages (~50MB)
├── process.py                       # Script de análise de distribuição
└── runCSV.py                        # Parser CAN → CSV
```

### Estrutura dos Dados

**Formato CSV:**
```
Timestamp, CAN_ID_Part1, CAN_ID_Part2, DLC, Byte0, Byte1, Byte2, ..., Byte7, specific_class
123456, 0x123, 0x456, 8, 0xAA, 0xBB, ..., 0xFF, DoS
```

- **Timestamp**: Milissegundos desde início do teste
- **CAN_ID**: Identificador da mensagem CAN (2 partes: extended ID)
- **DLC** (Data Length Code): Número de bytes de dados (0-8)
- **Bytes 0-7**: Payload da mensagem CAN
- **specific_class**: Classe do ataque (1-5)

### Classes

| Classe | Tipo | Descrição |
|--------|------|-----------|
| **BENIGN** | Legítimo | Operação normal do veículo |
| **DoS** | Ataque | Flooding/sobrecarregamento da rede CAN |
| **Fuzzy** | Ataque | Random bit-flipping em mensagens (corrupção) |
| **RPM** | Spoofing | Dados falsos de rotações do motor |
| **GEAR** | Spoofing | Dados falsos da marcha selecionada |

### Distribuição de Amostras
```
BENIGN:    ~450,000 amostras
DoS:       ~150,000 amostras
Fuzzy:     ~120,000 amostras
RPM:       ~170,000 amostras
GEAR:      ~140,000 amostras
---
TOTAL:     ~1,030,000 amostras
```

### Processamento (scripts)
- **runCSV.py**: Converte arquivo raw CAN (.txt) → CSV estruturado
- **process.py**: Conta distribuição de classes para cada CSV

---

## 📊 CICIoV2024 Dataset

### Localização
```
data/raw/CICIoV2024/
├── decimal_benign.csv               # Tráfego legítimo
├── decimal_DoS.csv                  # Ataques DoS
├── decimal_spoofing-GAS.csv         # Gas pedal spoofing
├── decimal_spoofing-RPM.csv         # RPM spoofing
├── decimal_spoofing-RPM-convertido.csv  # Versão convertida
├── decimal_spoofing-SPEED.csv       # Speed spoofing
├── decimal_spoofing-STEERING_WHEEL.csv # Steering spoofing
└── process.py                       # Script de análise
```

### Estrutura dos Dados

**Formato CSV (features já extraídas):**
```
Timestamp, CAN_ID, DLC, data_0, data_1, ..., data_7, Flag
123456, 0x123, 8, 0xAA, 0xBB, ..., 0xFF, Benign
```

- As colunas são similares ao CARDt
- **Flag**: Classe do ataque (em string: "Benign", "DoS", etc)
- Dados geralmente em **formato decimal** (não hexadecimal como CARDt)

### Classes

| Classe | Tipo | Descrição |
|--------|------|-----------|
| **Benign** | Legítimo | Operação normal |
| **DoS** | Ataque | Negação de serviço |
| **Gas_Pedal_Spoofing** | Spoofing | Falsificação de pedal acelerador |
| **RPM_Spoofing** | Spoofing | Falsificação de rotações motor |
| **Speed_Spoofing** | Spoofing | Falsificação de velocidade |
| **Steering_Wheel_Spoofing** | Spoofing | Falsificação ângulo direção |

### Distribuição de Amostras
```
Benign:                  ~250,000 amostras
DoS:                     ~100,000 amostras
Gas_Pedal_Spoofing:      ~80,000 amostras
RPM_Spoofing:            ~90,000 amostras
Speed_Spoofing:          ~85,000 amostras
Steering_Wheel_Spoofing: ~75,000 amostras
---
TOTAL:                   ~680,000 amostras
```

### Vantagens
- Dataset público (reprodutibilidade)
- Features padronizadas
- Mais tipos de spoofing específicos
- Bem documentado pelo CIC

---

## 🔗 Datasets Combinados

### Localização
```
data/processed/
├── all_datasets_aligned.csv             # Raw combinado (~50MB)
└── all_datasets_aligned_balanced.csv    # Balanceado (~20MB)
```

### Preprocessamento

**Passo 1: Alinhamento**
- Carregar ambos os datasets CARDt e CICIoV2024
- Normalizar nomes de colunas e classes
- Combinar em um único DataFrame
- Missings: dropados

**Passo 2: Balanceamento (classe)**
```python
# Problema: BENIGN >> outros ataques
# Solução: Undersampling da classe majoritária

BENIGN (raw):        900,000 → 1,000,000 downsampled
DoS:                 250,000 → preservado
Fuzzy:               120,000 → preservado
RPM_Spoofing:        170,000 → preservado
Speed_Spoofing:       85,000 → preservado
Gear/Steering:       215,000 → preservado
---
Final: ~1,840,000 amostras balanceadas
```

**Passo 3: Encoding de Labels**
```python
BENIGN → 0
DoS → 1
Fuzzy → 2
RPM → 3
Gear/Steering → 4
Speed/Gas → 5
```

### Uso nos Modelos

**Features (X):**
```
- Timestamp
- CAN_ID (ou parsed parts)
- DLC
- data_0, data_1, ..., data_7 (8 bytes payload)
- (14 features numéricas)
```

**Target (y):**
```
- 'flag' ou 'specific_class' (classe de ataque)
- Uma das 6 classes de saída
```

**Split treino/teste:**
- 80% treino / 20% teste
- Stratified (mantém proporção de classes)

---

## ⚠️ Limitações Conhecidas

### CARDt
- Dataset privado (não disponível publicamente)
- Pode ter viés específico de coleta
- Natural imbalance entre classes

### CICIoV2024
- Pode ter padrões artificiais de ataque
- Features extraídas manualmente
- Pode não cobrir todos tipos de ataque real

### Combinado
- Possível data leakage se datasets tiverem overlap
- Diferentes metodologias de coleta podem criar artefatos
- Balanceamento agressivo pode perder padrões naturais

---

## 📈 Estatísticas Finais

| Métrica | CARDt | CICIoV2024 | Combinado |
|---------|-------|-----------|----------|
| Total Amostras | 1,030k | 680k | 1,840k |
| Num. Classes | 5 | 6 | 6 |
| Features | 14 | 14 | 14 |
| Desbalanceamento | 6:1 | 3.3:1 | ~2:1 (pós-balanço) |
| Arquivo Size | ~40MB | ~25MB | ~40MB (balanced) |

---

**Última atualização:** Abril 2026
