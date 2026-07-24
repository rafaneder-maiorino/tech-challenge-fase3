# Dataset Card — Medical Abstracts (Tech Challenge Fase 3)

## 1. Identificação

| Campo | Valor |
|---|---|
| Nome | Medical Text / Medical Abstracts TC Corpus |
| Fonte | Hugging Face Hub — [`123rc/medical_text`](https://huggingface.co/datasets/123rc/medical_text) |
| Licença | Apache 2.0 |
| Idioma | Inglês |
| Tarefa | Classificação de texto multiclasse (5 classes) |
| Arquivos | `train.csv` (11.550 linhas), `test.csv` (2.888 linhas) |
| Colunas | `condition_label` (int, 1–5), `medical_abstract` (texto livre) |
| Cópia local | `data/raw_abstracts/` |

O repositório foi identificado por correspondência exata dos hashes de blob dos
arquivos baixados (`train.csv` → `1b46b58c2a07f059647e2046b75902515684fb1e`).

### Classes

| `condition_label` | Condição |
|---|---|
| 1 | neoplasms (neoplasias) |
| 2 | digestive system diseases (doenças do aparelho digestivo) |
| 3 | nervous system diseases (doenças do sistema nervoso) |
| 4 | cardiovascular diseases (doenças cardiovasculares) |
| 5 | general pathological conditions (condições patológicas gerais) |

---

## 2. Números do EDA

### Volume e unicidade

| Split | Linhas brutas | Abstracts únicos | Linhas duplicadas |
|---|---|---|---|
| train | 11.550 | 9.445 | 2.105 (18,2%) |
| test | 2.888 | 2.770 | 118 (4,1%) |

Nenhum valor nulo em `condition_label` ou `medical_abstract`.

### Comprimento dos textos (train bruto)

| Métrica | Caracteres | Palavras |
|---|---|---|
| mínimo | 170 | 24 |
| p25 | 847 | 122 |
| **mediana** | **1.208** | **175** |
| média | 1.229 | 180 |
| p75 | 1.587 | 234 |
| máximo | 3.999 | 596 |

Textos curtos e homogêneos (~180 palavras por abstract), compatíveis com um
classificador leve TF-IDF + modelo linear e com o requisito de baixa latência
da Fase 3.

### Distribuição das 5 classes (train bruto, 11.550 linhas)

| Classe | Linhas | % |
|---|---|---|
| 1 — neoplasms | 2.530 | 21,9% |
| 2 — digestive system diseases | 1.195 | 10,4% |
| 3 — nervous system diseases | 1.540 | 13,3% |
| 4 — cardiovascular diseases | 2.441 | 21,1% |
| 5 — general pathological conditions | 3.844 | 33,3% |

O `test.csv` tem distribuição praticamente idêntica (21,9% / 10,4% / 13,3% /
21,1% / 33,3%). Desbalanceamento moderado (~3:1 entre a maior e a menor
classe) — tratado com `class_weight='balanced'` no baseline.

---

## 3. Achados de qualidade e decisões do pipeline

Dois problemas estruturais foram encontrados no corpus bruto — ambiguidade de
rótulo (3.1) e vazamento entre splits (3.3). Ambos são tratados em
`src/data/prepare.py`, que registra cada etapa via `logging`. A seção 3.2
documenta o experimento que definiu como tratar o primeiro.

### 3.1 O corpus é multirrótulo achatado (não é 1 linha por abstract)

O mesmo abstract aparece em até 4 linhas, com `condition_label` diferente em
cada uma:

| Ocorrências do mesmo abstract | Qtd. de abstracts (train) |
|---|---|
| 1 linha | 7.489 |
| 2 linhas | 1.810 |
| 3 linhas | 143 |
| 4 linhas | 3 |

**1.956 dos 9.445 abstracts únicos do train (20,7%) têm rótulos conflitantes**
(113 de 2.770, ou 4,1%, no test). Portanto a deduplicação não é apenas remoção
de linhas repetidas: é uma decisão de resolução de rótulo.

**Decisão: descartar os abstracts ambíguos** (`--label-strategy unambiguous`,
default). Sobram 7.489 abstracts no train e 2.657 no test — 100% deles com um
único rótulo. A tarefa permanece multiclasse, formato exigido pelo enunciado.
O total de descartados é logado a cada execução.

A estratégia anterior — **voto majoritário com desempate pelo menor
`condition_label`** — continua disponível via `--label-strategy majority`,
para reprodutibilidade do experimento da seção 3.2.

### 3.2 Evidência: por que descartar e não forçar um rótulo

`src/experiments/label_strategy_comparison.py` compara as duas estratégias com
**todo o resto idêntico** (mesma deduplicação, mesma remoção de vazamento,
mesmo split 85/15, `random_state=42`, mesmo TF-IDF + LogisticRegression). A
regra de rótulo é aplicada ao `train.csv` **e** ao `test.csv` — aplicá-la só ao
train faria as duas colunas medirem coisas diferentes.

| | majority | unambiguous |
|---|---|---|
| n_train / n_val / n_test | 7.188 / 1.269 / 2.770 | 5.634 / 995 / 2.657 |
| validação — acurácia / macro-F1 | 0,7959 / 0,7863 | 0,7970 / 0,7924 |
| test — acurácia / macro-F1 | 0,6653 / 0,6666 | 0,6699 / 0,6707 |
| **test comum — acurácia / macro-F1** | **0,6632 / 0,6644** | **0,6699 / 0,6707** |

Como `unambiguous` também encolhe o holdout (2.770 → 2.657), as linhas "test"
das duas colunas não são o mesmo conjunto. A linha **test comum** restringe
ambos os modelos às 2.657 linhas presentes nos dois holdouts — a única
comparação estritamente pareada.

**Resultado: `unambiguous` ganha +0,63 ponto de macro-F1 nas mesmas linhas,
treinando com 22% menos exemplos** (5.634 vs 7.188).

#### O mecanismo do viés na classe 5

Todo o ganho está concentrado em uma classe:

| Classe | F1 majority | F1 unambiguous | Δ |
|---|---|---|---|
| 1 — neoplasms | 0,7589 | 0,7637 | +0,005 |
| 2 — digestive system diseases | 0,6547 | 0,6440 | −0,011 |
| 3 — nervous system diseases | 0,6595 | 0,6473 | −0,012 |
| 4 — cardiovascular diseases | 0,7538 | 0,7498 | −0,004 |
| **5 — general pathological conditions** | **0,5063** | **0,5486** | **+0,042** |

Pelas matrizes de confusão, a diferença é de **recall** da classe 5:
**0,411 → 0,470** (360/876 → 412/876 acertos), com precisão praticamente
inalterada (0,659 vs 0,658).

A explicação é o desempate. A classe 5 ("general pathological conditions") é o
rótulo genérico atribuído *em conjunto* com um rótulo específico — é justamente
a classe que mais aparece nos abstracts ambíguos. Como o desempate favorece o
**menor** `condition_label`, esses casos eram sistematicamente reetiquetados
como 1–4. O modelo aprendia, a partir de rótulo errado, a evitar a classe 5.
Descartar os ambíguos remove esse ruído em vez de codificá-lo no alvo.

*Ressalva:* a margem é pequena — 0,0063 de macro-F1 equivalem a ~18 predições
em 2.657, em um único split com uma única seed. O que a evidência sustenta com
segurança não é que `unambiguous` seja superior, e sim que **descartar 22% do
treino não custa desempenho**, e produz um alvo sem ruído introduzido por
regra arbitrária de desempate. Confirmar a diferença exigiria múltiplas seeds
ou bootstrap sobre o test.

Números completos (distribuições por split, F1 por classe, matrizes de
confusão das duas estratégias) em `reports/label_strategy_comparison.json`.

### 3.3 Vazamento entre train e test

**860 abstracts aparecem simultaneamente em `train.csv` e `test.csv`** — 11,5%
do train e **32,4% do test**, já após o descarte dos ambíguos. Sem tratamento,
quase um terço da avaliação de holdout mediria memorização, não generalização.

**Decisão:** remover os 860 abstracts do **train**, preservando o test
intacto como holdout honesto. Train: 7.489 → 6.629 abstracts.

*Consequência conhecida:* os abstracts vazados são desproporcionalmente da
classe 5, então removê-los do train introduz um **desvio de prior** entre
treino e holdout:

| Classe | % no train final | % no test |
|---|---|---|
| 1 | 26,5% | 22,5% |
| 2 | 8,5% | 10,0% |
| 3 | 12,5% | 13,1% |
| 4 | 23,5% | 21,5% |
| 5 | 29,1% | 33,0% |

Esse desvio é a principal explicação para a queda de desempenho entre
validação e test no baseline (ver `reports/baseline_metrics.json`): a classe 5
tem recall 0,70 na validação e 0,47 no test. `class_weight='balanced'`
compensa apenas parcialmente. A alternativa — remover os duplicados do test em
vez do train — preservaria o prior mas reduziria o holdout para 1.797 exemplos
e deixaria o conjunto de avaliação enviesado para abstracts "não repetidos";
optou-se pela avaliação honesta.

Note que o desvio é menor sob `unambiguous` do que era sob `majority`
(classe 5: 29,1% vs 33,0% aqui, contra 22,7% vs 31,6% antes) — descartar os
ambíguos também atenua parte do desbalanceamento induzido pelo vazamento.

### 3.4 Splits finais

| Split | Abstracts | Origem |
|---|---|---|
| `data/processed/train.parquet` | 5.634 | 85% do train limpo, estratificado |
| `data/processed/val.parquet` | 995 | 15% do train limpo, estratificado |
| `data/processed/test.parquet` | 2.657 | `test.csv` sem ambíguos (holdout) |

`random_state=42` em toda a pipeline.

---

## 4. Decisão sobre o alvo

O enunciado da Fase 3 descreve um cenário de **triagem de urgência**
(normal / atenção / urgente) e, na seção "Dataset Sugerido", recomenda
explicitamente o *Medical Abstracts TC Corpus*. Há uma tensão entre as duas
coisas, e ela é assumida aqui de forma explícita:

- **O corpus rotula condição médica, não urgência clínica.** `condition_label`
  identifica o sistema/grupo de doenças discutido no abstract (neoplasias,
  digestivo, nervoso, cardiovascular, condições gerais). Não existe nenhum
  sinal de gravidade, prioridade ou tempo-para-atendimento no rótulo.
- **O enunciado sugere este dataset** e exige apenas "uma coluna de texto e
  uma coluna de target (classificação/urgência) com pelo menos 2.000
  amostras" — requisito atendido com folga (7.489 abstracts de rótulo único;
  5.634 no treino final após remoção de vazamento).
- **Optou-se por não fabricar um alvo de urgência.** Rotular urgência a partir
  destes textos exigiria supervisão fraca por palavras-chave, cuja avaliação
  (seção 5.2) mostrou cobertura e consistência insuficientes. Um alvo
  inventado tornaria todas as métricas do projeto não interpretáveis.

**O pipeline é agnóstico ao alvo.** `src/data/prepare.py` e
`src/models/baseline.py` dependem apenas de duas colunas — `medical_abstract`
(texto) e `condition_label` (classe inteira). A cadeia completa da Fase 3
(API FastAPI → Docker → CI/CD → DAG Airflow → Prometheus/Grafana → ONNX)
é exercitada de ponta a ponta independentemente da semântica do rótulo.
Trocar por um corpus de urgência real, quando disponível, é uma questão de
apontar o pipeline para outro CSV com as mesmas duas colunas.

Na documentação e na demonstração, o classificador é apresentado pelo que ele
de fato faz: **roteamento de laudos por área clínica** — uma tarefa real de
triagem administrativa hospitalar, ainda que não seja triagem de gravidade.

---

## 5. Datasets avaliados e descartados

### 5.1 FedMML Emergency Department Triage Dataset

- Fonte: Hugging Face (`data/raw/fedmml_ed_triage_dataset.csv`), licença
  CC-BY-4.0, dados **sintéticos**.
- Alvo `esi_level` (1–5), que é exatamente a escala de urgência do enunciado.

**Motivos do descarte:**

1. **Volume real muito menor que o aparente.** 87.234 linhas, mas apenas
   **4.231 notas clínicas únicas** — cada texto se repete ~20 vezes em
   encontros diferentes. O volume efetivo cai para menos da metade do corpus
   de abstracts.
2. **Rótulo carimbado no texto.** As notas contêm o próprio nível de urgência
   em linguagem quase literal. Exemplo de nota com `esi_level` baixo:
   *"67yo M requesting Medication question. Patient ambulatory, no acute
   distress. **Non-urgent visit.**"* Um classificador treinado nisso aprende a
   detectar o carimbo, não o quadro clínico — acurácia próxima do teto e sem
   significado.
3. Distribuição fortemente concentrada em ESI 3 (41.389 linhas, 47%) e
   ESI 1 com apenas 924 linhas.

### 5.2 Supervisão fraca (weak supervision) sobre os abstracts

Tentativa de derivar um alvo de urgência a partir dos próprios abstracts, com
duas listas de marcadores léxicos:

- **Alta urgência:** `acute`, `emergency`, `mortality`, `fatal`,
  `intensive care`, `life-threatening`
- **Baixa urgência:** `chronic`, `elective`, `routine`, `asymptomatic`,
  `mild`, `follow-up`

Resultado sobre os 9.445 abstracts únicos:

| Situação | Abstracts | % |
|---|---|---|
| Só marcador de alta urgência | 1.201 | 12,7% |
| Só marcador de baixa urgência | 1.773 | 18,8% |
| **Rotuláveis sem conflito** | **2.974** | **31,5%** |
| Marcadores dos dois grupos (conflito) | 586 | 6,2% |
| Nenhum marcador (descartados) | 5.885 | 62,3% |

**Motivos do descarte:**

1. **Cobertura de 31,5%.** Quase dois terços do corpus ficariam sem rótulo, e
   os 2.974 restantes seriam uma amostra enviesada — só os abstracts que por
   acaso usam vocabulário de urgência.
2. **~20% de conflito entre as regras.** 586 abstracts disparam marcadores dos
   dois grupos, o equivalente a 19,7% do conjunto rotulável — as regras se
   contradizem em um a cada cinco casos que conseguiriam rotular.
3. **Viés de gênero textual.** Termos como `acute` e `mortality` em um abstract
   científico descrevem a população estudada, não a urgência de um paciente
   individual. Um abstract sobre epidemiologia de infarto é "sobre" alta
   mortalidade sem que exista qualquer paciente a triar.

---

## 6. Uso pretendido e limitações

**Uso pretendido:** demonstração acadêmica de um ciclo de vida de modelo em
produção (Tech Challenge FIAP Fase 3). Classificação de abstracts médicos em
inglês por área clínica.

**Limitações:**

- Não é um modelo de triagem de urgência e **não deve ser usado para decisão
  clínica** de qualquer natureza.
- Domínio de origem: abstracts científicos publicados. Laudos hospitalares
  reais têm vocabulário, abreviações e estrutura distintos — desempenho fora
  do domínio não foi medido.
- Corpus exclusivamente em inglês.
- **Cobertura da distribuição.** 20,7% dos abstracts do train são descartados
  por rótulo ambíguo (seção 3.1). O modelo não é treinado nem avaliado nesse
  recorte — que é real e provavelmente o mais difícil do corpus (abstracts que
  tocam mais de um sistema clínico). Em produção esses casos existem e serão
  classificados assim mesmo, com desempenho não medido.
- Desvio de prior entre train e holdout documentado na seção 3.3.

---

## 7. Reprodução

```bash
# preparação (estratégia de rótulo default: unambiguous)
uv run python -m src.data.prepare                          # -> data/processed/
uv run python -m src.data.prepare --label-strategy majority  # estratégia anterior

# baseline
uv run python -m src.models.baseline  # -> models/baseline.joblib, reports/baseline_metrics.json

# experimento que motivou o default (seção 3.2)
uv run python -m src.experiments.label_strategy_comparison  # -> reports/label_strategy_comparison.json
```

`data/processed/`, `models/*.joblib` e os dados brutos não são versionados. O
`prepare` baixa o corpus sozinho (`src/data/download.py`) na revisão fixa
`3ad6e168`, conferindo o sha256 de `train.csv` e `test.csv` — logo um clone
limpo reproduz os mesmos splits e as mesmas métricas sem nenhum passo manual.
Se os arquivos já existirem e o checksum bater, nada é baixado de novo.

Baseline (TF-IDF 1–2 gramas, `min_df=2`, `max_features=50000`, stopwords em
inglês + `LogisticRegression(class_weight='balanced')`), treinado nos 5.634
abstracts de rótulo único:

| Split | Acurácia | Macro-F1 |
|---|---|---|
| validação (n=995) | 0,797 | 0,792 |
| **test (holdout, n=2.657)** | **0,670** | **0,671** |

F1 por classe no test: 0,764 (neoplasms), 0,644 (digestive), 0,647 (nervous),
0,750 (cardiovascular), 0,549 (general pathological). A classe 5 segue sendo o
gargalo, pelo desvio de prior da seção 3.3.

Métricas completas, relatório por classe e matriz de confusão em
`reports/baseline_metrics.json`.
