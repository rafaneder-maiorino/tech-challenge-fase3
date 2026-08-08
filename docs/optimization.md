# Otimização — Etapa 4, partes 1 e 2

Conversão do pipeline scikit-learn para ONNX, quantização dinâmica, um segundo
runtime de serving que não carrega scikit-learn nem scipy, e o comparativo de
latência entre os dois.

As seções 1 a 9 cobrem a parte 1 (conversão, quantização e imagem); as seções
10 a 12 cobrem a parte 2 (medição).

---

## 1. Resultado em uma tabela

| O que | Antes (sklearn) | Depois (ONNX quantizado) | Variação |
|---|---|---|---|
| **Inferência pura, P50** | 0,339 ms | 0,114 ms | **−66,5%** |
| Fim a fim sequencial, P50 | 2,051 ms | 1,722 ms | −16,0% |
| Fim a fim concorrente, P50 | 9,232 ms | 5,445 ms | −41,0% |
| Fim a fim concorrente, P99 | 24,589 ms | 53,395 ms | **+117,2%** |
| Throughput concorrente | 794,6 rps | 859,7 rps | +8,2% |
| Artefato do modelo | 3,96 MiB (joblib) | 1,59 MiB | **−59,7%** |
| Imagem de serving | 556 MB | 409 MB | **−147 MB (−26,4%)** |
| Ambiente Python na imagem | 250 MB | 135 MB | **−46%** |
| macro-F1 no test set | 0,670673 | 0,672040 | +0,0014 (ruído) |

Nenhum backend ficou abaixo do quality gate de 0,62 herdado da DAG de retreino.

Duas leituras que a tabela exige e a seção 10 detalha. Primeiro, o ganho de
inferência é grande (−66%) mas **diluído** no fim a fim, porque HTTP,
validação e fila do threadpool não são tocados pela otimização. Segundo, o P99
sob concorrência **piorou** de forma reprodutível — é o achado desconfortável
desta etapa e está documentado como tal, não escondido.

E a ressalva que atravessa tudo: os dois backends não computam exatamente a
mesma função (seção 3.2). É um comparativo entre dois modelos ligeiramente
diferentes, não o mesmo modelo em dois runtimes.

---

## 2. Conversão para ONNX

`src/models/export_onnx.py` carrega `models/baseline.joblib` e converte o
pipeline inteiro — `TfidfVectorizer` + `LogisticRegression` — com `skl2onnx`.

```bash
uv run python -m src.models.export_onnx
```

Três escolhas de conversão merecem registro.

**`target_opset=18`.** Fixado em vez de deixar o conversor escolher, para o
grafo ser reprodutível entre máquinas.

**`zipmap=False`.** Sem isso o classificador devolve uma lista de dicionários
`{classe: probabilidade}` em vez de um tensor de floats. O tensor é mais rápido
e muito mais simples de consumir no backend.

**`black_op={"LinearClassifier"}`.** Esta é a menos óbvia e está explicada na
seção 4 — sem ela, a quantização não faz absolutamente nada.

O grafo resultante usa três domínios: `ai.onnx` (StringNormalizer,
TfIdfVectorizer, MatMul), `ai.onnx.ml` (Normalizer, ArrayFeatureExtractor) e
`com.microsoft` (Tokenizer). O último significa que o artefato **exige
onnxruntime** — não roda em qualquer runtime ONNX genérico. Para este projeto
isso é irrelevante, mas é uma amarra real.

---

## 3. Equivalência numérica: 0,79% das predições divergem

Esta foi a parte que consumiu mais tempo, e o número não é zero.

Rodando os 2.657 abstracts do split de teste pelos dois caminhos:

| Backend | acurácia | macro-F1 | divergências vs sklearn | maior diferença de probabilidade |
|---|---|---|---|---|
| sklearn | 0,669928 | 0,670673 | — | — |
| ONNX | 0,669928 | 0,671332 | **21 (0,79%)** | 0,0802 |
| ONNX quantizado | 0,670681 | 0,672040 | 24 (0,90%) | 0,0783 |

Números completos em [`reports/onnx_equivalence.json`](../reports/onnx_equivalence.json).

A primeira conversão divergia em **30** predições, com diferença de
probabilidade de até **0,13** — grande demais para ser arredondamento de
float32. Investigando o grafo, apareceram duas causas distintas.

### 3.1 Tokens de um caractere (corrigido)

O `token_pattern` padrão do scikit-learn é `(?u)\b\w\w+\b`: exige **no mínimo
dois** caracteres de palavra. O `Tokenizer` que o skl2onnx emite vem com
`mincharnum=1`, então mantém tokens de um caractere que o sklearn descarta.

Isso não afeta só os unigramas. Um token a mais **desloca todo o fluxo** e,
portanto, todos os bigramas construídos a partir dele:

```
"vitamin a deficiency"
  sklearn: [vitamin, deficiency]     -> bigrama "vitamin deficiency"
  ONNX   : [vitamin, a, deficiency]  -> bigramas "vitamin a", "a deficiency"
```

O bigrama que está no vocabulário nunca é gerado. A correção é trivial —
`mincharnum=2` no grafo convertido — e está em `_align_tokenizer_with_sklearn`.
Efeito medido: **30 → 21 divergências**, maior diferença 0,13 → 0,080.

### 3.2 Stopwords antes dos n-gramas (não corrigível)

O que sobra tem uma causa única e sem solução dentro da especificação ONNX.

O `TfidfVectorizer` foi treinado com `stop_words="english"`. O sklearn remove as
stopwords **antes** de montar os n-gramas, então bigramas se formam entre
palavras que ficaram adjacentes só depois da remoção:

```
"the treatment of cancer"
  sklearn: [treatment, cancer]              -> bigrama "treatment cancer"  ✓ no vocabulário
  ONNX   : [the, treatment, of, cancer]     -> "the treatment", "treatment of", "of cancer"  ✗ nenhum
```

Os unigramas continuam corretos: uma stopword simplesmente não casa com nada em
`pool_strings` e é ignorada. O problema é exclusivo dos bigramas — e **71,1% do
vocabulário são bigramas**.

Quantificando sobre 400 documentos: das ocorrências de bigrama do vocabulário,
o ONNX perde **24,1%**. É muita coisa.

**Por que não dá para corrigir.** A tentativa natural é inserir um
`StringNormalizer` com o atributo `stopwords` entre o `Tokenizer` e o
`TfIdfVectorizer`. Foi tentado e o onnxruntime recusa carregar o modelo:

```
[ShapeInferenceError] Input shape must have either [C] or [1,C] dimensions where C > 0
```

O `StringNormalizer` exige que a dimensão seja **estaticamente conhecida e
positiva**. A saída de um `Tokenizer` tem tamanho que depende do dado — quantos
tokens o documento tem — e portanto é dinâmica. Fixar o batch em 1 não resolve:
a dimensão problemática é a de tokens, não a de batch. Não há como colocar
remoção de stopwords depois da tokenização em um grafo ONNX válido.

As alternativas foram descartadas:

- **`max_skip_count=1`** no `TfIdfVectorizer` geraria "treatment cancer" pulando
  "of", mas geraria também skip-gramas espúrios entre palavras não-stopword
  ("acute myocardial infarction" → "acute infarction"), que o sklearn nunca
  produz. Troca um erro por outro.
- **Regex negativo** no `Tokenizer` para não emitir stopwords: o RE2 usado pelo
  onnxruntime não suporta lookahead.
- **Retreinar sem `stop_words`**: resolveria, mas muda o modelo e invalidaria
  todas as métricas das etapas 1 a 3. Fora do escopo desta parte.

**Conclusão honesta:** o skl2onnx não consegue representar fielmente
`stop_words="english"` combinado com `ngram_range=(1,2)`. O impacto agregado é
pequeno — 0,79% das predições, macro-F1 praticamente igual — mas é uma diferença
de comportamento real, não ruído de precisão numérica, e está documentada aqui
em vez de escondida atrás de uma tolerância generosa.

Onde isso aparece na prática: textos curtos e densos em stopwords sofrem
proporcionalmente mais. É por isso que o teste
`test_real_onnx_model_agrees_with_the_sklearn_baseline` compara confiança com
tolerância de 0,10 e não de 0,01.

---

## 4. Quantização dinâmica: a versão ingênua não faz nada

O caminho óbvio é converter e chamar `quantize_dynamic`. Feito assim, o
resultado é:

```
onnx        : 2.671.005 bytes
onnx quant  : 2.671.197 bytes   (+0,01%)
```

192 bytes **a mais**. Zero nós alterados, zero inicializadores alterados. A
quantização foi um no-op silencioso — o tipo de coisa que passa despercebida se
só se olhar "rodou sem erro".

**A causa.** Por padrão o skl2onnx emite a regressão logística como um único
operador `LinearClassifier` do domínio `ai.onnx.ml`, que guarda os coeficientes
como **atributos do nó**. O `quantize_dynamic` só reescreve **inicializadores**
que alimentam `MatMul`/`Gemm`/`Conv`. Atributos são invisíveis para ele.

**A correção.** `black_op={"LinearClassifier"}` proíbe esse operador e força o
conversor a decompor o classificador em `MatMul + Add + Softmax`. Aí a matriz de
coeficientes 50000×5 vira um inicializador de verdade, alimentando um `MatMul`:

```
inicializadores, antes: shape_tensor, idfcst
inicializadores, depois: coef [50000,5], intercept, classes, shape_tensor, idfcst
```

E a quantização passa a funcionar:

```
decomposto        : 2.421.289 bytes (2,309 MiB)
decomposto+quant  : 1.672.168 bytes (1,595 MiB)   −30,9%
```

O `MatMul` vira `DynamicQuantizeLinear + MatMulInteger + Cast + Mul`, e `coef`
vira `coef_quantized` em UINT8 mais um par escala/zero-point.

**Um detalhe a mais:** a chamada exige
`extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT}`. Sem isso o
quantizador aborta, porque não consegue inferir o tipo do tensor que sai da
cadeia do `Tokenizer` — o operador vem do domínio `com.microsoft` e a inferência
de shape do ONNX não o conhece.

### Degradação de acurácia: nenhuma

| | macro-F1 | acurácia |
|---|---|---|
| ONNX | 0,671332 | 0,669928 |
| ONNX quantizado | 0,672040 | 0,670681 |

O macro-F1 **subiu** 0,0007. Isso não é a quantização melhorando o modelo — é
ruído: 5 predições mudaram entre os dois grafos (0,19%), e algumas calharam de
cair do lado certo. A maior diferença de probabilidade entre ONNX e ONNX
quantizado é 0,0123, uma ordem de grandeza menor que a diferença causada pelas
stopwords. Int8 sobre uma matriz de coeficientes de regressão logística é
praticamente lossless neste caso.

O gate de 0,62 não chegou perto de ser ameaçado — e agora é verificado de
verdade: `validate_backends` **levanta exceção** se qualquer backend ficar
abaixo dele, e essa validação roda dentro do build da imagem.

---

## 5. Tamanho dos artefatos

| Artefato | bytes | MiB | vs joblib |
|---|---|---|---|
| `models/baseline.joblib` | 4.152.756 | 3,960 | — |
| `models/model.onnx` | 2.421.289 | 2,309 | **−41,7%** |
| `models/model.quantized.onnx` | 1.672.168 | 1,595 | **−59,7%** |

Ambos os grafos vão para a imagem: juntos são ~4 MB, irrelevante perto da
camada do interpretador, e a parte 2 precisa dos dois para comparar lado a lado
sem rebuild.

---

## 6. Backend selecionável na API

`src/api/backends.py` define dois backends atrás de um `Protocol` de um método
só. A seleção é por variável de ambiente:

```bash
MODEL_BACKEND=sklearn  uv run uvicorn src.api.main:app   # padrão
MODEL_BACKEND=onnx     uv run uvicorn src.api.main:app
```

Cada backend tem seu artefato padrão, então trocar de backend sem mexer em
`MODEL_PATH` já resolve o arquivo certo:

| `MODEL_BACKEND` | artefato padrão | engine |
|---|---|---|
| `sklearn` (padrão) | `models/baseline.joblib` | joblib + scikit-learn |
| `onnx` | `models/model.onnx` | onnxruntime |

`MODEL_PATH` continua tendo precedência quando definido — é assim que se aponta
o backend ONNX para `model.quantized.onnx`.

**A regra que sustenta tudo:** cada backend importa seu engine **dentro do
próprio construtor**. Um `import joblib` no topo de `backends.py` tornaria o
arquivo não-importável na imagem ONNX, que não tem joblib, e a separação
inteira perderia o sentido. `numpy` é a única exceção, por ser dependência
obrigatória tanto do scikit-learn quanto do onnxruntime.

Isso é testado, e não só afirmado:
`test_onnx_backend_does_not_import_scikit_learn` roda um **subprocesso** que
carrega o backend ONNX, faz uma predição e falha se `sklearn`, `scipy`,
`joblib` ou `pandas` aparecerem em `sys.modules`. Tem de ser subprocesso: a
sessão de testes já importou scikit-learn para montar o modelo de apoio, então
uma verificação in-process passaria independentemente do que o backend faz.

`/health` passou a reportar `model_backend`, para a parte 2 provar qual engine
respondeu em vez de inferir pelo container que foi iniciado.

---

## 7. A imagem ONNX

`Dockerfile.onnx` — mesmo aplicativo, mesmo corpus, mesmo modelo treinado. A
diferença está no que sobra na última camada.

```bash
docker build -f Dockerfile.onnx -t tc-fase3-api:onnx .
```

O estágio de treino **continua precisando** do scikit-learn completo: o modelo é
ajustado com sklearn e só depois exportado. Esse custo fica num estágio
descartado — não é o que embarca.

O flag que decide tudo está no estágio de dependências:

```dockerfile
RUN uv sync --frozen --no-default-groups --extra onnx
```

Sem `--extra sklearn`, o scikit-learn e o scipy não são sequer baixados.

### O que saiu

| | sklearn | onnx |
|---|---|---|
| Imagem (`docker images`) | 556 MB | **409 MB** |
| Imagem (comprimida, o que trafega no pull) | 119,1 MB | **92,6 MB** |
| `/opt/venv` | 250 MB | **135 MB** |

Pacotes que desapareceram do runtime: `sklearn`, `scikit_learn.libs`, `scipy`,
`scipy.libs`, `joblib`, `narwhals`, `threadpoolctl`.
Pacotes que entraram: `onnxruntime`, `flatbuffers`, `google` (protobuf),
`packaging`. `numpy` permanece nos dois.

O piso é a imagem base `python:3.11-slim` (~150 MB) mais o próprio
onnxruntime. Ir abaixo disso exigiria trocar a base (distroless, Alpine) ou o
pacote `onnxruntime` por uma build reduzida — fora do escopo desta parte.

### O bug do locale

A primeira build da imagem ONNX **falhou**, e só dentro do container:

```
Failed to construct locale with name:en_US.UTF-8:
locale::facet::_S_create_c_locale name not valid
```

O `StringNormalizer` do onnxruntime constrói um `std::locale` na inicialização
da sessão, com padrão `en_US.UTF-8`. O `python:3.11-slim` não traz definições de
locale, então a sessão estoura antes de servir qualquer requisição. No macOS
nunca apareceu.

Instalar `locales` na imagem resolveria e custaria dezenas de MB — exatamente o
que esta etapa está tentando reduzir. A correção foi fixar o atributo `locale`
do nó em `"C"`, que existe em qualquer sistema
(`_pin_string_normalizer_locale`). A única coisa que muda é o case folding fora
do ASCII, e o regex do `Tokenizer` logo em seguida é `[a-zA-Z0-9_]+`, que
descarta não-ASCII de qualquer forma. A equivalência medida ficou idêntica antes
e depois da mudança.

---

## 8. Dependências: extras por backend

O engine de inferência deixou de ser dependência base e virou **extra**:

```toml
[project.optional-dependencies]
sklearn = ["joblib>=1.5.3", "scikit-learn>=1.9.0"]
onnx = ["onnxruntime>=1.24.0"]
```

Consequência prática: **`uv sync` puro não instala backend nenhum** e a suíte de
testes não roda. O uv 0.11 não tem `default-extras`, então não há como declarar
"instale ambos por padrão" no `pyproject.toml`. Desenvolvimento local e CI usam:

```bash
uv sync --all-extras
```

As imagens usam `--extra sklearn` ou `--extra onnx`. Se o backend pedido não
tiver seu engine instalado, `load_backend` levanta `ImportError` dizendo qual
extra falta.

---

## 9. Como reproduzir

```bash
uv sync --all-extras

# 1. corpus e modelo (se ainda não existirem)
uv run python -m src.data.prepare
uv run python -m src.models.baseline

# 2. converter, quantizar e validar os três backends
uv run python -m src.models.export_onnx

# 3. testes: a suíte de contrato roda contra os dois backends
uv run pytest -v

# 4. as duas imagens
docker build -t tc-fase3-api:sklearn .
docker build -f Dockerfile.onnx -t tc-fase3-api:onnx .
docker images tc-fase3-api
```

O CI faz o mesmo: constrói as duas imagens no mesmo runner, confere que os dois
backends classificam o mesmo texto com o mesmo label, publica os tamanhos no
resumo da run e falha se `sklearn`, `scipy`, `joblib` ou `pandas` reaparecerem
no runtime ONNX.

---


## 10. Comparativo de latência (parte 2)

A otimização mexe em **uma** coisa: quanto tempo leva para transformar uma
string num vetor de probabilidades. Medir só o fim a fim enterraria isso dentro
de parsing de requisição, fila do threadpool e serialização de JSON, e
reportaria um "ganho" que fala mais da stack web do que do modelo. Por isso a
medição separa três níveis.

`scripts/compare_backends.py` reaproveita o payload fixo e a definição de
percentil de `benchmark_latency.py`, então todo número aqui é diretamente
comparável com `reports/latency_baseline.json` da Etapa 1 — não apenas
parecido. Resultado completo em
[`reports/backend_comparison.json`](../reports/backend_comparison.json).

**Condições:** macOS arm64, 8 CPUs, Python 3.11.15, payload de 1.056 caracteres
(sha256 `51b3174a…`), 50 requisições de warmup descartadas em cada nível,
2.000 iterações de inferência pura em 3 rodadas alternadas, 500 requisições
sequenciais, 2.000 requisições a concorrência 8.

### Nível A — inferência pura, sem HTTP

`backend.predict()` chamado direto no processo. É o que a técnica controla.

| | sklearn | ONNX quantizado | delta |
|---|---|---|---|
| P50 | 0,339 ms | **0,114 ms** | **−66,5%** |
| P95 | 0,366 ms | **0,132 ms** | **−64,0%** |
| P99 | 0,380 ms | **0,138 ms** | **−63,6%** |
| média | 0,343 ms | **0,116 ms** | **−66,1%** |

Aqui o ganho é grande e estável: quatro execuções independentes deram entre
−66,5% e −66,9% na mediana. É o número honesto da otimização.

### Nível B — fim a fim, sequencial

Uma requisição por vez, o protocolo da Etapa 1. A tabela de baixo é o mesmo
conjunto de requisições, mas usando o `latency_ms` que o servidor reporta, que
cronometra só a chamada `predict()` dentro do handler.

| | sklearn | ONNX quantizado | delta |
|---|---|---|---|
| **cliente (fim a fim)** | | | |
| P50 | 2,051 ms | 1,722 ms | −16,0% |
| P95 | 2,500 ms | 2,110 ms | −15,6% |
| P99 | 2,845 ms | 2,382 ms | −16,3% |
| **servidor (só inferência)** | | | |
| P50 | 0,656 ms | 0,323 ms | **−50,8%** |
| P95 | 0,771 ms | 0,407 ms | **−47,2%** |
| P99 | 0,830 ms | 0,421 ms | **−49,3%** |

A diluição está explícita: os mesmos 0,33 ms economizados valem −51% quando
medidos na inferência e −16% quando medidos no cliente. O resto da requisição —
~1,4 ms de HTTP, validação Pydantic e serialização — não mudou, porque não é o
que a otimização toca.

### Nível C — fim a fim, concorrência 8

O cenário da Etapa 3, onde a fila do threadpool do Starlette domina.

| | sklearn | ONNX quantizado | delta |
|---|---|---|---|
| **cliente (fim a fim)** | | | |
| P50 | 9,232 ms | 5,445 ms | −41,0% |
| P95 | 16,868 ms | 13,569 ms | −19,6% |
| P99 | 24,589 ms | 53,395 ms | **+117,2%** |
| média | 10,044 ms | 9,283 ms | −7,6% |
| **servidor (só inferência)** | | | |
| P50 | 0,517 ms | 0,226 ms | −56,3% |
| P95 | 0,590 ms | 0,561 ms | −4,9% |
| P99 | 0,683 ms | 1,064 ms | **+55,8%** |

**Throughput** (requisições concluídas por segundo, com a fila sempre cheia):

| | sklearn | ONNX quantizado | delta |
|---|---|---|---|
| sequencial | 480,5 rps | 561,4 rps | **+16,8%** |
| concorrente | 794,6 rps | 859,7 rps | **+8,2%** |

### O P99 sob carga piorou — e não é ruído

A mediana melhora 41%, mas o P99 do ONNX é consistentemente **pior**. Em quatro
execuções o P99 do cliente ficou em +44%, +78%, +87% e +117%: a direção se
repete sempre, só a magnitude varia. O mesmo aparece no P99 de inferência pura
do servidor (+42% a +90%).

Hipóteses testadas e descartadas, medindo em processo com 8 threads contra a
mesma sessão:

| Configuração | P50 | P99 |
|---|---|---|
| sessão única, arena ON (atual) | 0,248 ms | 0,608 ms |
| sessão única, arena OFF | 0,230 ms | 0,739 ms |
| sessão por thread | 0,192 ms | 1,054 ms |

Nenhuma corrige a cauda; sessão por thread piora. `intra_op_num_threads` também
não é o fator — ver seção 10.1.

A explicação mais coerente com os dados é a troca clássica de saturação. O
onnxruntime libera o GIL e faz trabalho de verdade em C++ paralelamente, então
entrega **mais throughput** (+8,2%) e mantém a máquina mais quente; o caminho
do sklearn é dominado por trabalho Python sob o GIL, o que limita a
concorrência efetiva e produz uma distribuição mais lenta porém mais estreita
(P99 de inferência de 0,683 ms contra 1,064 ms). Some-se que o gerador de carga
roda na **mesma máquina de 8 CPUs** que os containers: com 8 requisições em
voo, cliente e servidor disputam CPU, e o backend mais rápido empurra o sistema
mais perto do limite.

Isso é uma interpretação, não uma causa comprovada. O que está comprovado é o
comportamento: **ONNX entrega mediana e throughput melhores em troca de uma
cauda pior sob saturação.** Para triagem clínica, onde o P99 é o que define a
experiência do pior caso, isso merece uma decisão explícita antes de um deploy
com concorrência alta — e um teste de carga em máquina separada do gerador,
que esta medição não tem.

### 10.1 `intra_op_num_threads=1` — a hipótese da parte 1 estava certa pelo motivo errado

A parte 1 fixou `intra_op_num_threads=1` supondo que "paralelismo intra-op
tende a custar hand-off de thread". Medido:

| | SEQ P50 | SEQ P99 | CONC(8) P50 | CONC(8) P99 |
|---|---|---|---|---|
| `intra=1` | 0,112 ms | 0,136 ms | 0,261 ms | 0,623 ms |
| `intra=2` | 0,112 ms | 0,136 ms | 0,246 ms | 0,616 ms |
| `intra=4` | 0,112 ms | 0,135 ms | 0,242 ms | 0,597 ms |
| padrão do ort | 0,112 ms | 0,134 ms | 0,245 ms | 0,649 ms |

A latência sequencial é **idêntica** em todas as configurações, e as diferenças
sob concorrência estão dentro do ruído. Não há hand-off custando tempo: o grafo
é pequeno demais para haver o que paralelizar dentro de uma execução. Manter
`1` continua sendo a escolha certa — evita que cada sessão suba um pool de
threads ocioso — mas **não é uma alavanca de latência**, e a justificativa
escrita na parte 1 estava errada.

### Consolidado

| Dimensão | sklearn | ONNX quantizado | delta |
|---|---|---|---|
| Inferência pura, P50 | 0,339 ms | 0,114 ms | **−66,5%** |
| Fim a fim sequencial, P50 | 2,051 ms | 1,722 ms | −16,0% |
| Fim a fim concorrente, P50 | 9,232 ms | 5,445 ms | −41,0% |
| Fim a fim concorrente, P99 | 24,589 ms | 53,395 ms | **+117,2%** |
| Throughput concorrente | 794,6 rps | 859,7 rps | +8,2% |
| Artefato | 3,960 MiB | 1,595 MiB | **−59,7%** |
| Imagem | 556 MB | 409 MB | **−26,4%** |

**Onde a otimização ganhou:** inferência pura (−66%), tamanho de artefato
(−60%), tamanho de imagem (−26%) e throughput (+8% a +17%).

**Onde não mudou nada:** o custo de HTTP, validação e serialização — ~1,4 ms
por requisição no nível B, idêntico nos dois backends. E a fila do threadpool,
que no nível C responde por praticamente toda a latência (9,2 ms de cliente
contra 0,5 ms de inferência no sklearn). Nenhuma das duas é tocada por
ONNX ou quantização, e nenhuma quantidade de otimização de modelo as reduziria.

**Onde piorou:** a cauda sob saturação.

### A ressalva que a tabela não mostra

Os dois backends **não computam a mesma função**. O skl2onnx não consegue
aplicar `stop_words` antes da construção de n-gramas (seção 3.2), então o grafo
ONNX perde ~24% das ocorrências de bigrama do vocabulário e 0,79% das predições
do test set diferem. A qualidade agregada é indistinguível — macro-F1 0,6707
contra 0,6720, ambas muito acima do gate de 0,62 — mas isto é um comparativo de
latência entre **dois modelos ligeiramente diferentes**, não o mesmo modelo em
dois runtimes. A ressalva está gravada no próprio
`reports/backend_comparison.json`, em `equivalence_caveat`, para não se perder
de vista quando alguém ler só os números.

---

## 11. Como reproduzir o comparativo

O gerador de carga roda na mesma máquina que os containers, então qualquer
outro processo pesado contamina a medição. A stack de monitoração da Etapa 3
(o Prometheus raspa a API a cada 5 s) e a do Airflow precisam estar paradas:

```bash
docker compose -f docker-compose.monitoring.yml down
docker stop airflow-airflow-scheduler-1 airflow-airflow-webserver-1 airflow-postgres-1
```

Sobe os dois backends lado a lado. O container ONNX é apontado para o grafo
**quantizado**, que já vem na imagem — não é preciso rebuild:

```bash
docker run -d --name tc-bench-sk --no-healthcheck -p 8000:8000 tc-fase3-api:sklearn
docker run -d --name tc-bench-onnx --no-healthcheck -p 8001:8000 \
  -e MODEL_PATH=/app/models/model.quantized.onnx tc-fase3-api:onnx

uv run python scripts/compare_backends.py
```

O script confere via `/health` que cada porta serve o backend e o artefato
esperados antes de medir, e aborta se não servir — medir o container errado é
um erro silencioso fácil de cometer.

**Por que os dois containers ficam de pé o tempo todo, e não um de cada vez:**
derrubar e subir entre as medições introduz variação (page cache, frequência de
CPU, estado da VM do Docker) exatamente entre os dois números que se quer
comparar. Manter os dois no ar elimina isso. A carga, porém, é aplicada a **um
backend por vez** — se os dois fossem saturados juntos, o nível C mediria
disputa de escalonador, não o backend. O container ocioso custa
aproximadamente nada, e o healthcheck é desligado com `--no-healthcheck` para
que ele não suba um interpretador Python no meio de uma medição. Os níveis são
intercalados sklearn-depois-onnx para que as duas medições fiquem o mais
próximas possível no tempo.

Ao terminar:

```bash
docker rm -f tc-bench-sk tc-bench-onnx
docker start airflow-postgres-1 airflow-airflow-scheduler-1 airflow-airflow-webserver-1
```

---

## 12. O que fica para a parte 3

O vídeo de demonstração. A medição de latência e o corte de imagem estão
fechados; o que falta é apresentá-los.

Duas coisas que este comparativo deixou em aberto e que não são bloqueantes:

- **A cauda sob saturação** merece um teste de carga com o gerador em máquina
  separada dos containers, para separar contenção de CPU cliente/servidor do
  comportamento real do onnxruntime.
- **O handler síncrono.** O nível C mostra que a fila responde por ~95% da
  latência sob concorrência. Tornar `/predict` assíncrono, ou dimensionar o
  threadpool do Starlette, teria efeito muito maior sobre a latência fim a fim
  do que qualquer otimização de modelo — mas é outra etapa de trabalho, não
  esta.
