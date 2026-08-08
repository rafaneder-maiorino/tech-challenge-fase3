# Otimização — Etapa 4, parte 1

Conversão do pipeline scikit-learn para ONNX, quantização dinâmica e um segundo
runtime de serving que não carrega scikit-learn nem scipy.

O ganho de **inferência** era esperado como pequeno e não é medido aqui — isso é
a parte 2. O resultado desta parte é o **corte da imagem**: 556 MB → 409 MB.

---

## 1. Resultado em uma tabela

| O que | Antes | Depois | Variação |
|---|---|---|---|
| Artefato do modelo | 3,96 MiB (joblib) | 1,59 MiB (ONNX quantizado) | **−59,7%** |
| Imagem de serving | 556 MB | 409 MB | **−147 MB (−26,4%)** |
| Ambiente Python na imagem | 250 MB | 135 MB | **−46%** |
| macro-F1 no test set | 0,670673 | 0,672040 | +0,0014 (ruído) |

Nenhum backend ficou abaixo do quality gate de 0,62 herdado da DAG de retreino.

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

## 10. O que fica para a parte 2

Nada de latência foi medido aqui, de propósito. Os dois runtimes ficam prontos
para serem comparados lado a lado:

- duas imagens que sobem em portas diferentes e reportam `model_backend` em
  `/health`
- três artefatos (`joblib`, `onnx`, `onnx` quantizado), todos selecionáveis por
  `MODEL_PATH`
- `scripts/benchmark_latency.py` da Etapa 1 e o baseline em
  `reports/latency_baseline.json` como referência

Uma decisão já tomada e que a parte 2 deve validar: o `OnnxBackend` fixa
`intra_op_num_threads=1` e `inter_op_num_threads=1`. Uma requisição carrega um
abstract só, então paralelismo intra-op tende a custar hand-off de thread em vez
de economizar tempo — mas isso é hipótese até ser medido.
