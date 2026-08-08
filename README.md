# Triagem Automática de Laudos Médicos — Tech Challenge FIAP Fase 3

Classificador de texto (NLP) que roteia laudos/abstracts médicos por área
clínica, servido por uma API REST em container Docker. O projeto exercita o
ciclo de vida completo do modelo em produção: preparação de dados, treino,
serving, CI/CD, orquestração, monitoração e otimização de latência.

**Status:** Etapas 1 a 3 concluídas — decisão arquitetural, API em Docker,
baseline de latência, CI/CD, DAG de retreino no Airflow e a stack de
monitoração com Prometheus e Grafana. Falta a Etapa 4.

| Entrega | Estado |
|---|---|
| Etapa 1 — Decisão arquitetural e API inicial | ✅ concluída |
| Etapa 2 — CI/CD (GitHub Actions) | ✅ concluída |
| Etapa 2 — DAG de retreino (Airflow) | ✅ concluída |
| Etapa 3 — Prometheus + Grafana via Docker Compose | ✅ concluída |
| Etapa 4 — Otimização de latência (ONNX/quantização) + vídeo | ⏳ pendente |

---

## 1. O que o modelo faz

Recebe o texto de um laudo/abstract em inglês e devolve uma das cinco áreas
clínicas do corpus:

| `label` | Área clínica |
|---|---|
| 1 | neoplasms (neoplasias) |
| 2 | digestive system diseases |
| 3 | nervous system diseases |
| 4 | cardiovascular diseases |
| 5 | general pathological conditions |

O corpus rotula **condição médica, não urgência clínica**. A decisão de manter
esse alvo — e não fabricar um rótulo de urgência por regras — está justificada
em [`docs/dataset-card.md`](docs/dataset-card.md), seção 4. O pipeline é
agnóstico ao alvo: depende apenas de uma coluna de texto e uma de classe.

**Desempenho no holdout (2.657 abstracts):** acurácia 0,670 / macro-F1 0,671.

---

## 2. Decisão arquitetural de deploy

> A Etapa 1 pede a decisão **textual** de arquitetura, não o deploy real em
> nuvem. Esta seção é essa decisão.

### 2.1 Batch ou tempo real?

O cenário é triagem no ponto de atendimento: o laudo chega, precisa ser
roteado, e o valor da classificação **decai a zero se ela chega depois que o
paciente já foi encaminhado**. Isso sozinho já elimina batch como caminho
principal — mas há três argumentos técnicos que reforçam.

**a) O comportamento computacional do modelo favorece tempo real (D1·A2).**
A aula 2 organiza a escolha pela característica do algoritmo: modelos lineares
tendem a APIs de baixa latência; modelos baseados em distância (KMeans e
similares) tendem a pipelines batch. O classificador aqui é TF-IDF +
regressão logística — custo de inferência **constante** por requisição, sem
dependência do volume total. Medido no container: **0,54 ms** de inferência,
1,39 ms fim a fim. Escala horizontalmente sem planejamento de capacidade.

**b) Não há training-serving skew a resolver (D4·A5).** O risco clássico do
serving em tempo real é a variável fácil de calcular em batch e cara de
calcular em milissegundos — o caso que motiva Feature Stores. Aqui **todas as
features saem do próprio texto da requisição**, pelo mesmo `TfidfVectorizer`
serializado junto com o modelo. Treino e inferência usam literalmente o mesmo
objeto. O argumento mais forte a favor de batch não se aplica.

**c) O custo do tempo real é baixo neste caso.** O artefato tem 4 MB e a
imagem 556 MB; uma réplica pequena atende o volume de um hospital de
referência com folga. O trade-off custo × latência que normalmente empurra
para batch aqui é marginal.

**Decisão: inferência síncrona em tempo real, via API REST em container.**

**Onde batch permanece:** o batch não é descartado, é realocado para onde de
fato ganha — reprocessamento histórico e **retreino periódico**, que rodam em
recursos efêmeros e agendados (Etapa 2, via Airflow). É a arquitetura mista
que as aulas descrevem: tempo real no caminho do paciente, batch no caminho do
modelo.

### 2.2 Comparativo entre provedores

Mesma lógica arquitetural nas três nuvens — execução, armazenamento e
distribuição de artefatos — com serviços equivalentes:

| Papel | AWS (D1·A3) | Azure (D1·A4) | GCP (D1·A5) |
|---|---|---|---|
| Registro de imagens | ECR | **ACR** | Artifact Registry |
| Tempo real, controle total | EC2 | Azure VM | Compute Engine |
| Tempo real, serverless | Lambda | **Container Apps** | Cloud Run |
| Batch / job agendado | AWS Batch | **Container Apps Jobs** | Cloud Run Jobs |
| ML gerenciado | SageMaker | Azure ML | Vertex AI |
| Armazenamento | S3 | Blob Storage | Cloud Storage |
| Observabilidade nativa | CloudWatch | Azure Monitor | Cloud Logging/Monitoring |
| Identidade sem segredo | IAM roles | Managed identities | Service accounts |

Como as três cobrem o mesmo desenho, a decisão se resolve por restrições
concretas do projeto, não por capacidade bruta.

**Descartado — serviço de ML gerenciado (SageMaker / Azure ML / Vertex AI).**
Sobra de ferramenta para um modelo de 4 MB. Registro, versionamento e
endpoints gerenciados resolvem problemas de governança que este projeto não
tem, ao custo de acoplamento ao provedor e de uma superfície operacional que
não caberia em 5 minutos de vídeo.

**Descartado — AWS Lambda.** O ambiente de serving tem 262 MB só de
dependências (scikit-learn arrasta scipy e numpy). Isso estoura o limite de
250 MB de pacote descompactado do Lambda, forçando o modo container image —
que funciona, mas anula a simplicidade que justificaria escolher Lambda.
Somado ao cold start com carregamento de modelo no startup, é a pior opção
para latência de triagem.

**Descartado — VM (EC2 / Azure VM / Compute Engine).** Latência previsível e
controle total, mas transfere para a equipe patching, escalonamento e
disponibilidade. Custo fixo 24×7 sem contrapartida: o modelo não precisa de
GPU nem de tuning de kernel.

**Finalistas — Cloud Run e Azure Container Apps.** Tecnicamente equivalentes
para este caso: ambos servem a mesma imagem OCI, escalam automaticamente,
expõem URL pública e leem a porta de uma variável de ambiente.

### 2.3 Escolha: Azure Container Apps

1. **A mesma imagem roda local e em produção.** O `Dockerfile` deste repo lê
   `${PORT}` e sobe em `0.0.0.0` — o contrato que o ACA (e o Cloud Run)
   espera. Não há reescrita nem empacotamento específico do provedor.
2. **ACR + managed identities fecham o CI/CD sem segredo estático (D1·A4).**
   O GitHub Actions da Etapa 2 faz build e push para o ACR autenticando por
   identidade gerenciada, sem credencial de longa duração no repositório.
3. **Container Apps Jobs cobre o retreino no mesmo ecossistema.** O batch da
   seção 2.1 vira um Job agendado usando a mesma imagem, sem introduzir um
   segundo runtime.
4. **Experiência prévia da equipe com ACA.** Critério legítimo de arquitetura:
   entre duas opções tecnicamente empatadas, a que o time já opera tem menor
   risco de execução e menor tempo até um deploy funcionando — exatamente o
   recurso escasso em um Tech Challenge.

**Ressalva assumida — escala a zero × latência.** O maior atrativo de custo do
ACA é escalar a zero, e isso conflita diretamente com o requisito de latência:
a primeira requisição após a escala a zero paga o cold start do container mais
o carregamento do modelo. Para triagem clínica isso é inaceitável em horário
de operação. A configuração pretendida é **`minReplicas: 1` durante o horário
hospitalar**, aceitando o custo de uma réplica sempre quente, com escala a zero
apenas em janelas de baixíssimo movimento. Trocar custo por previsibilidade de
latência é a decisão correta quando o consumidor é clínico.

---

## 3. Como executar

Pré-requisitos: **Docker Desktop** e **uv**. O Python 3.11 é provisionado pelo
próprio uv.

### 3.1 Instalar o uv

**macOS / Linux**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3.2 Caminho mais curto: subir a API em Docker

Não é preciso preparar dados nem treinar antes: o build baixa o corpus
(revisão fixa, com verificação de checksum) e treina o modelo dentro da
imagem.

**macOS / Linux**
```bash
docker build -t tc-fase3-api:latest .
docker run -d --name tc-api -p 8000:8000 tc-fase3-api:latest
curl http://localhost:8000/health
```

**Windows (PowerShell)**
```powershell
docker build -t tc-fase3-api:latest .
docker run -d --name tc-api -p 8000:8000 tc-fase3-api:latest
Invoke-RestMethod http://localhost:8000/health
```

Testando uma predição:

**macOS / Linux**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Acute myocardial infarction treated with primary coronary angioplasty."}'
```

**Windows (PowerShell)**
```powershell
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post `
  -ContentType "application/json" `
  -Body '{"text":"Acute myocardial infarction treated with primary coronary angioplasty."}'
```

Resposta:
```json
{
  "label": 4,
  "label_name": "cardiovascular diseases",
  "confidence": 0.9439,
  "latency_ms": 4.949
}
```

Documentação interativa (Swagger): <http://localhost:8000/docs>

Para parar e remover o container:
```bash
docker rm -f tc-api
```

### 3.3 Execução local, sem Docker

Os comandos são idênticos nos dois sistemas.

```bash
uv sync                                  # cria o ambiente e instala tudo
uv run python -m src.data.prepare        # baixa o corpus e gera data/processed/
uv run python -m src.models.baseline     # treina e salva models/baseline.joblib
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

`src.data.prepare` é idempotente: se os CSVs já existirem e o checksum bater,
nada é baixado de novo.

### 3.4 Testes

```bash
uv run pytest -q
```

18 testes cobrindo o contrato da API. Não exigem modelo treinado: a suíte
ajusta um classificador mínimo em memória e aponta `MODEL_PATH` para ele.

### 3.5 Benchmark de latência

Com o container **já em execução** (seção 3.2):

```bash
uv run python scripts/benchmark_latency.py
```

Grava `reports/latency_baseline.json`. Este é o baseline que a Etapa 4 vai
usar para comparar com o modelo otimizado, então o payload é fixo (sha256
registrado no relatório) e o JSON guarda hardware, versão do modelo e
protocolo da medição.

Parâmetros: `--requests` (200), `--warmup` (20, descartadas), `--url`,
`--output`.

### 3.6 Lint e formatação

```bash
uv run ruff check .          # regras de lint
uv run ruff format --check . # formatação (só verifica)
uv run ruff check --fix .    # corrige o que é seguro corrigir
uv run ruff format .         # aplica a formatação
```

São exatamente os dois comandos que o job `lint` do CI roda, então o que passa
aqui passa lá. A configuração está em `pyproject.toml`, em `[tool.ruff]`.

---

## 4. Resultados da Etapa 1

**Latência** — 200 requisições sequenciais contra o container, após 20 de
warmup descartadas. Apple M-series (arm64), macOS 15.5.

| Métrica | Fim a fim (cliente) | Inferência (servidor) |
|---|---|---|
| P50 | 1,39 ms | 0,54 ms |
| P95 | 1,79 ms | 0,60 ms |
| P99 | 1,92 ms | 0,62 ms |
| média | 1,46 ms | 0,55 ms |
| desvio | 0,16 ms | 0,03 ms |

A diferença entre as duas colunas (~0,85 ms) é overhead de HTTP e
serialização, não do modelo — informação relevante para a Etapa 4, porque
**otimizar o modelo só ataca a coluna da direita**. Zerar a inferência
cortaria menos de 40% da latência fim a fim.

**Imagem Docker** — 556 MB, três estágios, usuário não-root (uid 10001),
`HEALTHCHECK` que valida `model_loaded` e não apenas liveness. O ambiente de
serving instala só o necessário para inferir: `pandas`, `pyarrow` e o cliente
do Hugging Face ficam no grupo `train` e não entram no runtime.

---

## 5. CI/CD (Etapa 2)

Workflow em `.github/workflows/ci.yml`, disparado em **push para `main`** e em
**pull request** para `main`. Três jobs:

| Job | O que faz | Depende de |
|---|---|---|
| `lint` | `ruff check` + `ruff format --check` | — |
| `test` | os 18 testes do pytest, com cobertura no log | — |
| `build` | build da imagem Docker, sobe o container e valida `/health` e `/predict` | `lint` e `test` |

`lint` e `test` rodam em paralelo; `build` só começa quando os dois passam.
A ordem é intencional: build de imagem é o passo caro, e não faz sentido pagar
por ele para um código que nem passa no lint.

### 5.1 Por que três jobs e não um script só

O enunciado pede no mínimo duas automações (verificação de código e testes).
O terceiro job existe porque lint e teste, sozinhos, não provam que **a imagem
que vai para produção funciona** — a suíte de testes roda contra um modelo
sintético em memória, nunca contra o artefato real. O job `build` fecha essa
lacuna: constrói a imagem, sobe o container e exige que `/health` responda
`model_loaded: true` e que `/predict` devolva um rótulo válido.

A verificação de `/health` checa **readiness, não liveness**. A API sobe mesmo
quando o modelo falha ao carregar — de propósito, para conseguir reportar o
problema em vez de entrar em crash loop. Um teste que só checasse HTTP 200
passaria com o modelo quebrado.

### 5.2 O job `build` treina o modelo do zero — e por quê

O `Dockerfile` baixa o corpus e treina o baseline durante o build (estágio
`trainer`), então o job `build` no CI **treina de verdade**, sem atalho. A
decisão foi tomada medindo, não por suposição — primeiro localmente:

```bash
docker build --no-cache -t tc-fase3-api:bench .
```

| Etapa do build | Local (`--no-cache`) | Runner do CI |
|---|---|---|
| `uv sync` das dependências de serving | 9,7 s | 4,2 s |
| `uv sync` das dependências de treino | 18,3 s | 5,5 s |
| **download do corpus + prepare + treino** | **12,6 s** | **15,3 s** |
| exportação da imagem | 5,0 s | 1,5 s |
| **build completo** | **~39 s** | **~27 s** |

O runner sai na frente nos `uv sync` porque tem link muito mais rápido para o
PyPI; no treino, que é CPU, a máquina local ganha. No fim o build inteiro leva
**27 s no CI**, e o job `build` fecha em 38 s contando checkout e smoke test.

**Não há motivo para pular o treino**, e pular seria pior: o job deixaria de
cobrir justamente o caminho de `data/prepare.py` e `models/baseline.py`, que a
suíte de testes não exercita.

Esse número é pequeno porque o modelo é um TF-IDF + regressão logística sobre
~14 mil abstracts. A conclusão **não** se transfere para um modelo maior: se o
treino passasse da casa dos minutos, o caminho seria treinar fora do build
(job agendado ou DAG do Airflow), publicar o artefato em um registry e fazer o
`Dockerfile` apenas baixá-lo — que é exatamente o desenho previsto para a
etapa de retreino.

### 5.3 Cache

O cache de dependências fica em `astral-sh/setup-uv` com `enable-cache: true`,
com chave em `uv.lock`: enquanto o lock não muda, `lint` e `test` reaproveitam
os pacotes já baixados. Cada job usa um `cache-suffix` próprio — rodam em
paralelo e, com a mesma chave, disputam a reserva do cache (um dos dois falha
ao salvar e emite warning); além disso instalam conjuntos diferentes.

O build da imagem **não** usa cache de camadas, deliberadamente. O ganho seria
pequeno (27 s a frio) e o risco é real: uma camada de treino em cache faria o
job publicar um modelo velho e ainda assim reportar sucesso, que é exatamente
o tipo de falha silenciosa que este job existe para pegar.

Tempo total do workflow: `lint` 10 s, `test` 15 s (em paralelo), `build` 38 s.

### 5.4 Cobertura

O job `test` reporta cobertura de `src/api` (97%), não do `src/` inteiro. A
suíte é de **contrato de API**, não de qualidade de modelo — medir contra
`src/` inteiro produziria ~17%, um número que diz mais sobre o escopo da suíte
do que sobre o que ela cobre. Os módulos de dados e treino são exercitados de
ponta a ponta pelo job `build`, que roda o pipeline completo dentro da imagem.

Um dos 18 testes (`test_real_baseline_model_serves_predictions`) é pulado no
CI: ele exige `models/baseline.joblib`, que não é versionado. No CI o placar é
**17 passaram, 1 pulado**; localmente, com o modelo treinado, passam os 18.

---

## 6. Monitoração (Etapa 3)

Stack de observabilidade em três serviços numa rede compartilhada: a API
instrumentada, o Prometheus coletando e o Grafana com datasource e dashboard
provisionados por arquivo. Independente da stack do Airflow.

```bash
docker compose -f docker-compose.monitoring.yml up -d --build
uv run python scripts/generate_load.py --duration 180
```

| Serviço | URL | Imagem |
|---|---|---|
| API | <http://localhost:8000> | build local do `Dockerfile` |
| Prometheus | <http://localhost:9090> | `prom/prometheus:v3.13.2` |
| Grafana | <http://localhost:3000> | `grafana/grafana:13.1.3` |

O Grafana abre direto no dashboard, com acesso anônimo em papel `Viewer` — é uma
stack local de avaliação, não expor assim fora de `localhost`.

Métricas expostas em `GET /metrics`:

| Métrica | Tipo | O que responde |
|---|---|---|
| `api_requests_total` | Counter | tráfego por rota e status HTTP |
| `api_request_duration_seconds` | Histogram | latência ponta a ponta do handler |
| `api_predictions_total` | Counter | predições por classe (1 a 5) |
| `api_model_loaded` | Gauge | serviço apto a servir, ou só no ar |

As duas últimas são observabilidade de **modelo**, não de HTTP.
`api_model_loaded` existe por causa do achado da Etapa 1: a API sobe mesmo sem
conseguir carregar o artefato, para que `/health` reporte o problema em vez de o
container entrar em crash loop — e nesse estado ela responde `503` em todo
`/predict` enquanto qualquer check de liveness a considera saudável.

O ajuste não trivial foi a calibração dos buckets do histograma, que mudou
depois da primeira medição sob carga real: o número de 0,55 ms da Etapa 1 é
inferência sequencial, mas sob concorrência a latência ponta a ponta fica em
~3,8 ms porque o handler é síncrono e roda no threadpool do Starlette. Os dois
regimes precisam de fronteiras próprias.

Detalhes, queries PromQL e o snapshot do dashboard populado:
[`docs/monitoring-setup.md`](docs/monitoring-setup.md) e
[`docs/monitoring-dashboard-snapshot.md`](docs/monitoring-dashboard-snapshot.md).

---

## 7. Estrutura do projeto

```
.github/workflows/
  ci.yml                    # CI: lint, testes e build da imagem (Etapa 2)
src/
  labels.py                 # vocabulário de classes, sem dependências pesadas
  api/main.py               # FastAPI: /predict, /health e /metrics
  api/metrics.py            # instrumentação Prometheus (Etapa 3)
  data/download.py          # download do corpus (revisão fixa + checksum)
  data/prepare.py           # dedup, vazamento, split estratificado
  models/baseline.py        # TF-IDF + LogisticRegression
  experiments/              # comparação de estratégias de rótulo
airflow/                    # stack do Airflow, DAG de retreino (Etapa 2)
docker-compose.monitoring.yml   # API + Prometheus + Grafana (Etapa 3)
monitoring/
  prometheus.yml            # scrape da API
  grafana/provisioning/     # datasource e provider de dashboards
  grafana/dashboards/       # dashboard versionado, 8 painéis
scripts/
  benchmark_latency.py      # baseline de latência da Etapa 1
  generate_load.py          # tráfego misto para popular os painéis
  inspect_*.py              # EDA exploratória
tests/                      # testes de contrato da API e das métricas
docs/
  dataset-card.md           # fonte, EDA, decisões e limitações do dataset
  airflow-setup.md          # DAG de retreino e quality gate
  monitoring-setup.md       # stack de observabilidade e queries PromQL
  monitoring-dashboard-snapshot.md   # dashboard populado, valores medidos
  _conhecimento/            # transcrições das aulas
reports/                    # métricas do modelo e de latência
```

Dados brutos, `data/processed/` e `models/*.joblib` não são versionados — são
regenerados pelos comandos da seção 3.3. A justificativa da escolha (baixar
com revisão fixa em vez de versionar parquet) está na seção 7 do dataset card.

---

## 8. Referências

- Enunciado e critérios: [`docs/_conhecimento/07-tech-challenge-fase3.md`](docs/_conhecimento/07-tech-challenge-fase3.md)
- Mapa de aulas por critério: [`docs/_conhecimento/99-MAPA-TECH-CHALLENGE.md`](docs/_conhecimento/99-MAPA-TECH-CHALLENGE.md)
- Deploy em nuvem (D1·A1–A6): [`docs/_conhecimento/01-deploy-em-nuvem.md`](docs/_conhecimento/01-deploy-em-nuvem.md)
- Dataset: [`docs/dataset-card.md`](docs/dataset-card.md) · corpus
  [`123rc/medical_text`](https://huggingface.co/datasets/123rc/medical_text) (Apache 2.0)
