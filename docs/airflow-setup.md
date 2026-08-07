# Airflow — setup local (Etapa 2)

Stack do Apache Airflow que orquestra a pipeline de ingestão e treino do Tech
Challenge, e a DAG `train_pipeline` que roda em cima dela.

| Item | Valor |
| --- | --- |
| Versão do Airflow | `2.11.0` (base `apache/airflow:2.11.0-python3.11`) |
| Imagem usada | `tech-challenge-fase3/airflow:2.11.0` (build local, ver `airflow/Dockerfile`) |
| Executor | `LocalExecutor` |
| Metadata DB | `postgres:16-alpine` (sem porta exposta) |
| UI | http://localhost:8080 |
| Credenciais | definidas no seu `airflow/.env` (não versionado) |
| Containers | 3 de longa duração + 1 one-shot (`airflow-init`) |
| Arquitetura | arm64 nativo (sem emulação QEMU) |
| DAG | `train_pipeline` — `ingest` → `prepare` → `train` |

---

## Primeiro uso

O repositório é público, então **nenhum segredo está versionado**. O `.env` é
obrigatório: sem ele o `docker compose` falha com uma mensagem explícita
apontando para o `.env.example`.

```bash
cd airflow
cp .env.example .env
```

Depois abra o `.env` e preencha os três campos marcados como `<GERAR>`:

```bash
# FERNET_KEY — criptografa Connections e Variables
python3 -c "import base64,os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())"

# SECRET_KEY — assina os cookies de sessão da UI
python3 -c "import secrets; print(secrets.token_hex(32))"

# senha do admin
python3 -c "import secrets; print(secrets.token_urlsafe(12))"
```

Se o usuário admin já existir no banco, mudar `AIRFLOW_ADMIN_PASSWORD` no `.env`
**não** troca a senha — o `airflow-init` só cria o usuário quando ele não existe.
Para aplicar uma senha nova, ou troque pela UI (Security → List Users) ou recrie
o banco com `down -v`.

---

## Comandos

Todos rodados a partir do diretório `airflow/`:

```bash
cd airflow

# subir (a primeira vez precisa do --build para montar a imagem com as
# dependências de treino)
docker compose -f docker-compose.airflow.yml up -d --build

# subidas seguintes
docker compose -f docker-compose.airflow.yml up -d

# acompanhar o primeiro start (o airflow-init é o que demora)
docker compose -f docker-compose.airflow.yml logs -f airflow-init

# status / saúde
docker compose -f docker-compose.airflow.yml ps
curl -s http://localhost:8080/health | python3 -m json.tool

# derrubar (preserva o banco de metadados)
docker compose -f docker-compose.airflow.yml down

# derrubar e APAGAR o banco (reset total — refaz migrations e recria o usuário)
docker compose -f docker-compose.airflow.yml down -v
```

Acesso à UI: <http://localhost:8080>, com as credenciais do seu `.env`.

Rebuild da imagem só é necessário quando as versões no `airflow/Dockerfile`
mudarem. Editar a DAG ou o código de `src/` **não** exige rebuild: os dois
entram por volume.

---

## Decisão: LocalExecutor em vez de CeleryExecutor

**Escolhido: `LocalExecutor` + Postgres.**

O critério de avaliação do TC pede *"DAG funcional realizando ingestão e treino"* —
ou seja, correção da pipeline, não escalabilidade. As duas opções entregam
exatamente o mesmo resultado para esse critério; a diferença está no custo
operacional e na superfície de falha.

| | LocalExecutor | CeleryExecutor |
| --- | --- | --- |
| Containers | 3 (postgres, scheduler, webserver) | 6+ (add redis, worker, flower) |
| Paralelismo | subprocessos do scheduler, limitado pela máquina | horizontal, multi-nó |
| Peças que podem quebrar | Postgres | Postgres + Redis + broker + workers |
| RAM em idle | ~1,2 GB | ~2,5 GB |
| Ganho para o critério do TC | — | nenhum |

O CeleryExecutor só compensa quando é preciso escalar workers horizontalmente,
o que não é o caso: a pipeline roda tarefas sequenciais de ingestão e treino em
uma única máquina. Redis e workers seriam três serviços a mais para diagnosticar
quando algo falhasse, sem nenhum ganho no que está sendo avaliado.

O `LocalExecutor` ainda executa tasks em paralelo (via subprocessos), então
branches paralelas na DAG continuam funcionando.

Componentes deliberadamente **omitidos** da stack mínima:

- **`triggerer`** — só é necessário para *deferrable operators*, que uma DAG de
  ingestão/treino não usa. Se um dia for preciso, é um container a mais rodando
  `command: triggerer`.
- **`flower`** — UI de monitoramento do Celery; não se aplica ao LocalExecutor.

### Por que Airflow 2.11.0 e não 3.x

A versão mais recente disponível é a `3.3.0`. Ficamos na `2.11.0` (última da
linha 2.x) porque o Airflow 3 muda peças de infraestrutura relevantes justamente
para quem está montando a stack pela primeira vez: o `webserver` virou
`api-server`, o `dag-processor` passou a ser um componente obrigatório separado,
a autenticação passou para o `SimpleAuthManager` (com senha gerada em arquivo, em
vez de `airflow users create`) e é preciso configurar `JWT_SECRET`. Como o
objetivo declarado desta etapa é minimizar risco de infra, a 2.11 — que é o
caminho documentado na esmagadora maioria dos tutoriais e respostas de
Stack Overflow — é a escolha de menor atrito.

A tag foi fixada como `2.11.0-python3.11` (nunca `:latest`): o `python3.11` casa
com o `.python-version` do projeto, o que evita divergência de versão quando as
DAGs passarem a importar código de `src/`.

---

## Cuidados específicos deste ambiente (macOS / Apple Silicon)

### `AIRFLOW_UID` — por que 50000 e não `id -u`

O `docker-compose.yaml` oficial do Airflow instrui a rodar `echo -e "AIRFLOW_UID=$(id -u)" > .env`.
**Essa instrução é para Linux e não deve ser seguida no macOS.**

No Linux, bind mounts preservam o uid numérico: se o container roda como 50000, os
arquivos em `logs/` ficam pertencendo ao uid 50000 no host — que não existe — e o
usuário não consegue lê-los ou apagá-los. Daí a necessidade de casar o uid.

No macOS, o Docker Desktop roda os containers dentro de uma VM Linux e o
compartilhamento de arquivos (VirtioFS/gRPC-FUSE) **traduz a propriedade
automaticamente**: o container enxerga os arquivos como se fossem dele, e o host
os enxerga como se fossem do usuário logado. Não existe o problema de permissão
que a variável resolve.

Como o uid do usuário no macOS é 501 e o usuário `airflow` dentro da imagem é
50000, forçar `AIRFLOW_UID=501` só criaria um uid sem nome dentro do container,
sem HOME próprio — exatamente o cenário que gera avisos do entrypoint. Por isso o
default aqui é **50000**, o uid nativo da imagem.

Verificação feita neste ambiente:

```
# dentro do container
uid=50000(airflow) gid=0(root) groups=0(root)

# no host, arquivos criados pelo container em logs/
drwxr-xr-x  rafaelnedermaiorino  staff  scheduler/
```

O container escreve como 50000 e o host lê como `rafaelnedermaiorino:staff`
(uid 501). Nenhum erro de permissão em `logs/`.

Os serviços rodam como `user: "<uid>:0"` — grupo 0 (root) — porque a imagem do
Airflow é OpenShift-compatible e dá permissão de escrita ao grupo 0 nos
diretórios de trabalho. Isso é o que faz o mesmo compose funcionar com qualquer
uid, em Linux ou macOS.

Para quem for rodar em **Linux**, o `.env.example` documenta o override:

```bash
echo "AIRFLOW_UID=$(id -u)" >> .env
```

### Compatibilidade arm64

Ambas as imagens são multi-arch e rodam nativamente no M2 — confirmado via
`docker image inspect --format '{{.Architecture}}'` → `arm64`. Não há emulação
QEMU (que deixaria o treino do modelo lento depois).

O `platform:` **não** foi fixado no compose de propósito: fixar `linux/arm64`
quebraria o arquivo para colegas de grupo em Linux/Windows x86 e para CI. O
manifesto multi-arch já resolve isso sozinho.

### Portas

| Porta | Uso | Situação |
| --- | --- | --- |
| 8080 | UI do Airflow (host → container) | **escolhida**, verificada livre |
| 8000 | API FastAPI do projeto (`Dockerfile`) | sem conflito com a 8080 |
| 5432 | Postgres do Airflow | **não exposta** ao host |
| 8974 | health check do scheduler | interna ao container |

Verificação antes de escolher (`lsof -nP -iTCP:<porta> -sTCP:LISTEN`): nenhuma das
portas candidatas tinha listener.

Duas observações sobre as preocupações levantadas:

- **Não há conflito entre o Airflow e a API do projeto.** A API roda na **8000**
  (`Dockerfile:56`, `EXPOSE 8000`), não na 8080. Ficou-se então com a 8080, que é
  o default do Airflow — menos desvio da documentação oficial na hora de
  debugar.
- **A porta 5000 não é usada em lugar nenhum desta stack**, então o AirPlay
  Receiver do macOS (que ocupa a 5000 e a 7000) é irrelevante aqui. A ressalva
  vale para o Flask/`airflow standalone` em setups antigos, não para este.

Se a 8080 estiver ocupada na máquina de quem for rodar, o override não exige
editar o compose:

```bash
AIRFLOW_WEB_PORT=8081 docker compose -f docker-compose.airflow.yml up -d
```

O Postgres **não** expõe porta no host de propósito: ele só é acessado pela rede
interna do compose, e não expor elimina qualquer chance de conflito com um
Postgres local na 5432.

### Volumes

Dois grupos de bind mounts:

| Host | Container | Modo | Para quê |
| --- | --- | --- | --- |
| `airflow/dags` | `/opt/airflow/dags` | rw | as DAGs |
| `airflow/logs` | `/opt/airflow/logs` | rw | logs das tasks |
| `airflow/plugins` | `/opt/airflow/plugins` | rw | plugins (vazio) |
| `airflow/config` | `/opt/airflow/config` | rw | config extra (vazio) |
| `src/` | `/opt/project/src` | **ro** | código do projeto |
| `data/` | `/opt/project/data` | rw | corpus e splits |
| `models/` | `/opt/project/models` | rw | artefatos treinados |
| `reports/` | `/opt/project/reports` | rw | métricas |

`src/` é montado **read-only** porque a DAG consome os módulos como estão e
nunca deve escrever neles. Como efeito colateral, o Python não consegue gravar
`__pycache__` lá dentro — daí `PYTHONDONTWRITEBYTECODE=1` no compose, que evita
a tentativa em vez de deixá-la falhar em silêncio a cada import.

`PYTHONPATH=/opt/project` é o que faz `from src.data.prepare import ...`
funcionar dentro das tasks.

Os módulos de `src/` usam caminhos relativos ao diretório de trabalho
(`Path("data/processed")` e afins). Dentro do container o cwd é `/opt/airflow`,
não a raiz do projeto — por isso a DAG monta **todos os caminhos de forma
absoluta** a partir de `PROJECT_ROOT=/opt/project` e os passa explicitamente
para `ensure_dataset()`, `prepare()` e `train_baseline()`.

O banco de metadados fica em um volume nomeado (`postgres-db-volume`), não em
bind mount — assim `down` preserva o histórico de execuções e só `down -v` reseta.

---

## A imagem própria (`airflow/Dockerfile`)

A imagem oficial do Airflow não traz as bibliotecas de treino, e as que ela traz
estão em versões antigas. A imagem do projeto herda dela e instala:

```
pandas==3.0.5   scikit-learn==1.9.0   huggingface-hub==1.24.0
pyarrow==25.0.0  joblib==1.5.3
```

**Estas versões precisam ser as mesmas do `uv.lock`.** Não é preciosismo: o
modelo treinado pela DAG é desserializado pela API, e um `Pipeline` do
scikit-learn salvo com joblib não é compatível entre versões diferentes da
biblioteca. A imagem base traz scikit-learn 1.6.1 e a API instala 1.9.0 — treinar
numa e carregar na outra geraria `InconsistentVersionWarning` e risco de erro
silencioso na inferência.

O `numpy` fica na 1.26.4 que a imagem já traz: pandas 3.0.5 e scikit-learn 1.9.0
funcionam com ela, o que evita a quebra de ABI que um upgrade para numpy 2.x
causaria nos pacotes compilados do Airflow.

`pip check` reclama de `apache-airflow-providers-google` e `-snowflake`, que
pedem `pandas<2.2`. São providers que este projeto não usa (as tasks são todas
`PythonOperator`) e que não são importados em runtime — o Airflow sobe e opera
normalmente, como as validações abaixo mostram.

---

## A DAG `train_pipeline`

### Como disparar

Pela UI, em <http://localhost:8080>:

1. A DAG aparece na lista como `train_pipeline`, **pausada** (é o padrão do
   compose, para nada disparar sozinho no primeiro `up`).
2. Ative o toggle à esquerda do nome.
3. Clique no botão ▶ (*Trigger DAG*) no canto direito.
4. Acompanhe em *Graph* ou *Grid*; os logs de cada task ficam em
   *Grid → (task) → Logs*.

Equivalente por linha de comando, se preferir:

```bash
docker compose -f docker-compose.airflow.yml exec airflow-scheduler \
  airflow dags trigger train_pipeline
```

### O que cada task faz

| Task | O que faz | Saída |
| --- | --- | --- |
| `ingest` | Chama `ensure_dataset()` de `src/data/download.py`. Baixa o corpus da revisão fixada no Hugging Face **só se ele não estiver presente**, e valida o sha256 dos arquivos nos dois casos. | XCom com `raw_dir`, `downloaded` (bool) e o tamanho + sha256 de cada arquivo. |
| `prepare` | Chama `prepare()` de `src/data/prepare.py`: deduplicação, resolução de rótulos ambíguos, remoção de vazamento entre treino e teste, split estratificado treino/validação. | XCom com `processed_dir`, contagem por split e o caminho de cada parquet. |
| `train` | Chama `train_baseline()` de `src/models/baseline.py` (TF-IDF + LogisticRegression), avalia em validação e teste, aplica o *quality gate* e publica o artefato. | XCom com caminhos, `macro_f1`, `accuracy`, limiar aplicado e a lista de modelos removidos pela retenção. |

As três tasks logam entrada e saída no log do Airflow — shapes dos splits,
checksums, tamanho do vocabulário, métricas por split e o veredito do gate.

Os módulos de `src/` são consumidos **como bibliotecas**, não via `BashOperator`:
a DAG importa `ensure_dataset`, `prepare` e `train_baseline` diretamente. Os
imports ficam dentro do corpo das funções, e não no topo do arquivo, porque o
topo é reexecutado a cada ciclo de parsing do scheduler — um `import pandas` ali
custaria segundos por ciclo sem nenhuma task estar rodando.

`prepare()` chama `ensure_dataset()` internamente. Como a task `ingest` já rodou,
essa chamada interna vira uma revalidação de checksum, não um novo download — a
separação em duas tasks existe para dar visibilidade e retry independentes.

### Agendamento

`schedule=None`: disparo manual. A revisão do corpus é fixada por commit hash, ou
seja, não muda sozinha — não há nada para um agendamento capturar.

Em produção isso viraria uma cadência real, por exemplo `schedule="0 3 * * 1"`
(segundas às 03:00) ou um trigger por Dataset quando novos laudos rotulados
chegassem. `catchup=False` deve permanecer nos dois casos: fazer backfill de um
job de treino só retreinaria o mesmo modelo N vezes sobre o mesmo corpus.

`retries=2` com `retry_delay` de 1 minuto, aplicados às três tasks. A única
exceção é a reprovação do quality gate, que falha de imediato — ver
[Por que o gate não faz retry](#por-que-o-gate-não-faz-retry).

### Onde os artefatos aparecem

Direto na árvore do projeto no host, via os volumes:

```
data/raw_abstracts/        corpus baixado (train.csv, test.csv)
data/processed/            train.parquet, val.parquet, test.parquet
models/model_<timestamp>.joblib    modelo versionado desta execução
models/baseline.joblib             cópia do último modelo aprovado — é o que a API serve
reports/metrics_<timestamp>.json   métricas correspondentes
```

O timestamp tem o formato `YYYYMMDD_HHMMSS` e vem do `logical_date` da execução,
não do relógio no instante do treino. A diferença importa em retry: as três
tentativas de uma mesma execução escrevem **no mesmo arquivo**, em vez de
deixarem um artefato órfão por tentativa.

`baseline.joblib` é uma **cópia** (`shutil.copy2`), não um symlink: um symlink em
`models/` não sobreviveria a um checkout no Windows sem developer mode, e a API
apenas abre o caminho.

**Retenção:** ao final de cada treino aprovado, a task mantém os 5
`model_*.joblib` mais recentes e apaga o resto. `baseline.joblib` nunca é
candidato à remoção, porque não casa com o glob `model_*.joblib`. Os
`metrics_*.json` **não** são podados: são arquivos de poucos KB e o histórico
completo de métricas é justamente o que alimenta o acompanhamento de drift.

Nada disso é versionado — `models/*.joblib` e `reports/metrics_*.json` estão no
`.gitignore`, junto com `airflow/logs/` e `airflow/.env`.

### O quality gate

Antes de publicar, a task compara o **macro-F1 no split de teste** com um limiar
mínimo. Abaixo do limiar, a task falha.

O limiar padrão é **0,62**, definido no código. Ele pode ser sobrescrito em tempo
de execução pela UI em **Admin → Variables**, na chave `min_macro_f1`, sem editar
nem recarregar a DAG. Uma Variable definida **sobrepõe** o default do código — se
existir uma `min_macro_f1` antiga no banco, é ela que vale.

#### De onde vem o 0,62

O valor foi calibrado a partir do baseline observado, não escolhido por
arredondamento. A execução de referência marca **0,6707** de macro-F1 no split de
teste; 0,62 fica cerca de **5 pontos abaixo** disso.

A folga é deliberada e tem dois lados:

- **larga o bastante** para absorver a variação normal de um retreino — mudança
  de corpus, reamostragem do split, ajuste de hiperparâmetro — sem reprovar um
  modelo que continua saudável;
- **apertada o bastante** para que uma regressão real seja barrada. Um limiar
  muito distante do score de verdade deixa de ser um gate e vira decoração:
  passaria mesmo um modelo visivelmente pior.

O limiar deve subir conforme o modelo melhorar. Se um dia o baseline chegar a
0,75 e o limiar continuar em 0,62, o gate volta a não proteger nada.

O ponto de ordem importa: **o gate roda antes da publicação**. Um modelo
reprovado fica em disco como `model_<timestamp>.joblib` para inspeção, mas
`baseline.joblib` continua apontando para o último modelo **aprovado**. Ou seja,
uma regressão de qualidade nunca chega à API — o pior caso é a API continuar
servindo o modelo anterior, que é o comportamento desejável.

Isso é o gate de qualidade de dados/modelo que conversa com o item D3·A7 do
plano de monitoramento.

#### Por que o gate não faz retry

A reprovação levanta `AirflowFailException`, que falha a task na hora e
**ignora a política de retry**. O gate é determinístico: os mesmos splits com o
mesmo `RANDOM_STATE` produzem o mesmo macro-F1, então retentar só gastaria um
minuto para chegar ao mesmo veredito.

Isso não desliga o retry do resto: `retries=2` continua valendo para as três
tasks. Falhas transitórias — rede no `ingest`, I/O no `prepare`, leitura dos
parquet ou escrita do artefato no próprio `train` — levantam exceções comuns e
seguem sendo retentadas normalmente. A isenção é só do gate, e sai de graça, sem
precisar configurar `retries` diferente por task.

---

## Tempo do primeiro start

Medido neste ambiente (MacBook Pro M2, Docker Desktop), após `down -v` para
garantir banco zerado:

| Marco | Tempo |
| --- | --- |
| `up -d` retorna | 12 s |
| `/health` responde `healthy` | 26 s |
| Os 3 containers `healthy` | **~30 s** |

O `airflow-init` (migrations + criação do usuário admin) leva ~5 s. É bem mais
rápido que o típico porque o Postgres é local e o M2 é rápido — em máquinas mais
modestas é normal esse passo levar 1–2 min, e é ele o gargalo do primeiro start.

**O primeiro start de todos inclui o download das imagens**, que não está
contabilizado acima: `apache/airflow:2.11.0-python3.11` tem **2,45 GB** e
`postgres:16-alpine` tem **411 MB**. Dependendo da conexão, são 3–10 min só de
pull. Para pré-baixar antes de subir:

```bash
docker pull apache/airflow:2.11.0-python3.11
docker pull postgres:16-alpine
```

Starts subsequentes (`down` sem `-v`, depois `up -d`) são mais rápidos: as
migrations já estão aplicadas e o `airflow-init` só confirma o estado.

---

## Verificação de que está tudo certo

```bash
# 1. containers healthy
docker compose -f docker-compose.airflow.yml ps
# postgres, airflow-scheduler, airflow-webserver → running (healthy)

# 2. scheduler batendo heartbeat
curl -s http://localhost:8080/health | python3 -m json.tool
# metadatabase: healthy, scheduler: healthy

# 3. DAGs de exemplo desabilitadas
docker compose -f docker-compose.airflow.yml exec airflow-scheduler airflow dags list
# so train_pipeline — nenhuma das ~50 DAGs de exemplo

docker compose -f docker-compose.airflow.yml exec airflow-scheduler airflow config get-value core load_examples
# false

# 4. a DAG parseia sem erro de import
docker compose -f docker-compose.airflow.yml exec airflow-scheduler airflow dags list-import-errors
# "No data found"
```

Resultado obtido nesta máquina: os três containers `healthy`, `/health` com
`metadatabase` e `scheduler` saudáveis, `load_examples=false`, e `airflow dags
list` mostrando apenas `train_pipeline`, sem erros de import.

Login validado de fato (não só o carregamento da tela): `POST /login/` com CSRF
token retornou `302` e o `GET /home` seguinte retornou `200` já autenticado.

---

## Execução validada da DAG

Disparada pela UI, com a stack recém-subida. Tempo por task:

| Task | Duração | Observação |
| --- | --- | --- |
| `ingest` | 0,4 s | corpus já presente → só revalidação de sha256 |
| `prepare` | 1,5 s | 5.634 treino + 995 validação + 2.657 teste |
| `train` | 4,2 s | vocabulário TF-IDF de 50.000 termos |
| **total** | **~6 s** | |

Um `ingest` que precise baixar o corpus de verdade (≈18 MB) leva bem mais que
0,4 s — o número acima é o caminho quente, que é o normal a partir da segunda
execução.

Métricas da execução:

| Split | n | accuracy | macro-F1 |
| --- | --- | --- | --- |
| validação | 995 | 0,7970 | 0,7924 |
| teste | 2.657 | 0,6699 | **0,6707** |

O macro-F1 de teste (0,6707) é o número a partir do qual o limiar de 0,62 foi
calibrado — ver [De onde vem o 0,62](#de-onde-vem-o-062).

Confirmações feitas no host, não só na UI:

- **`baseline.joblib` foi realmente regravado.** sha256 antes
  `9e3df983…`, depois `243f5813…`; mtime foi de `Jul 23 21:08` para
  `Jul 24 10:47`. O `model_<timestamp>.joblib` da execução tem hash idêntico ao
  do `baseline.joblib`, o que confirma que a cópia é fiel.
- **A API carrega o modelo novo.** Reiniciada, o `/health` respondeu
  `{"model_loaded": true, "model_path": "models/baseline.joblib"}` e o `/predict`
  classificou um resumo sobre infarto como `cardiovascular diseases` com
  confiança 0,708. O carregamento foi testado com `warnings.simplefilter("error")`,
  ou seja, qualquer `InconsistentVersionWarning` do scikit-learn teria virado
  erro — não houve nenhum.
- **A retenção funciona.** Após execuções sucessivas, `models/` estabilizou em
  5 arquivos `model_*.joblib` mais o `baseline.joblib`, com os mais antigos
  removidos.

### Teste do quality gate

Com `min_macro_f1` temporariamente em `0.99` (Admin → Variables), a DAG foi
disparada de novo:

```
[0s]  run=queued   ingest=None(try0)     prepare=None(try0)     train=None(try0)
[6s]  run=running  ingest=success(try1)  prepare=success(try1)  train=running(try1)
[11s] run=failed   ingest=success(try1)  prepare=success(try1)  train=failed(try1)
```

Comportamento observado, exatamente o esperado:

- a task `train` falhou **na primeira tentativa, sem retry** — o log do Airflow
  registra `Immediate failure requested`, que é a assinatura de
  `AirflowFailException`. A execução inteira levou 11 s;
- a mensagem de erro nomeia o problema e o que fazer com ele:
  `quality gate failed: test macro_f1 0.6707 is below the minimum of 0.9900.
  baseline.joblib was left untouched; the rejected model is at
  /opt/project/models/model_20260724_112748.joblib`;
- **`baseline.joblib` ficou intacto** — sha256 idêntico ao de antes do teste
  (`243f5813…`). O modelo reprovado ficou em disco como arquivo versionado,
  disponível para inspeção.

Removida a Variable, a execução seguinte voltou a passar com o limiar de código
(0,62) e publicou normalmente.

> **Comparação com a versão anterior:** enquanto o gate levantava um `ValueError`
> comum, a mesma reprovação consumia as 3 tentativas e **152 s** antes de a DAG
> ficar vermelha. Com `AirflowFailException` são **11 s** — o feedback de uma
> regressão de qualidade chega ~14× mais rápido, e o retry continua disponível
> para as falhas onde ele realmente ajuda.

---

## Problemas encontrados

### 1. Modelo treinado pelo Airflow não carregava na API (resolvido)

Foi o único problema real, e não é óbvio. A primeira execução da DAG terminou com
as três tasks verdes e gravou `models/baseline.joblib` normalmente — mas ao
carregar esse arquivo no ambiente da API:

```
ModuleNotFoundError: No module named 'dill'
```

**Causa.** O runtime de task do Airflow importa `dill` e chama `dill.extend()`,
que injeta cerca de 40 reducers próprios em `pickle._Pickler.dispatch`. O
`joblib` copia essa tabela para `NumpyPickler.dispatch` no momento em que é
importado. Como a task importa `joblib` depois disso, todo modelo salvo de dentro
de uma task serializa os escalares numpy do vocabulário TF-IDF via
`dill._dill._get_attr` — e o arquivo passa a exigir `dill` para ser lido. A API
não tem `dill` (depende só do grupo de serving do `pyproject.toml`), então
quebrava na carga.

O detalhe cruel é que **a DAG fica verde**: o treino termina, as métricas são
gravadas, o artefato existe. O defeito só aparece na hora de servir. Sem o teste
de carga cruzada, isso passaria batido até a API ser reiniciada em produção.

**Correção.** A task `train` chama `_restore_stock_pickler()` antes de importar
qualquer coisa de `src/`: reverte o patch com `dill.extend(False)` e, por
segurança, limpa entradas de dill que já tenham vazado para o dispatch do joblib.
O Airflow não é afetado — a serialização de XCom aqui é JSON
(`enable_xcom_pickling` é `false` por padrão) e carrega dicts simples.

**Verificação.** Depois da correção, o artefato não contém mais a string `dill`,
carrega no ambiente da API com warnings tratados como erro, e o `/predict`
responde corretamente.

Vale como lição geral: **"a DAG ficou verde" não é evidência de que o artefato
presta.** O teste que importa é carregar o artefato no ambiente que vai consumi-lo.

### 2. Riscos evitados por decisão de projeto

Estes não chegaram a acontecer — foram fechados no desenho:

| Risco | Como foi evitado |
| --- | --- |
| Permissão em `logs/` no macOS | `AIRFLOW_UID=50000` (uid da imagem) + grupo 0, em vez do `id -u` que a doc oficial sugere para Linux |
| Emulação QEMU no M2 | Imagens multi-arch verificadas como arm64 antes de subir |
| Conflito de porta | `lsof` nas portas candidatas antes de escolher; Postgres sem porta exposta |
| Fernet key rotativa | Chave fixa via `.env` — se ficasse vazia, o Airflow geraria uma nova a cada restart e as Connections salvas parariam de descriptografar |
| DAG disparando sozinha no primeiro up | `AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=true` |
| Modelo ilegível pela API por versão de lib | Versões do `Dockerfile` fixadas iguais às do `uv.lock`; imagem em `python3.11` como o `.python-version` |
| Modelo ruim chegando à API | Quality gate roda **antes** da publicação em `baseline.joblib` |
| Caminho relativo resolvendo errado | Todos os caminhos montados de forma absoluta a partir de `PROJECT_ROOT` |

Ruído esperado (não é erro): os comandos `airflow ...` via `exec` imprimem
`RemovedInAirflow3Warning` sobre unidades de métricas de timer. É um aviso de
deprecação da própria 2.11 e não afeta o funcionamento.

---

## Segurança

Nenhum segredo é versionado. As chaves e credenciais vivem em `airflow/.env`, que
está no `.gitignore`; o `.env.example` traz só placeholders `<GERAR>` e as
instruções de geração. O compose declara essas variáveis como obrigatórias
(`${VAR:?mensagem}`), então uma cópia do repositório sem `.env` falha na hora com
uma mensagem clara, em vez de subir com um valor default previsível.

> **Nota de histórico:** as primeiras versões deste compose traziam uma Fernet key
> e uma secret key de desenvolvimento em claro, e elas continuam no histórico do
> git. Devem ser consideradas queimadas. Não há impacto prático — davam acesso a
> um banco de metadados local, sem porta exposta, com dados descartáveis — mas as
> chaves em uso agora foram geradas do zero e nunca entraram no repositório.

Mesmo assim, esta stack é **estritamente local**: a UI escuta em `localhost` e o
Postgres não tem porta publicada. Para qualquer ambiente compartilhado, as
credenciais deveriam sair do `.env` e ir para um secret manager de verdade.

---

## Próximo passo

A pipeline de treino está orquestrada e validada ponta a ponta. O que falta na
Etapa 2 é acoplar o monitoramento: expor as métricas de `reports/metrics_*.json`
ao longo do tempo e usar a série histórica para detectar drift — que é o gatilho
natural para transformar o `schedule=None` em uma cadência de retreino real.
