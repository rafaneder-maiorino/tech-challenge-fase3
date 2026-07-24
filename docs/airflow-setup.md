# Airflow — setup local (Etapa 2)

Stack mínima do Apache Airflow para orquestrar a pipeline de ingestão e treino do
Tech Challenge. **Esta etapa cobre apenas a infraestrutura** — nenhuma DAG foi
escrita ainda, de propósito, para separar problemas de infra de problemas de
código de DAG.

| Item | Valor |
| --- | --- |
| Versão do Airflow | `2.11.0` (imagem `apache/airflow:2.11.0-python3.11`) |
| Executor | `LocalExecutor` |
| Metadata DB | `postgres:16-alpine` (sem porta exposta) |
| UI | http://localhost:8080 |
| Credenciais | `airflow` / `airflow` |
| Containers | 3 de longa duração + 1 one-shot (`airflow-init`) |
| Arquitetura | arm64 nativo (sem emulação QEMU) |

---

## Comandos

Todos rodados a partir do diretório `airflow/`:

```bash
cd airflow

# subir
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

Acesso à UI: <http://localhost:8080> → usuário `airflow`, senha `airflow`.

Configuração opcional: `cp .env.example .env` e ajustar. Todos os valores têm
default no compose, então o `.env` não é obrigatório.

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

Bind mounts de `dags/`, `logs/`, `plugins/` e `config/` para `/opt/airflow/*`.
As DAGs da próxima etapa vão em `airflow/dags/`.

O banco de metadados fica em um volume nomeado (`postgres-db-volume`), não em
bind mount — assim `down` preserva o histórico de execuções e só `down -v` reseta.

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
# "No data found" — nenhuma DAG registrada (esperado: ainda não escrevemos nenhuma)

docker compose -f docker-compose.airflow.yml exec airflow-scheduler airflow config get-value core load_examples
# false
```

Resultado obtido nesta máquina: os três containers `healthy`, `/health` com
`metadatabase` e `scheduler` saudáveis, `load_examples=false` e `airflow dags list`
retornando `No data found` — sem nenhuma das ~50 DAGs de exemplo.

Login validado de fato (não só o carregamento da tela): `POST /login/` com CSRF
token retornou `302` e o `GET /home` seguinte retornou `200` já autenticado.

---

## Problemas encontrados

Nenhum erro ocorreu durante o setup. A stack subiu de primeira e todos os
containers ficaram `healthy`.

Os pontos abaixo eram riscos conhecidos que foram **evitados por decisão de
projeto**, não problemas que apareceram:

| Risco | Como foi evitado |
| --- | --- |
| Permissão em `logs/` no macOS | `AIRFLOW_UID=50000` (uid da imagem) + grupo 0, em vez do `id -u` que a doc oficial sugere para Linux |
| Emulação QEMU no M2 | Imagens multi-arch verificadas como arm64 antes de subir |
| Conflito de porta | `lsof` nas portas candidatas antes de escolher; Postgres sem porta exposta |
| Fernet key rotativa | Chave fixa via env — se ficasse vazia, o Airflow geraria uma nova a cada restart e as Connections salvas parariam de descriptografar |
| DAG disparando sozinha no primeiro up | `AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=true` |
| Quebra ao importar `src/` depois | Imagem fixada em `python3.11`, igual ao `.python-version` |

Ruído esperado (não é erro): os comandos `airflow ...` via `exec` imprimem
`RemovedInAirflow3Warning` sobre unidades de métricas de timer. É um aviso de
deprecação da própria 2.11 e não afeta o funcionamento.

---

## Segurança

As credenciais (`airflow`/`airflow`), a Fernet key e a secret key estão versionadas
em claro no compose e no `.env.example`. Isso é aceitável **porque esta stack é
estritamente local** — a UI escuta em `localhost` e o Postgres não é exposto. Nada
disso deve ser reaproveitado em ambiente compartilhado ou exposto à internet.

O arquivo `airflow/.env` está no `.gitignore`, então overrides locais não vazam
para o repositório.

---

## Próximo passo

Escrever a DAG de ingestão e treino em `airflow/dags/`, apontando para o código de
`src/`. Como a infra já está validada isoladamente, qualquer falha a partir daqui
pode ser atribuída ao código da DAG.
