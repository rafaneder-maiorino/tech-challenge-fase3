# Monitoramento e Observabilidade — Etapa 3

Stack de observabilidade da API de triagem de laudos: a própria API instrumentada
com `prometheus_client`, um Prometheus coletando as métricas e um Grafana com
datasource e dashboard provisionados por arquivo.

Tudo nesta etapa vive em três lugares: `src/api/metrics.py` (instrumentação),
`monitoring/` (configuração de Prometheus e Grafana) e
`docker-compose.monitoring.yml` (a stack). A stack do Airflow, da Etapa 2, é
independente — projeto, rede, portas e imagens diferentes; nada é compartilhado.

---

## 1. Como subir

A partir da raiz do repositório:

```bash
docker compose -f docker-compose.monitoring.yml up -d --build
```

O `--build` é necessário na primeira vez: a imagem da API é construída do
`Dockerfile` do projeto, que baixa o corpus e treina o baseline dentro do build.

Serviços e portas:

| Serviço      | URL                     | Observação                                   |
| ------------ | ----------------------- | -------------------------------------------- |
| API          | <http://localhost:8000> | `/health`, `/predict`, `/metrics`, `/docs`   |
| Prometheus   | <http://localhost:9090> | UI de consulta e `Status → Targets`          |
| Grafana      | <http://localhost:3000> | abre direto no dashboard (acesso anônimo)    |

O Grafana sobe com acesso anônimo em papel `Viewer`, então o dashboard abre sem
login. Para editar, entrar como `admin` / `admin` (sobrescrevível por
`GRAFANA_ADMIN_USER` e `GRAFANA_ADMIN_PASSWORD`). **Isso vale para uma stack
local de avaliação; não expor assim fora de `localhost`.**

Portas ocupadas podem ser trocadas sem editar o compose:

```bash
API_PORT=8001 PROMETHEUS_PORT=9091 GRAFANA_PORT=3001 \
  docker compose -f docker-compose.monitoring.yml up -d
```

Conferir se está tudo de pé:

```bash
docker compose -f docker-compose.monitoring.yml ps
curl -s localhost:8000/health
curl -s 'localhost:9090/api/v1/targets?state=active' | grep -o '"health":"[a-z]*"'
```

O `depends_on` do Prometheus espera `service_healthy` da API, e o healthcheck da
API exige `model_loaded=true` — não basta o processo estar no ar. Se a API subir
sem conseguir carregar o artefato, o Prometheus não sobe junto e o problema
aparece no `docker compose ps` em vez de virar um gráfico vazio sem explicação.

Derrubar:

```bash
docker compose -f docker-compose.monitoring.yml down          # preserva os volumes
docker compose -f docker-compose.monitoring.yml down -v       # apaga a série histórica
```

### Versões das imagens

Fixadas, sem `:latest`:

| Imagem                    | Versão        | Por quê fixar                                            |
| ------------------------- | ------------- | -------------------------------------------------------- |
| `prom/prometheus`         | `v3.13.2`     | mudanças de PromQL entre majors alterariam as queries     |
| `grafana/grafana`         | `13.1.3`      | o JSON provisionado é migrado pela versão que o lê        |
| API (build local)         | `Dockerfile`  | tag `tech-challenge-fase3/api:monitoring`                 |

Com `:latest`, a mesma árvore de código produziria stacks diferentes conforme a
data do `docker compose up`, e um dashboard provisionado é sensível à versão do
Grafana que o interpreta.

---

## 2. Gerando carga

Um dashboard vazio não prova nada. `scripts/generate_load.py` produz tráfego
misturado contra a API:

```bash
uv run python scripts/generate_load.py                                  # 120 s a 30 req/s
uv run python scripts/generate_load.py --duration 300 --rps 40
uv run python scripts/generate_load.py --requests 5000 --concurrency 16
uv run python scripts/generate_load.py --error-rate 0.25 --seed 42      # erro alto, reprodutível
```

A mistura padrão é 87% de `/predict` válido, 8% de `/predict` inválido e 5% de
`/health`. Os textos válidos cobrem as cinco classes (dois abstracts por classe),
para o painel de distribuição mostrar espalhamento em vez de uma barra só. Os
payloads inválidos exercitam cinco caminhos distintos de validação — string
vazia, só espaço em branco, campo ausente, tipo errado e acima de
`MAX_TEXT_CHARS` — e todos resultam em `422`.

Ao final o script imprime o resumo por desfecho:

```
sent 7200 requests in 240.0s (30.0 req/s achieved)
  health:2xx                   349
  invalid:4xx                  588
  valid:2xx                    6263
```

Se alguma requisição não chegar na API, o script sai com código 1 e diz para
conferir o `docker compose ps` — falha de conexão é resultado, não ruído.

---

## 3. Métricas expostas pela API

Endpoint: `GET /metrics`, formato de exposição do Prometheus (`text/plain`).
Definidas em `src/api/metrics.py`.

### `api_requests_total` — Counter

Requisições HTTP atendidas.

| Label      | Valores                                   |
| ---------- | ----------------------------------------- |
| `method`   | `GET`, `POST`                             |
| `endpoint` | `/health`, `/predict`, `unmatched`        |
| `status`   | `200`, `422`, `503`, `500`, …             |

Duas decisões que mudam o que os painéis mostram:

- **`/metrics` não é contado.** O Prometheus raspa esse endpoint a cada 5 s. Se
  ele entrasse na conta, dominaria o painel de taxa de requisições e diluiria a
  proporção de erro com tráfego que nenhum cliente gerou.
- **Rota que não casa vira `unmatched`, não o path cru.** Registrar o path de um
  404 permitiria a qualquer um de fora criar cardinalidade ilimitada de labels
  com um loop de URLs aleatórias.

### `api_request_duration_seconds` — Histogram

Latência ponta a ponta do handler: entrada no middleware até a resposta pronta.
Inclui validação do payload, inferência e serialização. Labels: `method`,
`endpoint`.

Buckets em segundos (ver seção 5 para a calibração):

```
0.00025  0.0005  0.00075  0.001  0.0015  0.002  0.003  0.004
0.005    0.0075  0.01     0.025  0.05    0.1    0.5    1.0    +Inf
```

### `api_predictions_total` — Counter

Predições devolvidas, por classe prevista. Labels: `label` (`1` a `5`) e
`label_name` (`neoplasms`, `digestive system diseases`, `nervous system
diseases`, `cardiovascular diseases`, `general pathological conditions`).

Esta é a métrica de **modelo**, não de HTTP: um serviço pode responder 200 em
100% das requisições e ainda assim estar devolvendo um mix de classes que se
afastou da distribuição de treino. As cinco séries são criadas com valor zero no
startup — sem isso, uma classe ainda não predita simplesmente não apareceria no
painel, o que se confunde com "existe e vale zero".

> Nota sobre a numeração: o enunciado da tarefa fala em classes `0-4`, mas o
> `condition_label` do corpus e o `CONDITION_NAMES` de `src/labels.py` usam
> `1-5`. A métrica segue o domínio real, `1-5`.

### `api_model_loaded` — Gauge

`1` quando o classificador está carregado e apto a servir, `0` caso contrário.

Existe por causa de um achado da Etapa 1: a API sobe mesmo quando o `joblib.load`
falha, para que `/health` consiga reportar o problema em vez de o container
entrar em crash loop sem diagnóstico. Nesse estado o serviço está **UP** para
qualquer check de liveness e responde `503` em todo `/predict`. Só esta métrica
separa "processo no ar" de "apto a servir".

### Métricas automáticas do `prometheus_client`

A biblioteca também expõe `python_gc_*`, `python_info` e, em Linux,
`process_cpu_seconds_total` / `process_resident_memory_bytes`. Não são usadas
pelo dashboard, mas ficam disponíveis para investigar uso de memória do
container.

---

## 4. Instrumentação: manual, não `prometheus-fastapi-instrumentator`

A tarefa pedia para avaliar as duas opções. A escolha foi registrar as métricas à
mão contra o registry padrão do `prometheus_client`, por dois motivos:

1. **Controle de buckets.** O histograma padrão do instrumentator começa em 5 ms.
   Como a inferência sequencial deste serviço custa ~0,55 ms (Etapa 1), *toda*
   requisição cairia no primeiro bucket e P50, P95 e P99 reportariam o mesmo
   número. Dá para sobrescrever os buckets na biblioteca, então isso sozinho não
   fecha a questão.
2. **As métricas de modelo.** `api_predictions_total` e `api_model_loaded` não
   saem de nenhum instrumentador automático — são específicas do domínio e
   precisariam ser escritas à mão de qualquer jeito. Com elas já no código, a
   biblioteca passaria a pagar apenas pelo middleware HTTP, que são as ~40 linhas
   de `metrics_middleware`.

Custo da escolha: uma dependência a menos (`prometheus-client` já viria junto) e
o middleware é responsabilidade nossa — inclusive contar como `500` uma exceção
não tratada, que é feito explicitamente no `except` do middleware.

O registry é global por processo, então o módulo precisa ser importado uma única
vez por worker. A API roda **um** worker uvicorn por container, o que mantém os
números íntegros. Um deploy multi-worker precisaria de `PROMETHEUS_MULTIPROC_DIR`.

---

## 5. Calibração dos buckets

Este foi o ajuste não trivial da etapa, e ele mudou depois da primeira medição.

**Ponto de partida.** O baseline da Etapa 1
(`reports/latency_baseline.json`) mediu, a concorrência 1: P50 de 0,54 ms, P95 de
0,60 ms e P99 de 0,62 ms de inferência no servidor. Daí os quatro cortes abaixo
de 1 ms (0,25 / 0,5 / 0,75 / 1 ms).

**O que a carga real mostrou.** Rodando o gerador a 30 req/s com 8 requisições em
voo, a latência ponta a ponta medida pelo middleware ficou em **~3,8 ms de
média** (`_sum / _count`), quase uma ordem de grandeza acima do número da Etapa 1.

**Por quê.** Não é a inferência ficando mais lenta. O handler `predict` é um `def`
síncrono, então o Starlette o executa no threadpool do anyio; com requisições
concorrentes, o cronômetro do middleware passa a incluir o tempo de fila nesse
threadpool, além do parse e da serialização do JSON. Os 0,55 ms da Etapa 1 são
inferência pura em fila de um; os ~3,8 ms são o custo real de uma requisição sob
concorrência. Ambos são verdadeiros e medem coisas diferentes — o número honesto
para um painel de latência de API é o segundo.

**O ajuste.** A primeira versão dos buckets tinha só duas fronteiras entre 1 ms e
10 ms (2,5 e 5 ms), e 76% da massa caía na faixa de 1–5 ms: o P50 era interpolado
linearmente dentro de um bucket de 2,5 ms de largura, e o P99 caía no bucket de
5–10 ms, o último finito antes de um salto para 25 ms. Foram acrescentadas as
fronteiras de **1,5 / 2 / 3 / 4 / 7,5 ms**, levando a década de 1 a 10 ms de duas
para sete fronteiras.

O efeito é mensurável, e não é cosmético:

| Quantil | Buckets iniciais | Buckets calibrados |
|---|---|---|
| P50 | 3,5 ms | 4,2 ms |
| P95 | 8,6 ms | 7,1 ms |
| P99 | **9,7 ms** | **7,5 ms** |

O P99 caiu ~30% sem que nada no serviço mudasse. Os 9,7 ms nunca existiram: o
`histogram_quantile` interpola linearmente dentro do bucket em que o quantil cai,
e com o bucket de 5–10 ms um P99 real de 7,4 ms era empurrado para perto do topo
da faixa. Um alerta de latência configurado sobre o número antigo estaria
disparando por um artefato da própria instrumentação.

Os cortes abaixo de 1 ms **foram mantidos**: o regime sequencial continua real —
é o que acontece no smoke test do CI e numa demonstração com um usuário só — e
sem eles esse caso voltaria a colapsar num único bucket. Sob carga concorrente,
2,5% das requisições ainda caem abaixo de 1 ms.

Acima de 10 ms as fronteiras (25 / 50 / 100 / 500 ms / 1 s) existem para detectar
uma regressão, não para resolvê-la em detalhe.

Custo em cardinalidade: 17 fronteiras × 3 combinações de `method`/`endpoint`
≈ 51 séries de bucket. Irrelevante para um Prometheus local.

Os números medidos com os buckets finais, painel a painel, estão em
[`monitoring-dashboard-snapshot.md`](monitoring-dashboard-snapshot.md).

---

## 6. O dashboard

**Arquivo:** `monitoring/grafana/dashboards/api-monitoring.json`
**UID:** `tc-fase3-api` · **Pasta:** `Tech Challenge` · **Refresh:** 5 s ·
**Janela padrão:** últimos 15 minutos

Oito painéis, em três faixas.

### Faixa 1 — quatro indicadores de estado atual

| Painel                    | Query PromQL                                                                                          |
| ------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Status do modelo**      | `api_model_loaded`                                                                                      |
| **Requisições por segundo** | `sum(rate(api_requests_total[$__rate_interval]))`                                                     |
| **Taxa de erro**          | `sum(rate(api_requests_total{status!~"2.."}[$__rate_interval])) / sum(rate(api_requests_total[$__rate_interval]))` |
| **P99 de /predict**       | `histogram_quantile(0.99, sum by (le) (rate(api_request_duration_seconds_bucket{endpoint="/predict"}[$__rate_interval])))` |

O **Status do modelo** usa value mapping: `1` → `CARREGADO` em verde, `0` →
`DEGRADADO` em vermelho, sem série → `SEM DADOS`. É o painel que distingue
degradado de saudável quando tudo mais parece no ar.

A **Taxa de erro** fica verde abaixo de 5%, laranja entre 5% e 20% e vermelha
acima disso. Com a mistura padrão do gerador (8% de payloads inválidos) o valor
esperado gira em torno de 8%.

### Faixa 2 — tráfego e latência

**Total de requisições — taxa por segundo**

```promql
sum by (endpoint, status) (rate(api_requests_total[$__rate_interval]))
```

Uma série por combinação de rota e status. Séries `4xx` e `5xx` são forçadas para
vermelho por um override de regex, então um pico de erro se destaca sem precisar
ler a legenda.

**Latência de /predict — P50, P95, P99**

```promql
histogram_quantile(0.50, sum by (le) (rate(api_request_duration_seconds_bucket{endpoint="/predict"}[$__rate_interval])))
histogram_quantile(0.95, sum by (le) (rate(api_request_duration_seconds_bucket{endpoint="/predict"}[$__rate_interval])))
histogram_quantile(0.99, sum by (le) (rate(api_request_duration_seconds_bucket{endpoint="/predict"}[$__rate_interval])))
```

O `sum by (le)` é obrigatório: `histogram_quantile` precisa receber um conjunto
de buckets agregado pela dimensão `le`, e sem ele a query devolveria um quantil
por série de `method`, o que não é o que se quer ver.

`$__rate_interval` em vez de `[1m]` fixo: o Grafana o calcula a partir do
`timeInterval` do datasource (5 s, igual ao `scrape_interval`) e da janela
selecionada, garantindo pelo menos quatro amostras dentro do intervalo mesmo
quando se dá zoom out para várias horas.

### Faixa 3 — erro e modelo

**Taxa de erro — proporção de respostas não-2xx**: a mesma razão do indicador,
como série temporal, com linha de threshold tracejada em 20%.

Quando não há tráfego nenhum, o denominador é zero e a query devolve `NaN` — o
painel mostra uma lacuna. Isso é intencional: preencher com zero afirmaria "não
há erros", quando o fato é "não há informação".

**Distribuição de predições por classe**

```promql
sum by (label_name) (increase(api_predictions_total[$__range]))
```

Donut com o mix de classes na janela selecionada. `$__range` acompanha o time
picker, então o painel responde ao intervalo escolhido em vez de a uma janela
fixa. É o painel de observabilidade de modelo: um desvio forte em relação à
distribuição do corpus de treino é sinal de drift no tráfego de entrada.

---

## 7. Provisionamento: por que o dashboard sobrevive ao `down`

Nada é criado pela UI. Três arquivos, todos versionados:

```
monitoring/
├── prometheus.yml                                   # scrape de api:8000
└── grafana/
    ├── provisioning/
    │   ├── datasources/prometheus.yml               # datasource, uid fixo
    │   └── dashboards/dashboards.yml                # provider de dashboards
    └── dashboards/api-monitoring.json               # o dashboard
```

O `provisioning/` é montado read-only em `/etc/grafana/provisioning` e o Grafana
o lê na inicialização, antes de qualquer login. O provider aponta para
`/var/lib/grafana/dashboards`, onde `monitoring/grafana/dashboards/` está montado.

Dois detalhes que quebrariam isso se fossem deixados no padrão:

- **`uid: prometheus` fixo no datasource.** O JSON do dashboard referencia essa
  uid em cada painel. Se o Grafana gerasse uma uid aleatória a cada `up`, o
  dashboard subiria com todos os painéis órfãos, reclamando de "datasource not
  found".
- **`disableDeletion: true`** no provider, para uma exclusão acidental na UI não
  remover o dashboard enquanto o arquivo continua no repositório.

`allowUiUpdates: true` fica ligado de propósito: permite ajustar um painel na UI
e exportar o JSON, que é como o arquivo desta entrega foi finalizado. O arquivo
segue sendo a fonte da verdade — o provider o reaplica a cada 30 s e sobrescreve
qualquer edição não exportada.

### Verificação do ciclo down/up

```bash
docker compose -f docker-compose.monitoring.yml down
docker compose -f docker-compose.monitoring.yml up -d
curl -s -u admin:admin 'localhost:3000/api/dashboards/uid/tc-fase3-api' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['dashboard']['title'], '|', len(d['dashboard']['panels']), 'painéis')"
```

Resultado esperado: `Tech Challenge Fase 3 — API de Triagem | 8 painéis`.

O `down` sem `-v` preserva os volumes nomeados `prometheus-data` e
`grafana-data`, então a série histórica também continua lá e os gráficos voltam
populados. Com `down -v` o dashboard ainda volta (ele vem do arquivo), mas os
painéis começam vazios até rodar o gerador de carga de novo — o que é justamente
a diferença entre o que é provisionado e o que é dado.

---

## 8. Consultas úteis fora do dashboard

Direto no Prometheus (<http://localhost:9090>):

```promql
# O alvo está sendo raspado?
up{job="api"}

# Requisições que falharam validação, por minuto
rate(api_requests_total{status="422"}[5m]) * 60

# Requisições que bateram em rota inexistente
increase(api_requests_total{endpoint="unmatched"}[1h])

# Latência média (não é o mesmo que P50, mas revela cauda longa quando divergem)
rate(api_request_duration_seconds_sum{endpoint="/predict"}[5m])
  / rate(api_request_duration_seconds_count{endpoint="/predict"}[5m])

# Fração das requisições atendidas abaixo de 5 ms
sum(rate(api_request_duration_seconds_bucket{endpoint="/predict",le="0.005"}[5m]))
  / sum(rate(api_request_duration_seconds_count{endpoint="/predict"}[5m]))

# Modelo indisponível em algum momento da última hora
min_over_time(api_model_loaded[1h]) == 0
```

---

## 9. Problemas comuns

| Sintoma                                       | Causa provável                                                                 |
| --------------------------------------------- | ------------------------------------------------------------------------------ |
| Painéis vazios, `up{job="api"}` = 1            | Não houve tráfego. Rodar `scripts/generate_load.py`.                            |
| `up{job="api"}` = 0                            | API não subiu ou não ficou healthy: `docker compose logs api`.                  |
| Painéis órfãos, "datasource not found"         | `uid` do datasource divergindo do referenciado no JSON do dashboard.            |
| Taxa de erro em branco                         | Sem tráfego: o denominador é zero e a query devolve `NaN`. Esperado.            |
| Latência muito acima do baseline da Etapa 1    | Esperado sob concorrência — ver seção 5.                                        |
| `Status do modelo` em `DEGRADADO`              | Artefato não carregou. `curl -s localhost:8000/health` mostra o caminho tentado.|
| Porta 3000/9090/8000 em uso                    | `GRAFANA_PORT=3001 ... docker compose up -d`.                                   |

---

## 10. Testes

`tests/test_metrics.py` cobre o contrato da instrumentação:

```bash
uv run pytest tests/test_metrics.py -v
```

O registry é global no processo e todos os testes da sessão o compartilham, então
nenhuma asserção usa valor absoluto de contador — as verificações são sobre
deltas em volta de uma requisição e sobre o formato da exposição, que é do que um
scrape de fato depende. Entre outras coisas, os testes garantem que um `422` cai
numa série própria (senão o painel de erro leria zero), que `/metrics` não se
conta a si mesmo, que rotas inexistentes não viram label, e que as cinco classes
têm série desde o startup.
