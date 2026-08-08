# Snapshot do dashboard populado

Registro da validação da Etapa 3: a stack no ar, o gerador de carga rodando e
os oito painéis com dados. Os valores abaixo foram lidos **através do proxy de
datasource do Grafana** (`/api/datasources/proxy/uid/prometheus/api/v1/query`),
não direto do Prometheus — é a mesma rota que os painéis usam para renderizar,
então o que está aqui é o que o painel mostra.

Reproduzir:

```bash
docker compose -f docker-compose.monitoring.yml up -d --build
uv run python scripts/generate_load.py --duration 260 --rps 30 --seed 11
# com a carga ainda rodando, abrir http://localhost:3000
```

**Condições da medição:** 30 req/s, 8 requisições em voo, mistura padrão
(87% `/predict` válido, 8% `/predict` inválido, 5% `/health`), janela de
`rate()` de 1 minuto.

---

## Como o dashboard se apresenta

Três faixas, de cima para baixo.

**Faixa 1 — quatro indicadores grandes, lado a lado.** Da esquerda para a
direita: um bloco verde escrito `CARREGADO`, e três números com sparkline de
fundo — 30,1 req/s, 8,5% de erro em laranja, e 7,45 ms de P99 em verde.

**Faixa 2 — dois gráficos de série temporal, meio a meio.** À esquerda, três
linhas achatadas e estáveis: `/predict — 200` perto de 26 req/s bem no alto,
`/predict — 422` em vermelho rente ao eixo em ~2,6 req/s, e `/health — 200`
logo abaixo em ~1,7 req/s. À direita, as três curvas de latência empilhadas na
ordem esperada e sem cruzamentos: P50 verde em ~4,2 ms, P95 laranja em ~7,1 ms,
P99 vermelho em ~7,5 ms, todas praticamente retas — carga constante, sem cauda.

**Faixa 3 — erro e modelo.** À esquerda, uma faixa vermelha preenchida oscilando
em torno de 8,5%, bem abaixo da linha tracejada de threshold em 20%. À direita,
um donut de cinco fatias com a legenda em tabela ao lado: a fatia de *nervous
system diseases* é visivelmente a maior (~29%), *general pathological
conditions* é a menor (~10%), e as outras três ficam próximas de 20% cada.

Antes da carga, os mesmos painéis mostram: `CARREGADO` no status do modelo,
`sem tráfego` nos indicadores de taxa e latência, lacunas nas séries temporais,
e o donut com as cinco fatias em zero — as séries existem desde o startup, então
nenhuma classe some do painel.

---

## Valores medidos, painel a painel

### 1 · Status do modelo — `stat`

```promql
api_model_loaded
```

| Série | Valor | Exibido |
|---|---|---|
| `instance="api:8000", job="api"` | `1` | **CARREGADO** (verde) |

### 2 · Requisições por segundo — `stat`

```promql
sum(rate(api_requests_total[1m]))
```

**30,09 req/s** — bate com o alvo de 30 req/s do gerador, confirmando que
nenhuma requisição está sendo perdida entre o cliente e o contador.

### 3 · Taxa de erro — `stat`

```promql
sum(rate(api_requests_total{status!~"2.."}[1m])) / sum(rate(api_requests_total[1m]))
```

**8,52%** (laranja, faixa de 5% a 20%). O gerador foi configurado com
`--error-rate 0.08`; a diferença é a variação estatística da amostragem
aleatória dentro da janela de 1 minuto.

### 4 · P99 de /predict — `stat`

```promql
histogram_quantile(0.99, sum by (le) (rate(api_request_duration_seconds_bucket{endpoint="/predict"}[1m])))
```

**7,45 ms** (verde, abaixo do threshold de 10 ms).

### 5 · Total de requisições — taxa por segundo — `timeseries`

```promql
sum by (endpoint, status) (rate(api_requests_total[1m]))
```

| Série | req/s |
|---|---|
| `/predict` — `200` | 25,86 |
| `/predict` — `422` | 2,56 |
| `/health` — `200` | 1,67 |

Três séries, exatamente as três que o gerador produz. `/metrics` não aparece —
está excluído da instrumentação de propósito, senão o scrape de 5 s do
Prometheus somaria ~0,2 req/s de tráfego que nenhum cliente gerou.

O `/health` fica um pouco acima dos 5% configurados porque o healthcheck do
próprio container também bate nesse endpoint a cada 10 s. É tráfego real e
aparece como tal.

### 6 · Latência de /predict — P50, P95, P99 — `timeseries`

| Quantil | Valor |
|---|---|
| P50 | 4,18 ms |
| P95 | 7,08 ms |
| P99 | 7,45 ms |

Média aritmética no mesmo período, de `_sum / _count`: **3,81 ms**
(20,056 s / 5.266 requisições). A média abaixo do P50 é consistente com a massa
concentrada entre 1 e 5 ms e uma cauda curta.

### 7 · Taxa de erro — série temporal — `timeseries`

**8,52%**, estável ao longo da janela. Sem tráfego, o denominador é zero, a
query devolve `NaN` e o painel mostra uma lacuna — preencher com zero afirmaria
"não há erros" quando o fato é "não há informação".

### 8 · Distribuição de predições por classe — `piechart`

```promql
sum by (label_name) (increase(api_predictions_total[15m]))
```

| Classe | Predições (15 min) | Fatia |
|---|---|---|
| nervous system diseases | 3.188 | 29,2% |
| cardiovascular diseases | 2.276 | 20,8% |
| neoplasms | 2.185 | 20,0% |
| digestive system diseases | 2.138 | 19,6% |
| general pathological conditions | 1.139 | 10,4% |

---

## O que o painel de distribuição pegou

O gerador envia dez textos válidos, dois por classe — a entrada é uniforme, 20%
por classe. A saída não é: *nervous system* recebe ~29% e *general pathological
conditions* ~10%.

O painel está certo, e o desvio tem uma causa concreta. Passando os dez textos
pela API uma vez cada:

| Classe pretendida | Classe prevista | Confiança |
|---|---|---|
| 1 neoplasms | 1 neoplasms | 0,91 |
| 1 neoplasms | 1 neoplasms | 0,87 |
| 2 digestive | 2 digestive | 0,84 |
| 2 digestive | 2 digestive | 0,56 |
| 3 nervous | 3 nervous | 0,53 |
| 3 nervous | 3 nervous | 0,82 |
| 4 cardiovascular | 4 cardiovascular | 0,96 |
| 4 cardiovascular | 4 cardiovascular | 0,95 |
| 5 general | 5 general | 0,62 |
| 5 general | **3 nervous** | **0,36** |

Nove dos dez caem na classe pretendida. O décimo — o texto de dor crônica em
acompanhamento de longo prazo — vai para *nervous system diseases* com
confiança 0,36, a mais baixa do conjunto. Daí exatamente os números do donut:
3/10 do tráfego vira classe 3 (29,2%) e 1/10 vira classe 5 (10,4%).

Vale registrar por dois motivos. Primeiro, é o comportamento esperado de um
baseline com macro-F1 de 0,671: "dor crônica" é vocabulário que aparece nas duas
classes, e o corpus rotula condição médica, não urgência. Segundo, e mais
relevante para esta etapa: **o painel expôs um erro de classificação sem que
ninguém tivesse ido olhar o modelo.** É precisamente a diferença entre
observabilidade de HTTP e observabilidade de modelo — as 5.266 requisições
foram `200`, o painel de erro ficou nos 8,5% dos payloads inválidos, e nada no
lado HTTP indicaria que uma em cada dez predições vai para a classe errada.

---

## Validação do ciclo `down` / `up`

O provisionamento por arquivo é o que faz o dashboard sobreviver. Verificado
com o ciclo completo, sem `-v`:

```console
$ curl -s -u admin:admin 'localhost:3000/api/dashboards/uid/tc-fase3-api' | ...
Tech Challenge Fase 3 — API de Triagem | 8 painéis

$ docker compose -f docker-compose.monitoring.yml down
 Container tc-grafana  Removed
 Container tc-prometheus  Removed
 Container tc-api  Removed
 Network tc-fase3-monitoring_monitoring  Removed

$ docker compose -f docker-compose.monitoring.yml up -d
 Container tc-api  Healthy
 Container tc-prometheus  Started
 Container tc-grafana  Started

$ curl -s -u admin:admin 'localhost:3000/api/dashboards/uid/tc-fase3-api' | ...
Tech Challenge Fase 3 — API de Triagem | 8 painéis
painéis: ['Status do modelo', 'Requisições por segundo', 'Taxa de erro',
          'P99 de /predict', 'Total de requisições — taxa por segundo',
          'Latência de /predict — P50, P95, P99',
          'Taxa de erro — proporção de respostas não-2xx',
          'Distribuição de predições por classe']

$ curl -s -u admin:admin localhost:3000/api/datasources | ...
prometheus prometheus http://prometheus:9090
```

Os oito painéis e o datasource voltam idênticos, sem nenhuma ação na UI.

A série histórica também sobrevive, porque `prometheus-data` é um volume
nomeado. Consultado logo depois do `up`, cobrindo a janela anterior ao `down`:

```
neoplasms                        => 2212
digestive system diseases        => 2263
nervous system diseases          => 3228
cardiovascular diseases          => 2147
general pathological conditions  => 1144
```

Com `docker compose down -v` o dashboard ainda voltaria — ele vem do arquivo —
mas os painéis começariam vazios. É a diferença entre o que é provisionado e o
que é dado.

---

## Conferência do JSON versionado

O que o Grafana serve foi comparado, painel a painel, com
`monitoring/grafana/dashboards/api-monitoring.json` — `uid`, título, `refresh`,
posição de cada painel e a lista de expressões PromQL de cada alvo:

```
IDÊNTICO
  #1 stat        Status do modelo
  #2 stat        Requisições por segundo
  #3 stat        Taxa de erro
  #4 stat        P99 de /predict
  #5 timeseries  Total de requisições — taxa por segundo
  #6 timeseries  Latência de /predict — P50, P95, P99
  #7 timeseries  Taxa de erro — proporção de respostas não-2xx
  #8 piechart    Distribuição de predições por classe
```

O Grafana 13.1.3 armazenou o dashboard no `schemaVersion 39` em que ele foi
escrito, sem migração. O arquivo do repositório é o JSON final: não há uma
versão exportada divergindo da versionada.
