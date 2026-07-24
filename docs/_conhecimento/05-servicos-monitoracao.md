# Serviços de Monitoração
> Fonte: PDFs FIAP Pós Tech MLET — Fase 3 (Cloud and MLOps)
> Aulas extraídas: 8 de 8
> Data de extração: 2026-07-23

## Sumário
- [Aula 1 — Introdução à Observabilidade e Monitoramento em Sistemas de ML](#aula-1--introdução-à-observabilidade-e-monitoramento-em-sistemas-de-ml)
- [Aula 2 — Métricas e Monitoramento com Prometheus](#aula-2--métricas-e-monitoramento-com-prometheus)
- [Aula 3 — Visualização de Métricas e Dashboards com Grafana](#aula-3--visualização-de-métricas-e-dashboards-com-grafana)
- [Aula 4 — Monitoramento de Ambientes Containerizados com Prometheus](#aula-4--monitoramento-de-ambientes-containerizados-com-prometheus)
- [Aula 5 — Monitoramento em Nuvem com Azure Monitor](#aula-5--monitoramento-em-nuvem-com-azure-monitor)
- [Aula 6 — Monitoramento em Nuvem com Amazon CloudWatch](#aula-6--monitoramento-em-nuvem-com-amazon-cloudwatch)
- [Aula 7 — Integração Híbrida e Comparativo de Soluções de Monitoramento](#aula-7--integração-híbrida-e-comparativo-de-soluções-de-monitoramento)
- [Aula 8 — Tendências Avançadas e Melhores Práticas em Monitoramento de ML](#aula-8--tendências-avançadas-e-melhores-práticas-em-monitoramento-de-ml)

---

## Aula 1 — Introdução à Observabilidade e Monitoramento em Sistemas de ML
**Arquivo fonte:** `POSTECH - Aula 01.pdf` (11 páginas)
**Título na ementa:** Introdução à Observabilidade e Monitoramento em Sistemas de ML

### Conceitos-chave
- Três pilares da observabilidade (métricas, logs, traces)
- Monitoramento vs. observabilidade
- Especificidades de ML (data drift, concept drift)
- Ecossistema de ferramentas
- Plano de observabilidade

### Conteúdo

**O QUE VEM POR AÍ?**

Nesta aula, nosso objetivo é apresentar os conceitos fundamentais de observabilidade (métricas, logs, traces) e por que eles são críticos para sistemas de ML em produção.

Ao final desta aula, você será capaz de compreender e aplicar os três pilares da observabilidade; monitoramento vs. observabilidade; especificidades de ML; ecossistema de ferramentas; e plano de observabilidade.

**HANDS ON**

Agora, iremos elaborar um plano de observabilidade para um serviço de inferência — listar métricas (infra + modelo), definir ferramentas e alertas.

O plano deve incluir:
- Tabela de métricas por categoria (infraestrutura, modelo, negócio).
- Definição de thresholds de alerta por métrica.
- Escolha justificada de ferramentas (por que Prometheus e não Datadog?).
- Diagrama de arquitetura do stack de observabilidade.

**SAIBA MAIS**

A diferença fundamental entre monitoramento reativo (baseado em thresholds pré-definidos) e observabilidade proativa (baseada em exploração de dados) define como organizações respondem a incidentes em produção. Em sistemas de ML, essa distinção é ainda mais crítica porque falhas de modelo (data drift, concept drift) não geram erros tradicionais — o sistema continua respondendo HTTP 200, mas as predições degradam silenciosamente.

O conceito de observabilidade foi originalmente formalizado na teoria de controle por Rudolf Kálmán em 1960: um sistema é observável se, a partir de suas saídas, é possível reconstruir completamente seu estado interno. Transpondo para software, observabilidade significa que, dado o conjunto de métricas, logs e traces coletados, é possível diagnosticar qualquer comportamento anômalo sem necessidade de deploy adicional de instrumentação.

Os três pilares — métricas, logs e traces — não são redundantes. Métricas são numéricas, agregáveis e baratas de armazenar em time-series databases (TSDB); respondem "quanto" e "quando". Logs são registros textuais detalhados de eventos individuais; respondem "o que aconteceu" no nível de cada request. Traces conectam spans distribuídos entre serviços, revelando a anatomia completa de uma request multisserviço; respondem "onde" o tempo foi gasto.

Para sistemas de ML em produção, a observabilidade apresenta desafios únicos. O conceito de "correctness" é probabilístico — não há uma resposta deterministicamente certa contra a qual comparar a saída do modelo. Data drift (mudança na distribuição dos dados de entrada) e concept drift (mudança na relação entre features e target) degradam a performance do modelo sem gerar exceções ou erros visíveis na camada de aplicação.

A detecção de drift requer métricas estatísticas especializadas: Population Stability Index (PSI), Kullback-Leibler divergence, Kolmogorov-Smirnov test, Jensen-Shannon divergence. Cada métrica possui sensibilidades diferentes: PSI é intuitiva para stakeholders de negócio, enquanto KL divergence é assimétrica e mais sensível a mudanças nas caudas da distribuição.

Ferramentas modernas de observabilidade para ML convergem em duas categorias: (1) extensões de plataformas de observabilidade existentes (Prometheus + custom exporters, Grafana + ML plugins) e (2) plataformas dedicadas de ML monitoring (Evidently AI, WhyLabs, Arize, Fiddler). A primeira categoria integra-se melhor com a infraestrutura existente; a segunda oferece funcionalidades out-of-the-box para detecção de drift e explicabilidade.

**MERCADO, CASES E TENDÊNCIAS**

Estima-se que 85% das empresas que operam modelos em produção reportam pelo menos um incidente de degradação silenciosa por trimestre. O mercado de MLOps monitoring deve atingir US$4.2B até 2027 (MarketsandMarkets). Além disso, empresas como Uber (Michelangelo), Netflix (Metaflow), e Spotify (Hendrix) investem fortemente em observabilidade customizada para ML, enquanto startups como Evidently AI, WhyLabs e Arize democratizam essas capacidades para times menores.

**O QUE VOCÊ VIU NESTA AULA?**

Nesta aula abordamos os seguintes temas centrais: os três pilares da observabilidade; monitoramento vs. observabilidade; especificidades de ML; ecossistema de ferramentas e plano de observabilidade.

Na próxima aula, continuaremos explorando ferramentas e práticas de observabilidade para sistemas de ML em produção.

**REFERÊNCIAS**

- HUYEN, C. Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications. Sebastopol: O'Reilly Media, 2022.
- KÁLMÁN, R. E. On the General Theory of Control Systems. IRE Transactions on Automatic Control, v. 4, n. 3, 110-120, 1960.
- MULPURI, A. Monitoring Machine Learning Models in Production: Best Practices and Challenges. Journal of Scientific & Innovative Research, v. 12, n. 4, 112-128, 2023.
- SCULLEY, D. et al. Hidden Technical Debt in Machine Learning Systems. In: Advances in Neural Information Processing Systems 28 (NIPS 2015). Montreal, Canada, 2015. p. 2503–2511.

**PALAVRAS-CHAVE**

Observabilidade. MLOps. Drift. Monitoramento.

### Código e comandos
Nenhum bloco de código nesta aula.

### Ferramentas / serviços citados
- Prometheus, Grafana (com custom exporters / ML plugins)
- Evidently AI, WhyLabs, Arize, Fiddler (plataformas dedicadas de ML monitoring)
- Datadog (citado como comparação)
- Uber Michelangelo, Netflix Metaflow, Spotify Hendrix (observabilidade customizada)
- Métricas estatísticas de drift: PSI (Population Stability Index), KL divergence, Kolmogorov-Smirnov test, Jensen-Shannon divergence

### Aplicabilidade ao Tech Challenge Fase 3
- Fundamenta o requisito de observabilidade do classificador NLP: monitorar não só saúde de infra (HTTP 200) mas também degradação silenciosa do modelo (drift).
- Justifica a escolha de Prometheus + Grafana (extensão de plataforma existente com custom exporters) frente a plataformas SaaS, alinhado ao stack pedido no TC.
- Orienta a definição de tabela de métricas por categoria (infra, modelo, negócio) e thresholds de alerta para o serviço de inferência.

---

## Aula 2 — Métricas e Monitoramento com Prometheus
**Arquivo fonte:** `POSTECH - Aula 02.pdf` (11 páginas)
**Título na ementa:** Métricas e Monitoramento com Prometheus

### Conceitos-chave
- Arquitetura pull-based do Prometheus
- Tipos de métricas (Counter, Gauge, Histogram, Summary)
- Instrumentação de aplicação Python com `prometheus_client`
- Configuração do `prometheus.yml`
- PromQL básico

### Conteúdo

**O QUE VEM POR AÍ?**

Nesta aula, nosso objetivo é instalar e configurar o Prometheus para coletar métricas de aplicações de ML, compreendendo tipos de métricas e modelo de dados.

Ao final desta aula, você será capaz de compreender e aplicar a arquitetura do Prometheus; tipos de métricas; a instrumentação de aplicação Python; a configuração do prometheus.yml; e PromQL básico.

**HANDS ON**

Nesse hands on, iremos subir Prometheus + serviço de inferência via Docker Compose, instrumentar endpoint com prometheus_client e consultar métricas no Prometheus UI via PromQL.

Passo a passo:
- Criar FastAPI com endpoint `/predict` e `/metrics`.
- Adicionar `Counter`, `Histogram` e `Info` do `prometheus_client`.
- Configurar `prometheus.yml` com `scrape_config`.
- Docker Compose com dois serviços.
- Verificar scrape targets e executar queries PromQL.

**SAIBA MAIS**

O modelo pull-based do Prometheus inverte a responsabilidade: em vez de cada aplicação enviar métricas para um servidor central (push), o Prometheus solicita as métricas periodicamente. Essa arquitetura simplifica o deployment das aplicações (não precisam conhecer o endereço do servidor de métricas) e permite que o Prometheus controle a taxa de coleta.

O TSDB do Prometheus é otimizado para séries temporais com alta cardinalidade de labels, usando compressão por gorilla encoding (para valores float64) e indexação invertida (para labels). Cada série temporal é identificada unicamente pela combinação do nome da métrica e seus labels: `http_requests_total{method="POST", handler="/predict", status="200"}`. O storage engine organiza dados em blocos de duas horas, com WAL (Write-Ahead Log) para durabilidade.

Os quatro tipos de métricas refletem padrões fundamentais de instrumentação. Counter é ideal para eventos cumulativos — o valor nunca decresce, e `rate()` calcula a taxa de variação; Gauge captura estados instantâneos como temperatura, uso de memória ou número de goroutines ativas; Histogram distribui observações em buckets configuráveis — essencial para latência onde percentis (P50, P95, P99) importam mais que médias; e Summary calcula quantis no client-side, evitando a necessidade de buckets mas impedindo agregação entre instâncias.

Para instrumentação de APIs de ML em Python, a biblioteca `prometheus_client` oferece três padrões principais: (1) decorators (`@REQUEST_TIME.time()`) para medir duração de funções, (2) instrumentação explícita (`counter.inc()`, `histogram.observe()`) para eventos específicos e (3) `Info` e `Enum` para metadata estática (versão do modelo, variante de A/B test).

PromQL (Prometheus Query Language) é uma linguagem funcional para consultas em séries temporais. Operações fundamentais incluem `rate()` para calcular taxa de variação por segundo de um Counter, `histogram_quantile()` para derivar percentis de Histograms e operadores binários (`/`, `*`, `+`, `-`) para compor métricas derivadas como `error_rate = rate(errors) / rate(total)`. Subqueries permitem funções sobre ranges: `max_over_time(rate(requests_total[5m])[1h:])` retorna o pico de taxa de requests na última hora.

**MERCADO, CASES E TENDÊNCIAS**

O Prometheus é o segundo projeto graduado da CNCF após o Kubernetes, com adoção em mais de 75% das organizações que utilizam containers (CNCF Survey, 2023). Empresas como SoundCloud (criadores originais), DigitalOcean e Shopify operam instâncias com bilhões de séries temporais ativas. O ecossistema de exporters conta com mais de 700 integrações mantidas pela comunidade, cobrindo desde bancos de dados até hardware de rede.

**O QUE VOCÊ VIU NESTA AULA?**

Nesta aula abordamos os seguintes temas centrais: arquitetura do Prometheus; tipos de métricas; instrumentação de aplicação Python; configuração do prometheus.yml e PromQL básico.

Na próxima aula, continuaremos explorando ferramentas e práticas de observabilidade para sistemas de ML em produção.

**REFERÊNCIAS**

- BRAZIL, B. Prometheus: Up & Running. Sebastopol: O'Reilly Media, 2018.
- CHEN, M. et al. Time-Series Data Management for Large-Scale Monitoring Systems. IEEE Transactions on Knowledge and Data Engineering, v. 35, n. 8, 7890-7904, 2023.
- CNCF. Annual survey. 2023. Disponível em: https://www.cncf.io/reports/cncf-annual-survey-2023/. Acesso em: 19 mai. 2026.
- PROMETHEUS. Prometheus Documentation. 2024. Disponível em: https://prometheus.io/docs/introduction/overview/. Acesso em: 19 mai. 2026.

**PALAVRAS-CHAVE**

Prometheus. Séries Temporais. PromQL.

### Código e comandos

Série temporal identificada por nome + labels (exemplo do dump):

```promql
http_requests_total{method="POST", handler="/predict", status="200"}
```

Padrões de instrumentação com `prometheus_client` citados no dump:

```python
@REQUEST_TIME.time()
counter.inc()
histogram.observe()
```

Operações e queries PromQL citadas no dump:

```promql
rate(errors) / rate(total)
histogram_quantile()
max_over_time(rate(requests_total[5m])[1h:])
```

> [NOTA — não é conteúdo FIAP]: o dump não apresenta blocos de código completos; os itens acima são os snippets/queries exatos citados em prosa (endpoints `/predict` e `/metrics`; tipos `Counter`, `Histogram`, `Info`, `Enum`; configuração `prometheus.yml` com `scrape_config`).

### Ferramentas / serviços citados
- Prometheus (TSDB, gorilla encoding, WAL, blocos de 2h)
- `prometheus_client` (biblioteca Python)
- FastAPI, Docker Compose
- PromQL
- CNCF, SoundCloud, DigitalOcean, Shopify

### Aplicabilidade ao Tech Challenge Fase 3
- Núcleo do requisito de monitoramento: instrumentar a API do classificador NLP com `prometheus_client` expondo `/metrics` (Counter de requests, Histogram de latência, Info de versão do modelo).
- Configuração de `prometheus.yml` com `scrape_config` para coletar métricas do serviço de inferência via modelo pull-based.
- PromQL (`rate()`, `histogram_quantile()`) para derivar taxa de erro e percentis P95/P99 de latência que alimentarão dashboards e alertas.

---

## Aula 3 — Visualização de Métricas e Dashboards com Grafana
**Arquivo fonte:** `POSTECH - Aula 03.pdf` (10 páginas)
**Título na ementa:** Visualização de Métricas e Dashboards com Grafana

### Conceitos-chave
- Instalação e data sources
- Painéis essenciais (progressive disclosure, glanceability)
- Dashboard de infraestrutura ML
- Dashboard de modelo
- Template variables e alertas no Grafana

### Conteúdo

**O QUE VEM POR AÍ?**

Nesta aula, nosso objetivo é construir dashboards no Grafana conectados ao Prometheus, criando painéis para métricas de infra e de modelo.

Ao final desta aula você será capaz de compreender e aplicar instalação e data sources; painéis essenciais; dashboard de infraestrutura ML; dashboard de modelo; e alertas no Grafana.

**HANDS ON**

Iremos criar um dashboard Grafana com 6+ painéis: latência P95, throughput, CPU/memória, prediction distribution, alertas de latência > threshold. Exportar como JSON.

Arquitetura do dashboard:
- Row 1: Overview (stat panels - total requests, avg latency, error rate).
- Row 2: Infrastructure (time series - CPU, memory, network).
- Row 3: Model (time series - prediction distribution, drift score).
- Row 4: Alerts (alert list panel + annotation queries).

**SAIBA MAIS**

Dashboards eficazes seguem o princípio de progressive disclosure: o nível superior mostra status geral (verde/amarelo/vermelho) com estatísticas agregadas. O drill-down permite investigar serviço por serviço, endpoint por endpoint. Para ML, a camada adicional de métricas de modelo (drift, performance) deve estar visualmente integrada com métricas de infra para correlacionar degradação de modelo com eventos de infraestrutura.

A teoria de visualização de dados de Edward Tufte estabelece princípios aplicáveis: maximizar o data-ink ratio (proporção de tinta dedicada a dados vs. decoração), evitar chartjunk (elementos visuais sem informação) e usar small multiples (repetição do mesmo tipo de gráfico com diferentes dimensões) para comparações. No contexto de dashboards operacionais, o princípio de "glanceability" — a capacidade de extrair informação útil em menos de 5 segundos — deve guiar o design.

O Grafana implementa o conceito de template variables para dashboards parametrizáveis. Uma variable como `$service`, alimentada por uma query PromQL (`label_values(up, service)`), permite que o mesmo dashboard sirva para qualquer serviço instrumentado. Variables podem ser encadeadas: `$environment` filtra as opções disponíveis em `$service`, que filtra os endpoints disponíveis em `$endpoint`.

Para métricas de ML em Grafana, painéis especializados incluem: (1) Heatmap para distribuição de latência por hora, revelando padrões sazonais; (2) Time series com threshold annotations para marcar deployments de modelo; (3) Stat panels com sparklines para KPIs como accuracy e F1-score; (4) Table panels com conditional formatting para feature drift scores.

O sistema de alertas do Grafana Alerting (v8+) suporta multidimensional alerts: uma única rule pode avaliar condições para cada combinação de labels, gerando alertas individuais por serviço/endpoint. Notification policies permitem routing baseado em labels — alertas de infra vão para #ops-alerts, alertas de modelo vão para #ml-team. Contact points integram com PagerDuty, Slack, Teams, OpsGenie, e webhooks genéricos.

**MERCADO, CASES E TENDÊNCIAS**

O Grafana Labs atingiu US$240M ARR em 2024 com mais de 20 milhões de usuários da versão OSS e 5000+ clientes enterprise. Já o Grafana Cloud processa mais de 1.5 trilhões de métricas por dia. Casos notáveis incluem: Bloomberg (10.000+ dashboards para trading systems), JPMorgan Chase (observabilidade de infraestrutura bancária) e Tesla (monitoramento de frota de veículos). A aquisição do Asserts.ai em 2023 adicionou capabilities de entity-based observability.

**O QUE VOCÊ VIU NESTA AULA?**

Nesta aula abordamos os seguintes temas centrais: instalação e data sources; painéis essenciais; dashboard de infraestrutura ML; dashboard de modelo; e alertas no Grafana.

Na próxima aula, continuaremos explorando ferramentas e práticas de observabilidade para sistemas de ML em produção.

**REFERÊNCIAS**

- GRAFANA LABS. Grafana Documentation. 2024. Disponível em: https://grafana.com/docs/grafana/latest/. Acesso em: 19 mai. 2026.
- TUFTE, E. R. The Visual Display of Quantitative Information. Cheshire: Graphics Press, 2001.
- TURNBULL, J. Monitoring with Grafana: Track, Visualize, and Alert. New York: Apress.

**PALAVRAS-CHAVE**

Grafana. Visualização de Dados. Observabilidade.

### Código e comandos

Template variable via query PromQL citada no dump:

```promql
label_values(up, service)
```

> [NOTA — não é conteúdo FIAP]: variables encadeadas citadas em prosa: `$environment` → `$service` → `$endpoint`. O dashboard "6+ painéis" é descrito em texto (rows de Overview, Infrastructure, Model, Alerts) e não há JSON de dashboard no dump.

### Ferramentas / serviços citados
- Grafana (data sources, template variables, Grafana Alerting v8+, Heatmap, Time series, Stat panels, Table panels)
- Prometheus (data source)
- Contact points: PagerDuty, Slack, Teams, OpsGenie, webhooks
- Grafana Labs / Grafana Cloud / Asserts.ai

### Aplicabilidade ao Tech Challenge Fase 3
- Diretamente ligado ao requisito de dashboards Grafana: construir painéis de latência P95, throughput, CPU/memória e prediction distribution do classificador NLP, exportando como JSON.
- Uso de template variables (`$service`, `$environment`) para dashboards reutilizáveis e alertas de latência acima de threshold.
- Integração visual de métricas de modelo (drift score, distribuição de predições) com métricas de infra para correlação, apoiando a componente de observabilidade (20%) do TC.

---

## Aula 4 — Monitoramento de Ambientes Containerizados com Prometheus
**Arquivo fonte:** `POSTECH - Aula 04.pdf` (10 páginas)
**Título na ementa:** Monitoramento de Ambientes Containerizados com Prometheus

### Conceitos-chave
- Prometheus no Kubernetes (Operator Pattern)
- Service discovery automático
- Métricas de cluster (kube-state-metrics, cAdvisor)
- Recording rules e alerting rules (PrometheusRule)
- Custom metrics para HPA (Prometheus Adapter)

### Conteúdo

**O QUE VEM POR AÍ?**

Nesta aula, nosso objetivo é configurar Prometheus para ambientes Kubernetes — service discovery automático, métricas de pods e auto-scaling baseado em métricas.

Ao final desta aula você será capaz de compreender e aplicar Prometheus no Kubernetes; service discovery automático; métricas de cluster; recording rules e alerting rules; e custom metrics para HPA.

**HANDS ON**

Nesse hands on, veremos o deploy kube-prometheus-stack em cluster K8s (minikube), como configurar ServiceMonitor para serviço de inferência, alertas para latência alta e HPA baseado em custom metrics.

Passo a passo:
- `minikube start` com addon metrics-server.
- `helm install` kube-prometheus-stack.
- Deploy serviço de inferência com ServiceMonitor CRD.
- Criar PrometheusRule para alerta de latência.
- Instalar prometheus-adapter e configurar HPA.

**SAIBA MAIS**

A integração entre Prometheus e Kubernetes via Operator Pattern representa o estado da arte em observabilidade nativa de cloud. O Prometheus Operator traduz a intenção declarativa (ServiceMonitor YAML) em configuração imperativa (prometheus.yml scrape_configs), mantendo consistência mesmo quando pods escalam, morrem ou migram entre nodes.

O pattern de Custom Resource Definitions (CRDs) introduzido pelo Prometheus Operator estabeleceu um padrão seguido por dezenas de outros operadores. O ServiceMonitor define qual Service monitorar e com que parâmetros (interval, path, port), enquanto o PodMonitor faz o mesmo no nível de Pod, sem necessidade de Service. Já o PrometheusRule define recording e alerting rules como recursos Kubernetes, permitindo versionamento via GitOps.

O kube-state-metrics converte o estado dos objetos Kubernetes em métricas Prometheus: `kube_pod_status_phase{phase="Running"}`, `kube_deployment_spec_replicas`, `kube_node_status_condition{condition="Ready"}`. Combinadas com cadvisor (métricas de runtime de container), fornecem visão completa do cluster.

Para ML em Kubernetes, a combinação de HPA com custom metrics resolve o problema de auto-scaling baseado em métricas de negócio. O padrão CPU-based scaling é insuficiente para serviços de inferência: um modelo carregado em GPU pode ter CPU baixa mas estar saturado. Custom metrics como `inference_queue_depth` ou `prediction_latency_p95` refletem melhor a carga real.

O Prometheus Adapter funciona como um API server que traduz queries PromQL em respostas compatíveis com a Custom Metrics API do Kubernetes. A configuração mapeia métricas Prometheus para recursos Kubernetes: a métrica `http_request_duration_seconds` pode ser exposta como `pods/*/http_request_duration_seconds`, permitindo que o HPA referencie diretamente.

**MERCADO, CASES E TENDÊNCIAS**

O kube-prometheus-stack é o chart Helm mais instalado para observabilidade em Kubernetes, com mais de 50.000 stars no GitHub. Empresas como GitLab, Adidas e Zalando utilizam Prometheus Operator em produção com clusters de 1000+ nodes. A combinação com Thanos ou Cortex resolve o desafio de long-term storage e multi-cluster querying para organizações com dezenas de clusters.

**O QUE VOCÊ VIU NESTA AULA?**

Nesta aula abordamos os seguintes temas centrais: Prometheus no Kubernetes; Service discovery automático; métricas de cluster; recording rules e alerting rules e custom metrics para HPA.

Na próxima aula, continuaremos explorando ferramentas e práticas de observabilidade para sistemas de ML em produção.

**REFERÊNCIAS**

- BURNS, B. et al. Kubernetes: Up and Running. Sebastopol: O'Reilly Media.
- KUBERNETES DOCUMENTATION. Horizontal Pod Autoscaling. 2024. Disponível em: https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/. Acesso em: 19 mai. 2026.
- LINGESH, K. Prometheus Monitoring in Kubernetes Environments: Service Discovery and Scaling. International Journal of Computer Science Research, v. 8, n. 2, 45-58, 2025.

**PALAVRAS-CHAVE**

Prometheus Operator. Kubernetes. Custom Resource Definitions (CRDs).

### Código e comandos

Comandos do hands on citados no dump:

```bash
minikube start   # com addon metrics-server
helm install     # kube-prometheus-stack
```

Métricas de cluster (kube-state-metrics) citadas no dump:

```promql
kube_pod_status_phase{phase="Running"}
kube_deployment_spec_replicas
kube_node_status_condition{condition="Ready"}
```

Mapeamento de custom metric para a Custom Metrics API (Prometheus Adapter):

```
http_request_duration_seconds  ->  pods/*/http_request_duration_seconds
```

> [NOTA — não é conteúdo FIAP]: os comandos `minikube start` e `helm install` aparecem em prosa sem flags/args completos no dump; reproduzidos como citados. Custom metrics de exemplo: `inference_queue_depth`, `prediction_latency_p95`.

### Ferramentas / serviços citados
- Prometheus Operator, kube-prometheus-stack (Helm chart)
- CRDs: ServiceMonitor, PodMonitor, PrometheusRule
- kube-state-metrics, cAdvisor
- minikube, metrics-server, Helm
- prometheus-adapter, HPA (Horizontal Pod Autoscaler), Custom Metrics API
- Thanos, Cortex (long-term storage / multi-cluster)

### Aplicabilidade ao Tech Challenge Fase 3
- Se o classificador NLP for implantado em Kubernetes, ServiceMonitor CRD automatiza o scrape do endpoint `/metrics` sem editar `prometheus.yml` manualmente.
- PrometheusRule versiona alertas de latência via GitOps; custom metrics (ex.: `prediction_latency_p95`) habilitam HPA orientado a métricas de negócio, não só CPU.
- Base para observabilidade de infra containerizada complementando os dashboards Grafana exigidos.

---

## Aula 5 — Monitoramento em Nuvem com Azure Monitor
**Arquivo fonte:** `POSTECH - Aula 05.pdf` (10 páginas)
**Título na ementa:** Monitoramento em Nuvem com Azure Monitor

### Conceitos-chave
- Arquitetura do Azure Monitor (data plane unificado)
- Application Insights para ML (OpenTelemetry, auto-instrumentation)
- Custom metrics e events
- KQL (Kusto Query Language)
- Alertas e Action Groups

### Conteúdo

**O QUE VEM POR AÍ?**

Nesta aula, nosso objetivo é apresentar o Azure Monitor como uma plataforma de observabilidade integrada — Application Insights, Log Analytics e alertas para workloads de ML no Azure.

Ao final desta aula, você será capaz de compreender e aplicar a arquitetura do Azure Monitor; Application Insights para ML; custom metrics e events; KQL (Kusto Query Language) e alertas e Action Groups.

**HANDS ON**

Nesse hands on, iremos instrumentar o serviço de inferência com Application Insights SDK (Python), enviar custom metrics (latência, predições) e criar um alerta para latência > SLO via KQL.

Implementação:
- `pip install opencensus-ext-azure`.
- Configurar AzureExporter com `APPLICATIONINSIGHTS_CONNECTION_STRING`.
- Enviar custom metrics via TelemetryClient.
- Query em Log Analytics: `requests | where duration > 500`.
- Criar Alert Rule com Action Group (email + webhook).

> [NOTA — não é conteúdo FIAP]: o dump instrui `pip install opencensus-ext-azure` (SDK legado baseado em OpenCensus), enquanto o próprio texto (Saiba Mais) informa que "o SDK legado foi deprecado em favor de `azure-monitor-opentelemetry`" desde 2023. Há divergência entre o passo do hands on e a recomendação atual apresentada na mesma aula.

**SAIBA MAIS**

O Azure Monitor diferencia-se de soluções open-source (Prometheus+Grafana) por oferecer integração nativa com o ecossistema Azure: AKS, Azure ML, Cosmos DB, Event Hubs. Para workloads de ML no Azure, o Application Insights captura automaticamente o distributed trace desde o API Gateway até o endpoint de inferência, incluindo chamadas a serviços downstream.

A arquitetura do Azure Monitor segue o modelo de data plane unificado: todas as fontes de telemetria (VMs, containers, PaaS services, applications) convergem para dois stores — Azure Monitor Metrics (time-series database otimizado para queries em tempo real com retenção de 93 dias) e Log Analytics workspace (baseado em Azure Data Explorer/Kusto com retenção configurável até 2 anos).

O Application Insights utiliza o protocolo OpenTelemetry para instrumentação (desde 2023, o SDK legado foi deprecado em favor de azure-monitor-opentelemetry). O Auto-instrumentation detecta automaticamente frameworks como Flask, FastAPI, Django e gera spans para cada request HTTP, query de banco de dados e chamada HTTP outbound. Para ML, isso significa que o trace completo — desde o request de predição até a consulta ao feature store e ao modelo serializado — é capturado sem código adicional.

KQL (Kusto Query Language) é a linguagem de consulta para Log Analytics, inspirada em SQL mas otimizada para dados de telemetria. Operadores como `summarize`, `make-series` e `render` permitem análises complexas: anomalias em séries temporais (`series_decompose_anomalies`), previsão (`series_decompose_forecast`) e correlação entre tabelas (`join`, `union`).

O sistema de alertas do Azure Monitor suporta três tipos: (1) Metric alerts para condições em métricas numéricas (latência > 500ms por 5 minutos), (2) Log alerts para queries KQL que retornam resultados (count > threshold) e (3) Activity log alerts para eventos de plataforma (VM deallocated, deployment failed). Dynamic thresholds usam ML interno para detectar anomalias sem necessidade de definir valores estáticos.

**MERCADO, CASES E TENDÊNCIAS**

O Azure Monitor processa mais de 50 petabytes de dados de telemetria por dia. O Application Insights é utilizado por mais de 1 milhão de aplicações em produção. Casos notáveis incluem: Maersk (monitoramento de supply chain digital), Heineken (IoT de fábricas) e Stack Overflow (performance de API). A integração com Azure ML permite monitor drift automático em endpoints gerenciados.

**O QUE VOCÊ VIU NESTA AULA?**

Nesta aula abordamos os seguintes temas centrais: arquitetura do Azure Monitor; Application Insights para ML; custom metrics e events; KQL (Kusto Query Language); e alertas e Action Groups.

Na próxima aula, continuaremos explorando ferramentas e práticas de observabilidade para sistemas de ML em produção.

**REFERÊNCIAS**

- FERNÁNDEZ, J.; RAMÍREZ, L. Cloud Observability: Best Practices for Azure, AWS, and GCP. Birmingham: Packt Publishing, 2024.
- MICROSOFT. Azure Monitor Documentation. 2024. Disponível em: https://learn.microsoft.com/azure/azure-monitor/. Acesso em: 20 mai. 2026.
- MICROSOFT. Introdução ao Application Insights - Observabilidade do OpenTelemetry. 2024. Disponível em: https://learn.microsoft.com/azure/azure-monitor/app/app-insights-overview. Acesso em: 20 mai. 2026.

**PALAVRAS-CHAVE**

Azure Monitor. Application Insights. OpenTelemetry.

### Código e comandos

Passos do hands on citados no dump:

```bash
pip install opencensus-ext-azure
```

```
# variável de ambiente
APPLICATIONINSIGHTS_CONNECTION_STRING
```

Query KQL em Log Analytics citada no dump:

```kql
requests | where duration > 500
```

Operadores e funções KQL citados: `summarize`, `make-series`, `render`, `series_decompose_anomalies`, `series_decompose_forecast`, `join`, `union`.

### Ferramentas / serviços citados
- Azure Monitor (Metrics, Log Analytics workspace, data plane unificado)
- Application Insights (OpenTelemetry, auto-instrumentation, TelemetryClient, AzureExporter)
- SDK Python: `opencensus-ext-azure` (legado) / `azure-monitor-opentelemetry` (recomendado)
- KQL (Kusto Query Language), Azure Data Explorer
- Action Groups (email + webhook), Dynamic thresholds
- AKS, Azure ML, Cosmos DB, Event Hubs; frameworks Flask, FastAPI, Django

### Aplicabilidade ao Tech Challenge Fase 3
- Alternativa cloud-native (Azure) para monitorar o classificador NLP; auto-instrumentation do FastAPI gera traces sem código adicional.
- Custom metrics (latência, predições) via TelemetryClient e alertas KQL (`duration > 500`) para SLO, comparáveis ao stack Prometheus/Grafana.
- Dynamic thresholds e monitor de drift em endpoints Azure ML apoiam a detecção de degradação silenciosa citada no TC.

---

## Aula 6 — Monitoramento em Nuvem com Amazon CloudWatch
**Arquivo fonte:** `POSTECH - Aula 06.pdf` (10 páginas)
**Título na ementa:** Monitoramento em Nuvem com Amazon CloudWatch

### Conceitos-chave
- CloudWatch Metrics e namespaces isolados
- Custom Metrics (boto3, dimensions)
- CloudWatch Logs e Logs Insights
- Dashboards centralizados
- Alarmes, automação e Anomaly Detection

### Conteúdo

**O QUE VEM POR AÍ?**

Nesta aula, nosso objetivo é explorar AWS CloudWatch como serviço de observabilidade — métricas, logs, dashboards e alarmes para workloads de ML na AWS.

Ao final desta aula você será capaz de compreender e aplicar CloudWatch Metrics; Custom Metrics; CloudWatch Logs e Insights; dashboards centralizados e alarmes e automação.

**HANDS ON**

Nesse hands on, iremos publicar custom metrics de modelo via boto3, criar dashboard CloudWatch com latência + throughput + predições e configurar alarme de latência com notificação SNS.

Implementação:
- `boto3.client('cloudwatch').put_metric_data()` com namespace MLInference.
- Dimensões: ModelVersion, Environment.
- Criar dashboard JSON com widgets de métricas.
- `put_metric_alarm()` com threshold e SNS action.
- Simular carga e verificar se o alarme dispara.

**SAIBA MAIS**

O CloudWatch Anomaly Detection utiliza Machine Learning internamente para criar bandas de comportamento normal baseadas em padrões históricos (sazonalidade diária, semanal). Quando uma métrica de modelo sai da banda esperada, o alarme dispara sem necessidade de definir thresholds estáticos — ideal para métricas de ML cuja baseline muda conforme o modelo é atualizado.

A arquitetura do CloudWatch segue o modelo de namespaces isolados: cada serviço AWS publica métricas em seu próprio namespace (AWS/EC2, AWS/SageMaker, AWS/Lambda). Custom metrics utilizam namespaces definidos pelo usuário; dimensions são key-value pairs que identificam unicamente uma série temporal dentro do namespace.

Para ML na AWS, SageMaker Endpoints publicam automaticamente métricas de invocation (InvocationsPerInstance, ModelLatency, OverheadLatency, Invocations, Invocation4XXErrors, Invocation5XXErrors). Combinadas com custom metrics do modelo (prediction_confidence, drift_score), fornecem visão completa de saúde do endpoint.

Já CloudWatch Logs Insights é uma engine de consulta serverless para logs estruturados. A sintaxe proprietária (não SQL, não KQL) oferece operadores como `fields`, `filter`, `stats`, `sort`, `parse`. O Pattern detection automático (`pattern`) identifica templates de log sem configuração prévia. Para debugging de ML, queries como `filter @message like /drift/ | stats count(*) by bin(5m)` revelam frequência de detecções de drift.

O modelo de preços do CloudWatch incentiva agregação: métricas standard (1-minuto) são gratuitas para serviços AWS, mas custom metrics custam US$0.30/métrica/mês. High-resolution (1-segundo) custa adicional. Dashboards: US$3/dashboard/mês. Alarmes: US$0.10/alarm/mês. Este modelo econômico difere fundamentalmente do Prometheus (grátis, self-hosted) e deve informar decisões de instrumentação.

**MERCADO, CASES E TENDÊNCIAS**

O AWS CloudWatch monitora mais de 10 trilhões de data points por dia (re:Invent 2024) e o SageMaker Model Monitor integra com CloudWatch para detecção automática de data drift e model quality drift em endpoints gerenciados. Empresas como Lyft, Airbnb e Capital One combinam CloudWatch com soluções complementares (Datadog, New Relic) para observabilidade multi-layer.

**O QUE VOCÊ VIU NESTA AULA?**

Nesta aula abordamos os seguintes temas centrais: CloudWatch Metrics; Custom Metrics; CloudWatch Logs e Insights; dashboards centralizados e alarmes e automação.

Na próxima aula, continuaremos explorando ferramentas e práticas de observabilidade para sistemas de ML em produção.

**REFERÊNCIAS**

- AWS. Amazon CloudWatch Documentation. 2024. Disponível em: https://docs.aws.amazon.com/cloudwatch/. Acesso em: 20 mai. 2026.
- AWS. Amazon CloudWatch White Paper. Disponível em: https://spireminds.com/wp-content/uploads/2024/12/Amazon_CloudWatch_White_Paper.pdf. Acesso em: 20 mai. 2026.
- BARR, J. New – Amazon CloudWatch Anomaly Detection. 2019. Disponível em: https://aws.amazon.com/blogs/aws/new-amazon-cloudwatch-anomaly-detection/. Acesso em: 20 mai. 2026.

**PALAVRAS-CHAVE**

Amazon CloudWatch. Machine Learning Observability. Anomalias.

### Código e comandos

Publicação de custom metrics e alarme via boto3 (citados no dump):

```python
boto3.client('cloudwatch').put_metric_data()   # namespace MLInference; Dimensões: ModelVersion, Environment
put_metric_alarm()                              # com threshold e SNS action
```

Query CloudWatch Logs Insights citada no dump:

```
filter @message like /drift/ | stats count(*) by bin(5m)
```

Operadores de Logs Insights citados: `fields`, `filter`, `stats`, `sort`, `parse`, `pattern`.

> [NOTA — não é conteúdo FIAP]: o dashboard é descrito como "dashboard JSON com widgets de métricas" sem o JSON literal no dump.

### Ferramentas / serviços citados
- Amazon CloudWatch (Metrics, Logs, Logs Insights, Dashboards, Alarms, Anomaly Detection)
- boto3 (`put_metric_data`, `put_metric_alarm`)
- Namespaces: AWS/EC2, AWS/SageMaker, AWS/Lambda, MLInference (custom)
- SageMaker Endpoints / SageMaker Model Monitor (métricas: InvocationsPerInstance, ModelLatency, OverheadLatency, Invocations, Invocation4XXErrors, Invocation5XXErrors)
- SNS (notificação)
- Datadog, New Relic (observabilidade complementar)

### Aplicabilidade ao Tech Challenge Fase 3
- Alternativa cloud-native (AWS) para o classificador NLP: publicar custom metrics (latência, predições, `prediction_confidence`, `drift_score`) via boto3 no namespace `MLInference`.
- Alarmes com threshold + SNS e Anomaly Detection cobrem o requisito de alertas; comparação de custo com Prometheus (grátis, self-hosted) informa decisão de stack.
- Dashboards CloudWatch (latência + throughput + predições) espelham os painéis Grafana pedidos, úteis se o deploy for em AWS/SageMaker.

---

## Aula 7 — Integração Híbrida e Comparativo de Soluções de Monitoramento
**Arquivo fonte:** `POSTECH - Aula 07.pdf` (10 páginas)
**Título na ementa:** Integração Híbrida e Comparativo de Soluções de Monitoramento

### Conceitos-chave
- Desafios de observabilidade multi-nuvem
- Prometheus como camada unificada (`remote_write`, Thanos, Cortex)
- Grafana como painel único
- Exporters e integrações (CloudWatch Exporter, Azure Monitor Exporter)
- Estratégia de observabilidade unificada e naming conventions

### Conteúdo

**O QUE VEM POR AÍ?**

Nesta aula, nosso objetivo é combinar ferramentas de observabilidade em cenários híbridos (on-premises + cloud) e multi-nuvem (AWS + Azure + GCP).

Ao final desta aula, você será capaz de compreender e aplicar desafios de observabilidade multi-nuvem; Prometheus como camada unificada; Grafana como painel único; exporters e integrações e estratégia de observabilidade unificada.

**HANDS ON**

Agora, iremos configurar Grafana com data sources de Prometheus e CloudWatch (simulado) e um dashboard híbrido: métricas de serviço on-premises e cloud no mesmo painel.

Arquitetura:
- Grafana com três data sources: Prometheus local, CloudWatch (mock via localstack), Azure Monitor (mock).
- Dashboard com mixed queries que combinam origens.
- Template variable `$cloud` para filtrar por provedor.
- Alertas unificados independentemente da origem.

**SAIBA MAIS**

A estratégia de observabilidade unificada para ambientes multi-nuvem requer uma 'língua franca' de telemetria. Prometheus com `remote_write` para Thanos ou Cortex estabelece essa camada: independentemente de onde o workload roda (AWS EKS, Azure AKS, on-prem K8s), as métricas convergem para o mesmo formato e query language (PromQL).

O Thanos resolve três limitações fundamentais do Prometheus standalone: (1) retenção limitada — Thanos Store Gateway integra com object storage (S3, GCS, Azure Blob) para retenção ilimitada; (2) single-cluster view — Thanos Query agrega dados de múltiplos Prometheus com deduplicação; (3) downsampling — Thanos Compactor reduz granularidade de dados históricos (5m → 1h para dados > 30 dias).

O Cortex oferece uma alternativa com um trade-off diferente: multi-tenancy nativo (isolamento por tenant_id), horizontal scaling de todos os componentes (ingester, querier, compactor) e compatibilidade total com PromQL remote read/write. Organizações com requisitos de isolamento entre times ou clientes preferem Cortex.

Para cenários multi-nuvem reais, o pattern recomendado é: (1) cada cluster/região exporta via `remote_write` para um Thanos receive centralizado; (2) Grafana aponta para Thanos Query como data source principal; (3) CloudWatch Exporter e Azure Monitor Exporter rodam como sidecars nos clusters de cada provedor, convertendo métricas proprietárias para o formato Prometheus antes do `remote_write`.

O desafio de naming conventions é frequentemente subestimado. Sem padronização, a mesma métrica conceitual (latência de inferência) pode aparecer como `prediction_latency_seconds` (Prometheus), `InvocationModelLatency` (CloudWatch) e `requests/duration` (Application Insights). Uma taxonomy definida antecipadamente — com labels consistentes (service, environment, region, model_version) — é pré-requisito para dashboards cross-cloud.

**MERCADO, CASES E TENDÊNCIAS**

Segundo uma pesquisa da Flexera (2024) 89% das organizações adotam estratégia multi-nuvem, mas apenas 32% reportam ter observabilidade unificada cross-cloud. O mercado de observability platforms (Datadog, Splunk, New Relic, Dynatrace) cresce 20% ao ano, com diferenciação cada vez mais baseada em capabilities de correlação multi-source e AIOps. O Grafana Labs posiciona-se como a alternativa open-source/hybrid que evita vendor lock-in.

**O QUE VOCÊ VIU NESTA AULA?**

Nesta aula abordamos os seguintes temas centrais: desafios de observabilidade multi-nuvem; Prometheus como camada unificada; Grafana como painel único; exporters e integrações e estratégia de observabilidade unificada.

Na próxima aula, continuaremos explorando ferramentas e práticas de observabilidade para sistemas de ML em produção.

**REFERÊNCIAS**

- BANERJEE, S. Multi-Cloud Observability: Challenges and Architectural Patterns. International Journal of Scientific Research, v. 14, n. 9, 23-38, 2025.
- FLEXERA. 2024 State of the Cloud Report. 2024. Disponível em: https://www.flexera.com/state-of-the-cloud. Acesso em: 20 mai. 2026.
- THANOS AUTHORS. Thanos Documentation. 2024. Disponível em: https://thanos.io/tip/thanos/design.md/. Acesso em: 20 mai. 2026.

**PALAVRAS-CHAVE**

Observabilidade Multi-Cloud. Prometheus. Thanos.

### Código e comandos

Convenção de nomes — mesma métrica conceitual em diferentes plataformas (citada no dump):

```
prediction_latency_seconds   # Prometheus
InvocationModelLatency       # CloudWatch
requests/duration            # Application Insights
```

Labels consistentes recomendados: `service`, `environment`, `region`, `model_version`.

> [NOTA — não é conteúdo FIAP]: `remote_write` e demais componentes (Thanos receive/Query/Store Gateway/Compactor; Cortex ingester/querier/compactor) são citados em prosa, sem YAML de configuração no dump.

### Ferramentas / serviços citados
- Prometheus (`remote_write`, PromQL)
- Thanos (Store Gateway, Query, Compactor, receive) — object storage S3/GCS/Azure Blob
- Cortex (multi-tenancy, horizontal scaling)
- Grafana (múltiplos data sources, mixed queries, template variable `$cloud`)
- CloudWatch Exporter, Azure Monitor Exporter (sidecars)
- localstack (mock de CloudWatch)
- AWS EKS, Azure AKS, on-prem K8s
- Datadog, Splunk, New Relic, Dynatrace, Grafana Labs

### Aplicabilidade ao Tech Challenge Fase 3
- Grafana como painel único combinando data sources (Prometheus local + cloud) suporta um dashboard híbrido para o classificador NLP.
- Reforça a padronização de naming/labels (`service`, `environment`, `model_version`) essencial para dashboards e alertas coerentes no TC.
- Thanos/`remote_write` são caminho para retenção de longo prazo das métricas do modelo caso o TC exija histórico.

---

## Aula 8 — Tendências Avançadas e Melhores Práticas em Monitoramento de ML
**Arquivo fonte:** `POSTECH - Aula 08.pdf` (10 páginas)
**Título na ementa:** Tendências Avançadas e Melhores Práticas em Monitoramento de ML

### Conceitos-chave
- OpenTelemetry (OTel) como padrão aberto
- OpenTelemetry Collector (receivers, processors, exporters)
- Distributed tracing para ML
- AIOps (anomaly detection, event correlation, root cause analysis, predictive maintenance)
- Convergência de model monitoring com infrastructure monitoring

### Conteúdo

**O QUE VEM POR AÍ?**

Nesta aula, nosso objetivo é apresentar as tendências emergentes em observabilidade — OpenTelemetry como padrão aberto e AIOps para automação inteligente de operações.

Ao final desta aula, você será capaz de compreender e aplicar OpenTelemetry (OTel); instrumentação com OTel; distributed tracing para ML; AIOps e futuro da observabilidade em ML.

**HANDS ON**

Nesse hands on, iremos instrumentar serviço de inferência com OpenTelemetry SDK (Python), coletar traces distribuídos, visualizar no Jaeger e comparar com a abordagem Prometheus pura.

Implementação:
- `pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-jaeger`.
- Configurar TracerProvider com JaegerExporter.
- Criar spans para cada etapa do pipeline de inferência.
- Docker Compose: app + Jaeger (all-in-one) + Prometheus.
- Comparar: traces vs métricas para debugging de latência.

**SAIBA MAIS**

O OpenTelemetry representa a convergência do ecossistema de observabilidade em 'um padrão único. Antes, organizações precisavam escolher entre OpenTracing (para traces) e OpenCensus (para métricas) com vendor lock-in nos backends. Com OTel, a instrumentação é feita uma vez e os dados podem ser exportados para qualquer backend via configuração do Collector.

O OpenTelemetry Collector é o componente central da arquitetura: um processo independente que recebe telemetria (receivers), processa (processors) e exporta (exporters) para múltiplos backends simultaneamente. Receivers suportam OTLP, Jaeger, Zipkin, Prometheus. Processors incluem batching, filtering, sampling, e attribute manipulation. Exporters enviam para qualquer backend compatível.

Para instrumentação de pipelines de ML, o OTel oferece uma vantagem única sobre o Prometheus: traces capturam a estrutura hierárquica de uma request. Um trace de inferência pode mostrar: HTTP handler (50ms) → feature extraction (15ms) → model.predict() (30ms) → post-processing (5ms). Se feature extraction faz query a um feature store externo, esse span filho revela a latência de rede. Prometheus métricas mostram apenas a latência total ou por stage agregada — não por request individual.

A AIOps (Artificial Intelligence for IT Operations) aplica técnicas de ML para automatizar tarefas operacionais. Quatro capacidades fundamentais: (1) Anomaly detection — identificar comportamentos atípicos em séries temporais usando isolation forests, autoencoders, ou modelos sazonais; (2) Event correlation — agrupar alertas relacionados usando graph-based clustering, reduzindo alert fatigue; (3) Root cause analysis — inferir causa raiz usando causal inference ou graph traversal em dependency maps; (4) Predictive maintenance — prever falhas futuras usando forecasting de séries temporais.

A convergência de model monitoring com infrastructure monitoring é a tendência mais significativa para MLOps: em vez de ferramentas separadas para "o modelo está degradando?" e "a infra está saudável?", uma visão unificada correlaciona automaticamente — drift detectado coincide com deploy de nova versão de feature store? Latência subiu porque o modelo cresceu de tamanho após retrain?

**MERCADO, CASES E TENDÊNCIAS**

OpenTelemetry é o segundo projeto mais ativo da CNCF (após Kubernetes) com mais de 1000 contribuidores. A adoção cresceu 300% entre 2022-2024 (CNCF Survey). Vendors como Datadog, New Relic, Splunk e Dynatrace agora aceitam OTLP nativamente, reduzindo lock-in. O mercado de AIOps deve atingir US$40B até 2026, com players como Moogsoft, BigPanda e PagerDuty Intelligent Triage liderando. A integração de LLMs para incident summarization e runbook suggestion é o frontier atual.

**O QUE VOCÊ VIU NESTA AULA?**

Nesta aula abordamos os seguintes temas centrais: OpenTelemetry (OTel); instrumentação com OTel; distributed tracing para ML; AIOps e futuro da observabilidade em ML.

Esta foi a última aula da disciplina. Você agora possui uma visão completa do ecossistema de observabilidade para ML — desde os fundamentos (Prometheus, Grafana) até tendências emergentes (OpenTelemetry, AIOps).

**REFERÊNCIAS**

- BEYER, B. et al. The Site Reliability Workbook: Practical Ways to Implement SRE. Sebastopol: O'Reilly Media, 2018.
- DANG, Y. et al. AIOps: Real-World Challenges and Research Innovations. [s.l.]: ICSE-SEIP, 2019.
- LEEST, R. et al. OpenTelemetry in Production: Lessons from Large-Scale Deployments. arXiv:2501.xxxxx, 2025.
- OPENTELEMETRY AUTHORS. OpenTelemetry Documentation. 2024. Disponível em: https://opentelemetry.io/docs/. Acesso em: 20 mai. 2026.

**PALAVRAS-CHAVE**

OpenTelemetry. OTLP. AIOps.

### Código e comandos

Instalação do SDK OpenTelemetry citada no dump:

```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-jaeger
```

Estrutura hierárquica de um trace de inferência (exemplo do dump):

```
HTTP handler (50ms) → feature extraction (15ms) → model.predict() (30ms) → post-processing (5ms)
```

> [NOTA — não é conteúdo FIAP]: `arXiv:2501.xxxxx` aparece assim no dump (identificador placeholder, sem número completo). A configuração de `TracerProvider`/`JaegerExporter` e o Docker Compose (app + Jaeger all-in-one + Prometheus) são descritos em prosa, sem código literal.

### Ferramentas / serviços citados
- OpenTelemetry (OTel), OTLP, OpenTelemetry Collector (receivers/processors/exporters)
- OpenTracing, OpenCensus (predecessores)
- Jaeger, Zipkin (backends de tracing); Prometheus (receiver/backend)
- SDK Python: `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-jaeger`; TracerProvider, JaegerExporter
- AIOps: isolation forests, autoencoders, graph-based clustering, causal inference, forecasting
- Moogsoft, BigPanda, PagerDuty Intelligent Triage; Datadog, New Relic, Splunk, Dynatrace

### Aplicabilidade ao Tech Challenge Fase 3
- Distributed tracing (OTel + Jaeger) complementa Prometheus/Grafana ao revelar latência por etapa do pipeline de inferência do classificador NLP (feature extraction, `model.predict()`, post-processing).
- Convergência model + infra monitoring embasa correlação entre drift do modelo e eventos de deploy/infra, reforçando a observabilidade (20%) do TC.
- OTLP como padrão aberto evita vendor lock-in e mantém compatibilidade com o backend Prometheus já usado no projeto.

---
