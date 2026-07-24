# 99 — Mapa Tech Challenge Fase 3 → Aulas

> Cruza cada requisito/critério do Tech Challenge da Fase 3 com as aulas que o
> fundamentam. Base: enunciado em [07-tech-challenge-fase3.md](07-tech-challenge-fase3.md)
> e as 6 disciplinas transcritas nesta pasta.
> Data: 2026-07-23

## Legenda das disciplinas

| Código | Disciplina | Arquivo |
|---|---|---|
| **D1** | Deploy em Nuvem | [01-deploy-em-nuvem.md](01-deploy-em-nuvem.md) |
| **D2** | Integração com CI/CD (GitHub Actions) | [02-cicd-github-actions.md](02-cicd-github-actions.md) |
| **D3** | Pipeline de Treino e Deploy Automático | [03-pipeline-treino-deploy.md](03-pipeline-treino-deploy.md) |
| **D4** | Monitoração de Performance | [04-monitoracao-performance.md](04-monitoracao-performance.md) |
| **D5** | Serviços de Monitoração | [05-servicos-monitoracao.md](05-servicos-monitoracao.md) |
| **D6** | Latência e Performance em Dados Não Estruturados | [06-latencia-dados-nao-estruturados.md](06-latencia-dados-nao-estruturados.md) |

Notação: `D4·A2` = Disciplina D4, Aula 2.

---

## Mapa dos critérios de avaliação (100%)

| Requisito TC | Peso | Etapa | Aulas relevantes (disciplina · nº) | Ferramentas | Observações |
|---|---|---|---|---|---|
| **Modelagem e Otimização** — modelo NLP funcional, conversão/otimização (ONNX) e melhoria de latência demonstrada | **20%** | Etapa 4 | D6·A4 (pruning/quantização), D6·A5 (transfer learning/BERT), D6·A1 (fundamentos de latência, TTFT, custo de sequência), D6·A6 (GPU/TPU), D6·A7 (inferência distribuída); D4·A2 e D4·A3 (otimização de modelos supervisionados I/II — aplicável a TF-IDF+RandomForest); D4·A1 (latência vs. throughput, P95/P99); D3·A3 (treinamento e validação) | scikit-learn, TF-IDF, ONNX Runtime, quantização (int8), pruning, DistilBERT (opção) | Técnica mínima exigida: ONNX **ou** quantização **ou** pruning. Comparar latência original vs. otimizada (baseline da Etapa 1). ⚠️ A aula específica de NLP (D6·A2) está ausente — ver seção de lacuna. |
| **CI/CD (GitHub Actions)** — workflow rodando testes básicos | **15%** | Etapa 2 | D2·A1 (conceitos CI/CD em ML), D2·A2 (CI para Data Science), D2·A3 (pipeline CI/CD projeto ML), D2·A4 (testes automatizados/pytest), D2·A5 (containerização), D2·A6 (entrega contínua), D2·A7 (boas práticas MLOps); D3·A8 (CI/CD de ML); D3·A6 (qualidade de código/lint) | GitHub Actions, pytest, lint (flake8/ruff), Docker, `on: push`/`pull_request` | Boa prática obrigatória: ≥2 automações (ex.: lint + test). Fluxo sugerido: lint → test → build. |
| **Orquestração (Airflow)** — DAG de ingestão e treino | **15%** | Etapa 2 | D3·A5 (orquestração com Airflow), D3·A1 (introdução ao pipeline), D3·A2 (ingestão/feature engineering), D3·A7 (treino automático/re-treino); D4·A7 (orquestração e escalabilidade); D2·A8 (aprendizado contínuo) | Apache Airflow, DAG, PythonOperator, `.py` da DAG | DAG mínima: ler CSV → treinar → salvar modelo. Entregável: arquivo `.py` da DAG. |
| **Monitoramento** — Compose (API + Prometheus + Grafana) com dashboard | **20%** | Etapa 3 | D5·A1 (observabilidade em ML), D5·A2 (Prometheus/métricas), D5·A3 (Grafana/dashboards), D5·A4 (monitoramento de containers), D5·A7 (integração/comparativo), D5·A8 (boas práticas); D4·A8 (monitoramento e manutenção de modelos); D4·A1 (métricas de latência P50/P95/P99) | prometheus_client, Prometheus, Grafana, Docker Compose, PromQL | Boa prática obrigatória: dashboard com ≥3 painéis (ex.: total de requisições, latência, taxa de erro). Métricas mínimas na API: tempo de requisição e contagem de chamadas. |
| **Documentação (README)** — arquitetura de nuvem escolhida + instruções | **15%** | Etapa 1 | D1·A1 (visão geral do deploy em nuvem), D1·A2 (comportamento computacional/estratégias de deploy), D1·A3 (AWS), D1·A4 (Azure), D1·A5 (GCP), D1·A6 (FinOps/segurança); D4·A5 (batch vs. tempo real — decisão de serving) | README.md, AWS/Azure/GCP, Docker | Etapa 1 pede decisão **textual** de arquitetura (batch vs. real-time) no README, não deploy real na nuvem. Justificar a escolha para triagem clínica (baixa latência ⇒ real-time). |
| **Vídeo STAR** — demonstração técnica e impacto (≤5 min) | **15%** | Etapa 4 | Transversal: D1 (arquitetura — Action), D2/D3 (pipeline — Result), D4/D5 (monitoração — Action), D6·A1/D4·A1 (latência alcançada — Result) | Gravação de tela, roteiro STAR | Estrutura: Situation (triagem clínica) · Task (latência/CI/CD/monitoramento) · Action (arquitetura + otimização + monitoração) · Result (pipeline rodando + latência + lições). |

---

## Mapa dos requisitos obrigatórios de repositório

| Requisito obrigatório | Etapa | Aulas relevantes | Ferramentas |
|---|---|---|---|
| API REST de inferência (recebe laudo, retorna classificação) | 1 | D3·A4 (deploy inicial de modelos), D4·A1 (FastAPI + micro-batching) | FastAPI, uvicorn |
| Dockerfile funcional para o serviço | 1 | D1·A2 (estratégias de deploy), D2·A5 (containerização) | Docker |
| Baseline de latência local | 1 | D4·A1 (latência vs. throughput, percentis) | time, numpy, locust/ab (opcional) |
| Pipeline CI/CD (lint → test → build) | 2 | D2·A1–A7, D3·A8 | GitHub Actions |
| DAG Airflow de treino/retreino | 2 | D3·A5, D3·A7 | Airflow |
| Stack de monitoramento via Docker Compose | 3 | D5·A2, D5·A3, D5·A4 | Docker Compose, Prometheus, Grafana |
| Histórico de commits semântico | — | D3·A6 (qualidade/reprodutibilidade) | git, Conventional Commits |
| Otimização de performance (≥1 técnica) | 4 | D6·A4, D4·A2/A3 | ONNX Runtime, quantização, pruning |

---

## Lacuna: Aula 2 de NLP/Áudio

**O que falta:** A disciplina **D6 (Latência e Performance em Dados Não Estruturados)** tem 7 de 8 aulas. A **Aula 2 — "Desafios de Performance em NLP e Áudio"** não foi disponibilizada pela FIAP no portal (confirmado pelo aluno). Não é falha de extração.

**Por que é a lacuna de maior impacto:** O Tech Challenge é justamente um **classificador de texto (NLP)** de laudos médicos. A única aula dedicada especificamente a desafios de performance em **NLP** é a que está ausente. Temas que ela provavelmente cobriria e que impactam diretamente o critério de Modelagem e Otimização (20%): custo de tokenização, tamanho de vocabulário, comprimento de sequência, trade-off TF-IDF vs. embeddings e batching de inferência textual.

**Cobertura compensatória (confirmada no conteúdo das aulas disponíveis):**

| Tema da aula ausente | Onde está coberto (parcial) | O que a aula disponível efetivamente traz |
|---|---|---|
| Comprimento de sequência e custo de atenção | **D6·A1** (Fundamentos de Latência) | Discute Time To First Token (TTFT) e a complexidade em função do comprimento de sequência `n` e da dimensionalidade do embedding `d`; gargalos de Amdahl em etapas sequenciais de pré/pós-processamento. |
| Embeddings vs. modelos leves; transfer learning para texto | **D6·A5** (Transfer Learning) | Trata transfer learning em Deep Learning citando **BERT** (Devlin et al., 2019) e GPT-3; fine-tuning e dimensão intrínseca de tarefas de **NLP**. Base para optar por DistilBERT/embeddings vs. TF-IDF. |
| Compressão do modelo para reduzir latência | **D6·A4** (Pruning e Quantização) | Pruning e quantização (int8) — aplicáveis ao classificador de texto para atingir a melhoria de latência exigida. |
| Otimização do modelo supervisionado (TF-IDF + RandomForest) | **D4·A2 / D4·A3** (Otimização de Modelos Supervisionados I/II) | Técnicas de otimização de modelos supervisionados, aplicáveis diretamente ao modelo base sugerido no enunciado (TF-IDF + Random Forest). |
| Batching de inferência textual; latência vs. throughput | **D4·A1** (Latência vs. Throughput) | Micro-batching dinâmico com FastAPI, cálculo de percentis P50/P95/P99 com numpy — aplicável ao batching de requisições de texto. |
| Decisão de serving (real-time vs. batch) para texto | **D4·A5** (Previsões em Lote vs. Tempo Real) | Arquiteturas batch/speed/serving (Lambda/Kappa) e batch inference de embeddings; fundamenta a decisão de servir a triagem em tempo real. |

**Conclusão:** a ausência de D6·A2 **não bloqueia** nenhum entregável do TC. As técnicas necessárias (quantização/pruning/ONNX, batching, transfer learning, custo de sequência) estão distribuídas em D6·A1, D6·A4, D6·A5 e D4·A1/A2/A3/A5. Recomenda-se citar essa suplência no README ao justificar a otimização escolhida.

---

## Outras lacunas identificadas (requisitos do TC sem cobertura direta nas aulas)

| Requisito do TC | Situação nas aulas | Recomendação |
|---|---|---|
| **ONNX Runtime** (exemplo explícito de otimização no enunciado) | As aulas de D6 tratam quantização/pruning/transfer learning e aceleração por hardware, mas **não há um passo a passo verbatim de exportação para ONNX** nos PDFs extraídos. ONNX aparece como exemplo no enunciado, não como conteúdo de aula. | Seguir documentação oficial do `skl2onnx`/`onnxruntime`. Alternativamente, atender o critério via quantização (coberta em D6·A4). |
| **prometheus_client na API (instrumentação em código)** | D5 cobre Prometheus/Grafana conceitualmente e via configuração; o código de instrumentação em prosa foi reconstruído nos `.md`, mas **não há snippet verbatim completo de `prometheus_client` em FastAPI** nos PDFs. | Usar `prometheus_client` (Counter/Histogram) + `prometheus_fastapi_instrumentator`; apoiar-se em D4·A1 (FastAPI) e D5·A2 (métricas). |
| **Dataset de laudos médicos** | Nenhuma aula fornece/prepara o dataset; o enunciado sugere Medical Abstracts TC Corpus (Kaggle) ou MIMIC-III. | Item de dados do projeto, não de aula. Ver [07-tech-challenge-fase3.md](07-tech-challenge-fase3.md#dataset-sugerido). |
| **Roteiro/gravação do vídeo STAR** | Nenhuma aula ensina produção de vídeo; é entregável de comunicação. | Estruturar pelo método STAR do enunciado. |
| **Docker Compose multi-serviço (API+Prometheus+Grafana)** | D5 aborda os componentes; a composição exata dos três em um `docker-compose.yml` é montagem do projeto. | Base em D5·A2/A3/A4; compor os serviços com portas e `scrape_configs` do Prometheus. |

> Observação geral: os PDFs das disciplinas D1 (Deploy) e D2 (CI/CD) são majoritariamente conceituais — o código prático é referenciado em repositórios externos no GitHub e **não está nos PDFs**; por isso esses `.md` não contêm blocos de código. Ver detalhamento no [00-INDICE.md](00-INDICE.md).
