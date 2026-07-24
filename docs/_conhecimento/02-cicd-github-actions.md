# Integração com CI/CD (GitHub Actions)
> Fonte: PDFs FIAP Pós Tech MLET — Fase 3 (Cloud and MLOps)
> Aulas extraídas: 8 de 8
> Data de extração: 2026-07-23

## Sumário
- [Aula 1 — Conceitos de CI/CD aplicados a ML](#aula-1--conceitos-de-cicd-aplicados-a-ml)
- [Aula 2 — Integração Contínua para Data Science com GitHub Actions](#aula-2--integração-contínua-para-data-science-com-github-actions)
- [Aula 3 — Pipeline CI/CD com GitHub Actions (Projeto de ML)](#aula-3--pipeline-cicd-com-github-actions-projeto-de-ml)
- [Aula 4 — Testes Automatizados no Pipeline de ML](#aula-4--testes-automatizados-no-pipeline-de-ml)
- [Aula 5 — Automatizando o Deploy de Modelos (Parte 1: Containerização)](#aula-5--automatizando-o-deploy-de-modelos-parte-1-containerização)
- [Aula 6 — Automatizando o Deploy de Modelos (Parte 2: Entrega Contínua)](#aula-6--automatizando-o-deploy-de-modelos-parte-2-entrega-contínua)
- [Aula 7 — Boas Práticas de CI/CD em ML (MLOps)](#aula-7--boas-práticas-de-cicd-em-ml-mlops)
- [Aula 8 — Aprendizado Contínuo e Monitoramento de Modelos](#aula-8--aprendizado-contínuo-e-monitoramento-de-modelos)

---

## Aula 1 — Conceitos de CI/CD aplicados a ML
**Arquivo fonte:** `9MLET - Fase 3 - Integração com CICD - Aula 1.pdf` (17 páginas)
**Título na ementa:** Conceitos de CI/CD aplicados a ML

### Conceitos-chave
- Transição do DevOps tradicional para o MLOps.
- As quatro fontes de variabilidade em ML: Código, Dados, Hiperparâmetros e Ambiente Computacional.
- Anatomia do GitHub Actions: workflows, jobs, steps, runners e gatilhos (push / pull request).
- Quality Gates, Branch Protection Rules e contratos técnicos para merge seguro.
- Estratégia de feedback rápido (Shift-Left Testing) vs. validação profunda.
- Observabilidade, resposta a incidentes e segurança da cadeia de suprimentos (DevSecOps para ML).

### Conteúdo

**O QUE VEM POR AÍ?**

Você já tentou rodar o código de um colega e ele simplesmente quebrou por causa de uma versão diferente do Python ou de uma biblioteca?

Em Machine Learning, o "funciona na minha máquina" é um problema exponencial, pois o artefato final depende não apenas do código, mas também dos dados de treinamento e do ambiente computacional.

Nesta primeira aula, vamos quebrar o paradigma do DevOps tradicional e introduzir o MLOps. Você entenderá por que ferramentas de Integração Contínua (CI) não podem apenas compilar o código em projetos de IA; elas precisam orquestrar a reprodutibilidade. Vamos mergulhar no GitHub Actions como o nosso motor central de automação e engenharia de entrega.

**HANDS ON**

Nesta etapa prática, vamos construir o nosso primeiro pipeline de automação do zero. Você verá como criar a estrutura `.github/workflows` e escrever o seu primeiro arquivo YAML. Ao final, você terá um pipeline que reage automaticamente a Push e Pull Requests, executando Jobs e Steps em um Runner (máquina virtual) do Ubuntu para validar a sintaxe básica de um script Python.

**SAIBA MAIS**

Nesta primeira parte da disciplina, analisaremos as fundações arquiteturais e o controle de variabilidade no ciclo de vida de IA, utilizando cenários reais de engenharia para ilustrar a transição para plataformas de MLOps estruturadas.

**Transição Arquitetural: Do DevOps Tradicional ao MLOps**

Uma organização de tecnologia madura possui processos robustos de DevOps para suas aplicações web tradicionais, onde o código compilado possui comportamento determinístico (se passa nos testes, funciona em produção). Ao iniciar o desenvolvimento de produtos de Inteligência Artificial, a empresa tenta forçar os modelos de Machine Learning pela mesma esteira de CI/CD tradicional. O resultado é caótico: artefatos não determinísticos (modelos) são promovidos para produção baseados apenas em intuição ou na aprovação subjetiva ("looks good to me") de um(a) Cientista de Dados Sênior. Quando o modelo degrada em produção e gera prejuízos financeiros, a equipe de operações não consegue explicar a causalidade da falha, pois não há histórico de qual versão dos dados, código ou parâmetros gerou aquele artefato específico.

Redesenhar a arquitetura de entrega de software para tratar as especificidades de Machine Learning como elementos de primeira classe. O objetivo é remover a variabilidade humana do processo de release, transformando o comportamento desejado de engenharia em verificações automatizadas com critérios matemáticos e explícitos de aprovação, garantindo total rastreabilidade.

A equipe de MLOps formaliza a nova arquitetura utilizando o GitHub Actions como motor central de orquestração. Primeiramente, define-se que nenhuma mudança no comportamento do modelo ocorre sem o registro prévio em uma documentação viva, utilizando Architecture Decision Records (ADRs) armazenados no próprio repositório. Em seguida, os workflows YAML do GitHub Actions são reescritos. Em vez de apenas compilar código, o pipeline agora executa Jobs que comparam automaticamente as métricas de performance do novo modelo (ex: ganho de precisão) contra o modelo atualmente em produção (modelo champion vs challenger). Se o risco técnico ultrapassar o limite de aceitação predefinido nas variáveis de ambiente do repositório, o GitHub Actions bloqueia sumariamente o merge, independentemente de quantas aprovações humanas o Pull Request tenha recebido.

A organização experimenta uma queda drástica na variabilidade das entregas entre diferentes desenvolvedores. A previsibilidade de release aumenta, pois as regras do jogo estão codificadas no pipeline. Em ambientes regulados (como finanças ou saúde), o maior valor percebido é a auditoria: a empresa passa a ter um histórico criptográfico e imutável que explica perfeitamente por que uma mudança de IA foi aceita, mitigando riscos legais e reduzindo o tempo de análise de incidentes de semanas para minutos.

**Controle Absoluto das Fontes de Variabilidade em ML**

Uma squad de Ciência de Dados sofre cronicamente com a síndrome do "na minha máquina funciona". Um modelo de Gradient Boosting treinado no notebook de um(a) desenvolvedor(a) alcança 92% de F1-Score. No entanto, quando o pipeline tenta recriar o artefato no servidor, a performance cai para 75%. Ao investigar, descobre-se uma teia de variabilidades ocultas: a pessoa desenvolvedora local usou um banco de dados baixado na semana anterior (os dados mudaram), alterou hiperparâmetros manualmente sem versionar (parâmetros perdidos) e usou uma versão da biblioteca scikit-learn diferente da instalada no servidor (ambiente divergente). Sem mitigar isso, a adoção de ferramentas modernas torna-se inútil.

Conectar a teoria de controle de experimentação à execução no workflow, construindo uma esteira no GitHub Actions capaz de amarrar e congelar simultaneamente as quatro fontes de variabilidade em Machine Learning (Código, Dados, Hiperparâmetros e Ambiente Computacional), garantindo que qualquer experimento seja 100% reprodutível a qualquer momento no futuro.

O Arquiteto de MLOps desenha um pipeline estrito no GitHub Actions focando em cada pilar de variabilidade:

- **Código:** O gatilho de push invoca Jobs rigorosos de linting analítico, garantindo que o código de feature engineering seja idêntico em todas as máquinas.
- **Dados:** O pipeline integra nativamente o DVC (Data Version Control). Um step específico do GitHub Actions faz o checkout do código e logo em seguida executa `dvc pull`, garantindo que o Runner baixe do bucket S3 a exata fotografia imutável (o hash) do dataset correspondente àquele commit específico.
- **Parâmetros:** Para evitar testes manuais, utiliza-se a funcionalidade `strategy: matrix` do GitHub Actions. O pipeline dispara 5 Runners paralelos, cada um injetando uma combinação diferente de hiperparâmetros (como taxas de aprendizado) como variáveis de ambiente no script de treino, registrando tudo automaticamente em um servidor de rastreamento (como MLflow).
- **Ambiente:** Antes de qualquer cálculo matemático, o GitHub Actions obrigatoriamente constrói uma imagem Docker. Todo o treinamento é feito dentro de contêineres efêmeros, garantindo que versões do Sistema Operacional, drivers CUDA e bibliotecas Python sejam perfeitamente idênticas àquelas definidas pelo(a) desenvolvedor(a) no arquivo Dockerfile.

O impacto na engenharia é imediato e mensurável. Métricas de processo, como a taxa de falha de integração, despencam quase a zero. O tempo médio de recuperação (MTTR) melhora drasticamente, pois um plano de rollback agora significa simplesmente reverter para o commit anterior, sabendo que os dados e o ambiente computacional também reverterão com segurança. Essa leitura orientada a evidências cria uma fundação sustentável e escalável, diferenciando times amadores de operações de Inteligência Artificial de alta criticidade.

**Contratos técnicos para merge seguro e governança**

Em equipes de Inteligência Artificial em hipercrescimento, é comum que cientistas de dados com diferentes níveis de senioridade e backgrounds (estatística, física, computação) colaborem no mesmo repositório. Sem barreiras claras, códigos são enviados para a branch principal com inconsistências de estilo (dificultando a leitura), testes falhando silenciosamente ou, pior, com dependências vulneráveis e esquemas de dados incompatíveis. Isso gera uma esteira de integração frágil, onde o pipeline frequentemente quebra no momento de gerar a imagem Docker de produção, atrasando entregas críticas e gerando atritos entre as equipes de Dados e de Operações.

A arquitetura precisa estabelecer "contratos técnicos" rigorosos e inegociáveis antes que qualquer código ou artefato de modelo seja mesclado (merged) na base de código principal. O desafio é criar uma barreira de qualidade (um Quality Gate) que garanta conformidade corporativa e segurança da informação, mas sem estrangular a produtividade ou a autonomia de experimentação do cientista de dados.

O time formaliza essa governança transformando políticas escritas em verificações automatizadas no GitHub Actions. Configuram-se Branch Protection Rules no GitHub, exigindo que Status Checks específicos passem antes de habilitar o botão de Merge. No código (via arquivos YAML no diretório `.github/workflows/`), a equipe cria Jobs parciais:

- **Validação Estática e Linting:** Execução de ferramentas como Ruff ou Flake8 para padronizar o código Python.
- **Testes de Unidade:** Acionamento do PyTest rodando dentro de um contêiner efêmero no Runner do GitHub Actions para garantir que as funções de transformação de dados (Feature Engineering) matemáticas retornem os tensores esperados.
- **Contrato de Dados:** Execução de bibliotecas como Great Expectations ou Pandera em uma amostra de dados para garantir que a tipagem e a distribuição esperadas pelo modelo não foram violadas na nova versão do código.
- **Varredura de Segurança (SecOps):** A esteira invoca o Trivy para escanear a imagem base e o Bandit para buscar vulnerabilidades no código Python (como senhas hardcoded). Calibra-se a profundidade: verificações estáticas e leves rodam a cada commit; validações profundas de integração rodam apenas na aprovação final.

O pipeline deixa de ser um mero aglomerado de scripts executados em sequência e passa a atuar como um instrumento implacável e neutro de governança técnica. O retrabalho despenca, pois o(a) desenvolvedor(a) é impedido pelo sistema de integrar um código fora do padrão. A organização elimina a dependência de revisões humanas subjetivas para garantir qualidade básica, criando um histórico confiável e auditável para áreas de Compliance.

**Estratégia de feedback rápido e eficiência computacional**

Treinamentos de redes neurais profundas ou de algoritmos de gradient boosting robustos exigem alto custo computacional e tempo (horas ou até dias). Se o pipeline de CI disparar um treinamento completo a cada Push ou Pull Request (PR) criado pelo cientista de dados, os Runners do GitHub Actions (especialmente se usarem instâncias EC2 com GPUs na AWS) ficarão sobrecarregados. O custo financeiro de nuvem explodirá e a pessoa desenvolvedora terá que esperar horas apenas para descobrir que seu PR falhou por um erro bobo de sintaxe na última linha do script de inferência.

Desenhar uma arquitetura de integração contínua assimétrica. Ela deve fornecer respostas quase imediatas (em poucos minutos) ao(à) desenvolvedor(a) durante a fase de PR para manter o fluxo de trabalho ágil (o chamado Shift-Left Testing), reservando o processamento pesado e dispendioso apenas para os momentos em que a mudança já foi revisada, aprovada e está prestes a impactar o negócio.

A engenharia de MLOps divide os workflows do GitHub Actions em duas categorias distintas baseadas em gatilhos (`on: pull_request` vs `on: push` na main).

- **No Pull Request (Feedback Rápido):** O GitHub Actions executa apenas os testes unitários básicos e realiza um Dummy Training (um treinamento rápido de "sanidade"). Para isso, o pipeline baixa um dataset amostral minúsculo e treina o modelo por apenas 1 ou 2 épocas. O objetivo não é verificar a acurácia do modelo, mas sim provar que o grafo computacional compila, que não há vazamento de memória flagrante e que a função de perda (loss function) consegue ser calculada. O GitHub Actions utiliza o token automático (`GITHUB_TOKEN`) para publicar um comentário diretamente no PR com o resultado do Dummy Training. Além disso, faz uso intenso da funcionalidade `actions/cache` para armazenar dependências do Python (como pacotes do poetry ou pip), economizando minutos preciosos no build da máquina virtual.
- **Na Branch Principal (Validação Profunda):** Apenas após a aprovação por pares e o merge para a main, um segundo workflow do GitHub Actions entra em cena. Este sim faz o checkout do dataset completo (via DVC), aloca as instâncias pesadas com GPU, treina o modelo até a convergência e gera as métricas finais de negócio (F1-Score, RMSE, etc.) para registrar no repositório de modelos (Model Registry).

Essa estratégia equilibra perfeitamente os trade-offs entre velocidade de entrega, custo computacional e robustez. Reduz-se drasticamente o Lead Time de desenvolvimento diário da equipe. Os desenvolvedores recebem feedback sobre seus erros em minutos, enquanto a corporação protege seu orçamento de infraestrutura, garantindo que o treinamento caro só ocorra em códigos validados e maduros.

**Observabilidade e Resposta Sociotécnica a Incidentes**

Após o deploy, os sistemas de IA não falham da mesma forma que sistemas web tradicionais (como um "Erro 500"). Muitas vezes, um modelo continua respondendo às requisições (Status 200 OK), mas suas predições começam a degradar devido a mudanças no comportamento do mundo real (Concept Drift). Sem métricas adequadas e comunicação entre a esteira de CI/CD e as ferramentas de produção, a equipe perde a capacidade de detectar regressões cedo, e as falhas acabam sendo descobertas apenas quando os clientes finais reclamam ou a receita da empresa cai.

Quebrar o silo entre a etapa de entrega (o Deploy feito pelo GitHub Actions) e a etapa de operação continuada (Monitoring). É preciso construir uma ponte bidirecional onde o pipeline informe a produção sobre o que está sendo entregue, e a produção possa sinalizar ao repositório quando um modelo precisar ser aposentado ou retreinado.

O pipeline do GitHub Actions não morre quando o contêiner sobe no cluster Kubernetes. A equipe configura a etapa final do workflow para emitir Webhooks e anotações ativas (Deployment Markers) para plataformas de observabilidade, como Datadog, Grafana ou New Relic. A integração inclui enviar o hash exato do commit e a versão do modelo (ex: v2.4.1) para esses painéis. Assim, se os gráficos operacionais começarem a mostrar anomalias estatísticas, o engenheiro de plantão pode clicar na anotação do gráfico e ser levado diretamente para o Run específico no GitHub Actions, visualizando exatamente qual código gerou aquele modelo problemático e quem aprovou o PR. Além disso, criam-se rotinas de resposta a incidentes (Runbooks automatizados). Por meio da funcionalidade de `workflow_dispatch` do GitHub Actions, a equipe cria botões de emergência (Rollback Workflows) que permitem reverter rapidamente a produção para o artefato seguro anterior com apenas um clique na interface do GitHub.

O tempo médio de recuperação (MTTR) frente a degradações de modelo cai de dias para minutos. Ao tratar a observabilidade como parte indissociável do sistema sociotécnico de entrega, a organização reduz o estresse da equipe, fortalece a confiabilidade do produto e transforma o rigor do MLOps em uma vantagem competitiva real contra a concorrência.

**Segurança da Cadeia de Suprimentos e Compliance Regulatório**

Em uma instituição financeira que desenvolve modelos de Credit Scoring (avaliação de crédito), a equipe de dados prioriza a agilidade. Para acessar o banco de dados e os buckets de nuvem durante o treinamento, cientistas colocam credenciais em texto claro (hardcoded) nos scripts Python ou em arquivos `.env` commitados acidentalmente. Além disso, utilizam bibliotecas de código aberto (`pip install pandas`) sem verificar a integridade dos pacotes, abrindo portas para ataques de Supply Chain (como pacotes maliciosos inseridos no PyPI). Quando o Banco Central exige uma auditoria para entender como um modelo específico tomou uma decisão de negação de crédito há seis meses, a empresa entra em colapso: não há como provar matematicamente qual versão exata dos dados e dependências foi utilizada para gerar aquele artefato, resultando em multas e paralisação das operações.

Blindar o repositório contra vazamentos de credenciais, proteger a cadeia de suprimentos de software inserindo segurança no pipeline (DevSecOps para ML) e garantir a reprodutibilidade forense do modelo para satisfazer reguladores externos, sem que isso exija preenchimento manual de formulários de compliance por parte da equipe de desenvolvedores.

O time de Arquitetura implementa uma reformulação de segurança nativa no GitHub Actions:

- **Governança de Segredos sem Senhas Físicas:** Elimina-se o uso de chaves de API estáticas. O pipeline é reconfigurado para usar o GitHub OIDC (OpenID Connect). O GitHub Actions agora negocia tokens de acesso temporários e de vida curta (short-lived tokens) diretamente com a provedora de nuvem (AWS/GCP/Azure) apenas durante os minutos em que o Job de treinamento está rodando. Para segredos inevitáveis, utilizam-se os GitHub Environments, onde credenciais de produção só podem ser acessadas por branches protegidas e mediante aprovação de um comitê de segurança.
- **Segurança de Supply Chain:** Insere-se o Dependabot e CodeQL no repositório. O workflow no GitHub Actions passa a contar com um Job de varredura que executa ferramentas como pip-audit. Se o modelo tentar ser construído com uma biblioteca de IA que contenha uma vulnerabilidade CVE (Common Vulnerabilities and Exposures) crítica recém-descoberta, a esteira quebra e impede o deploy.
- **Cartório Digital (Reprodutibilidade):** Ao final do treinamento, o pipeline assina criptograficamente o artefato do modelo e a imagem Docker gerada (usando ferramentas como Cosign). O hash de assinatura, juntamente com a versão do dado e o commit, é gravado de forma imutável nos Releases do GitHub.

O risco de exfiltração de dados por vazamento de senhas é virtualmente eliminado. Durante a próxima auditoria regulatória, a equipe não precisa procurar em e-mails ou anotações; basta apresentar o log público do GitHub Actions correspondente àquela data. O pipeline atua como um cartório digital automatizado, onde a conformidade legal deixa de ser um esforço humano e passa a ser um subproduto natural da própria engenharia de software.

**Evolução de Maturidade: Autonomia vs. Governança**

Com o sucesso dos primeiros modelos, a organização escala de uma única squad de Inteligência Artificial para dez times distintos operando simultaneamente. A base de código infla. Inicialmente, o time de infraestrutura tentou criar um único workflow YAML monolítico de 2.000 linhas no GitHub para lidar com tudo, forçando todos os times a seguirem um padrão rígido e engessado. Isso gerou revolta: os cientistas de dados reclamavam que não tinham autonomia para testar novas arquiteturas de rede neural, pois qualquer mudança exigia aprovação da infraestrutura. Por outro lado, quando a infraestrutura afrouxou as regras, times começaram a copiar e colar scripts de CI/CD não otimizados, gerando drift (desvio) de configuração, faturas de nuvem altíssimas devido a instâncias ociosas e incidentes graves de concorrência no banco de produção.

Evoluir a maturidade da operação, saindo do caos de scripts manuais/copiados para uma verdadeira "Plataforma Interna de MLOps" (Internal Developer Platform). O desafio arquitetural é encontrar o ponto de equilíbrio perfeito: maximizar a autonomia criativa dos(as) cientistas de dados em seus ambientes de pesquisa (sandbox), mas impor uma governança corporativa férrea e padronizada no momento de entregar para produção.

A engenharia adota o padrão de Reusable Workflows (Workflows Reutilizáveis) do GitHub Actions.

- **Governança Centralizada:** A equipe de Plataforma cria um repositório isolado e altamente restrito contendo os templates YAML oficiais da empresa para treinamento, validação e deploy de modelos. Esses templates embutem todas as regras de segurança, limites de custo de GPU e contratos de dados discutidos anteriormente.
- **Autonomia na Ponta:** Os repositórios dos(as) cientistas de dados ficam incrivelmente enxutos. O arquivo de pipeline deles não possui lógicas complexas; ele apenas chama o template oficial apontando para os seus próprios dados (ex.: `uses: empresa/plataforma-mlops/.github/workflows/standard-train.yml@v2`).

**O QUE VOCÊ VIU NESTA AULA?**

Nesta aula, você diferenciou DevOps tradicional de MLOps e entendeu por que modelos exigem controle adicional de código, dados, parâmetros e ambiente. Também conheceu a anatomia do GitHub Actions: workflows, jobs, steps, runners e gatilhos. Na prática, criou a base de um pipeline de integração contínua capaz de validar mudanças automaticamente e reduzir dependência de conferência manual.

**PALAVRAS-CHAVE:** MLOps. GitHub Actions. CI/CD. Pipeline. Automação.

### Código e comandos
Nenhum bloco de código nesta aula. (O material menciona inline os seguintes elementos técnicos: diretório `.github/workflows`; comando `dvc pull`; diretiva `strategy: matrix`; gatilhos `on: pull_request` e `on: push`; token `GITHUB_TOKEN`; ação `actions/cache`; funcionalidade `workflow_dispatch`; comando `pip install pandas`; referência de reusable workflow `uses: empresa/plataforma-mlops/.github/workflows/standard-train.yml@v2`.)

### Ferramentas / serviços citados
GitHub Actions, DVC (Data Version Control), bucket S3, MLflow, Docker/Dockerfile, CUDA, scikit-learn, Ruff, Flake8, PyTest, Great Expectations, Pandera, Trivy, Bandit, EC2 (AWS), Kubernetes, Datadog, Grafana, New Relic, GitHub OIDC (OpenID Connect), GitHub Environments, Dependabot, CodeQL, pip-audit, Cosign, ADRs (Architecture Decision Records).

### Aplicabilidade ao Tech Challenge Fase 3
- Fundamenta o requisito de CI/CD com GitHub Actions (15%): estrutura `.github/workflows`, gatilhos `on: push` / `on: pull_request` e Jobs/Steps aplicáveis ao classificador NLP de laudos médicos.
- Justifica os Quality Gates (linting com Ruff/Flake8, PyTest em contêiner) antes do merge, base para os testes automatizados exigidos.
- Introduz Dockerização e reprodutibilidade de ambiente, pré-requisito para a containerização e entrega contínua do projeto.

### REFERÊNCIAS (Aula 1)
- FACURE, Matheus. *Causal inference for the brave and true*. [S. l.], [s. d.]. Disponível em: https://matheusfacure.github.io/python-causality-handbook/. Acesso em: 29 mar. 2026.
- CUNNINGHAM, Scott. *Causal inference: the mixtape*. New Haven: Yale University Press, 2021. Disponível em: https://mixtape.scunning.com/. Acesso em: 15 maio 2026.
- PETERS, Jonas; JANZING, Dominik; SCHÖLKOPF, Bernhard. *Elements of causal inference: foundations and learning algorithms*. Cambridge, MA: MIT Press, 2017. Disponível em: https://mitpress.mit.edu/9780262037310/elements-of-causal-inference/. Acesso em: 15 maio 2026.

> [NOTA — não é conteúdo FIAP]: as três referências da Aula 1 tratam de inferência causal e não têm relação direta com o tema CI/CD/GitHub Actions da aula; foram transcritas exatamente como constam no PDF.

---

## Aula 2 — Integração Contínua para Data Science com GitHub Actions
**Arquivo fonte:** `9MLET - Fase 3 - Integração com CICD - Aula 2.pdf` (16 páginas)
**Título na ementa:** Integração Contínua para Data Science com GitHub Actions

### Conceitos-chave
- "Higiene" de código de Ciência de Dados: linting, formatação e limpeza de Jupyter Notebooks.
- Redefinição de "integração" em ML (código + dados + ambiente).
- Testes semânticos de dados: Property-Based Testing e Shape Testing.
- Ambientes efêmeros, deriva de dependências e travas determinísticas (lockfiles).
- Dummy Training (treinamento de sanidade) como estratégia Fail-Fast.
- Rastreamento de experimentos e feedback de CI diretamente no Pull Request.

### Conteúdo

**O QUE VEM POR AÍ?**

Os Jupyter Notebooks são maravilhosos para a descoberta e experimentação matemática, mas são o pesadelo da Engenharia de Software. Variáveis globais soltas, metadados de execução ocultos e dependências flexíveis transformam o merge de código em uma dor de cabeça constante.

Nesta aula, vamos focar na "higiene" do código de Ciência de Dados. Você entenderá como impor contratos de qualidade rigorosos através da Integração Contínua, garantindo que nenhum código ruim polua a base principal da sua equipe.

**HANDS ON**

Nesta etapa prática, vamos transformar a nossa esteira de CI em um inspetor implacável. Configuraremos ferramentas de análise estática (Linters) como Black e Ruff diretamente no GitHub Actions. Além disso, você aprenderá a domar os Jupyter Notebooks utilizando nbstripout para limpar metadados e nbqa para testar células. Por fim, vamos travar nossas dependências usando gerenciadores determinísticos como o Poetry.

**SAIBA MAIS**

**Introdução Arquitetural: A Redefinição de "Integração" em ML**

No ecossistema de engenharia de software tradicional, a Integração Contínua (CI) é frequentemente resumida à capacidade de mesclar o código de múltiplos desenvolvedores em um repositório central, garantindo que o software compile e passe em testes lógicos. No entanto, quando aplicamos esse conceito à Ciência de Dados, a definição de "integração" sofre uma mutação arquitetural severa. Em Machine Learning, o código é apenas uma fração do sistema. O comportamento final do artefato depende indissociavelmente dos dados que fluem por ele e do ambiente matemático onde ele é executado.

Portanto, um pipeline de CI para ML no GitHub Actions não deve apenas validar a sintaxe da linguagem Python; ele deve atuar como um portal de governança científica. Ele precisa validar a higienização dos Jupyter Notebooks, garantir a imutabilidade do ambiente via conteinerização, testar a lógica de tensores e transformações matemáticas, e executar treinamentos de sanidade (dummy trainings) para provar que a rede neural ou o algoritmo consegue convergir. Abaixo, analisamos os pilares da Integração Contínua em MLOps através de estudos de caso rigorosos utilizando o framework STAR.

**O Caos dos Notebooks e a Padronização Sociotécnica de Código**

Em uma equipe de Inteligência Artificial composta majoritariamente por estatísticos e matemáticos, a ferramenta principal de trabalho é o Jupyter Notebook (`.ipynb`). Os(as) cientistas realizam experimentações fantásticas, mas o controle de versão desses arquivos no Git é um pesadelo. Os arquivos JSON por trás dos notebooks contêm metadados inúteis, contagens de execução e saídas de gráficos em base64 que geram conflitos de merge impossíveis de resolver. Pior ainda, códigos vão para a branch principal (main) cheios de importações não utilizadas, variáveis globais, caminhos de arquivos locais (`C:\Users\Cientista\dados.csv`) e ausência total de tipagem estática. Quando a equipe de Engenharia de Machine Learning tenta pegar esse notebook para colocar em produção, o retrabalho é colossal, demorando semanas para refatorar o experimento em scripts Python (`.py`) executáveis e testáveis.

Construir uma barreira arquitetural na fase inicial da Integração Contínua que atue como um "tradutor" e "higienizador" entre o mundo exploratório da Ciência de Dados e o mundo determinístico da Engenharia de Software. O pipeline de CI deve educar a equipe organicamente, rejeitando más práticas de código antes que elas poluam a base central, sem proibir o uso de notebooks na fase de pesquisa.

O Arquiteto de MLOps implementa um workflow estrito no GitHub Actions ativado pelo gatilho `on: [pull_request]`. Este pipeline de Integração Contínua não treina modelos; ele atua puramente na inspeção do código (Linting e Static Analysis).

- **Limpeza de Metadados:** O primeiro Job utiliza ferramentas como nbstripout ou jq para limpar automaticamente todas as saídas e metadados dos notebooks commitados, garantindo que apenas o código-fonte seja avaliado no diff do Pull Request.
- **Linting Adaptado:** Utilizando a ferramenta nbqa, o GitHub Actions consegue rodar linters tradicionais de Python diretamente dentro das células do notebook. O pipeline executa o Black para formatar a sintaxe (tamanho de linhas, aspas duplas) e o Ruff ou Flake8 para encontrar violações lógicas (variáveis declaradas e não usadas, importações redundantes).
- **Análise Estática Modular:** O CI verifica a presença de tipagem estática (Type Hints) nas funções extraídas, utilizando o mypy. Se uma função de transformação de dados não especificar que recebe um `pd.DataFrame` e retorna um `np.ndarray`, o GitHub Actions falha e impede o merge.
- **Feedback Automatizado:** Em vez de apenas falhar silenciosamente, o pipeline utiliza a API do GitHub (via `actions/github-script`) para comentar exatamente na linha de código do Pull Request onde a violação ocorreu, sugerindo a correção (ex: "Substitua o caminho local por uma variável de ambiente").

O atrito entre os Cientistas de Dados e os Engenheiros de ML desaparece. A revisão de código (Code Review) deixa de ser uma discussão chata sobre espaços e vírgulas (já que o robô de CI resolve isso) e passa a focar na arquitetura matemática do modelo. A base de código na main torna-se imaculada, modular e pronta para ser empacotada, reduzindo o tempo de refatoração para produção de semanas para zero.

**Testes Automatizados para Tensores e Engenharia de Features**

Uma organização possui um modelo de Churn (evasão de clientes) que foi exaustivamente revisado quanto à qualidade do código (PEP-8). No entanto, após uma atualização no pipeline de dados aprovada no CI, o modelo em produção começou a gerar predições absurdas. A investigação post-mortem revelou que o código estava sintaticamente perfeito, mas a lógica matemática da Engenharia de Features falhou: uma nova função de agregação no Pandas estava gerando silenciosamente valores NaN (Not a Number) para clientes sem histórico de compras.

Como o código não quebrou (o Python lidou bem com o NaN), o erro propagou-se para os tensores da rede neural, destruindo os gradientes durante o retreinamento. Testes unitários de software tradicionais (verificar se 1+1=2) falharam em capturar a semântica dos dados.

Elevar a Integração Contínua de uma validação sintática para uma validação semântica e matemática. É necessário garantir, durante o CI, que todas as funções de processamento de dados (o Feature Store local) operem corretamente diante de casos extremos (edge cases) de dados estatísticos, como distribuições distorcidas, vetores esparsos e matrizes de dimensionalidade inesperada.

A esteira de CI no GitHub Actions é expandida para incluir um estágio de "Testes Semânticos de Dados" (Data-Aware Testing), rodando em contêineres independentes.

- **Property-Based Testing:** Em vez de escrever testes unitários com valores fixos, a equipe integra a biblioteca Hypothesis ao PyTest no GitHub Actions. Durante a CI, o framework gera centenas de DataFrames sintéticos aleatórios (com strings vazias, datas inválidas, números negativos extremos) e joga contra as funções de feature engineering. O objetivo é tentar "quebrar" a matemática da função e garantir que ela trate exceções graciosamente.
- **Testes de Dimensionalidade (Shape Testing):** Para códigos de Deep Learning (PyTorch/TensorFlow), o pipeline inclui verificações de asserção rigorosas sobre a forma (shape) dos tensores. O Job verifica se, após um reshape ou pooling, o tensor de saída mantém a dimensionalidade exigida pela próxima camada da rede (ex: `assert tensor.shape == (batch_size, channels, height, width)`).
- **Contratos de Dados Simulados:** Utilizando Great Expectations em um subconjunto minúsculo de dados mocados (mock data), o CI valida se as colunas resultantes não violam regras de negócio (ex: "a idade normalizada deve estar estritamente entre 0 e 1").

O GitHub Actions passa a atuar como um escudo matemático. Erros sutis de agregação de dados ou broadcasting indesejado em arrays Numpy são capturados em minutos, na máquina virtual gratuita da esteira, antes de gastarem milhares de dólares em instâncias de GPU na nuvem tentando treinar em cima de dados corrompidos. A confiança da equipe na resiliência do pipeline de dados atinge níveis inéditos.

**Ambientes Efêmeros e a Maldição da Deriva de Dependências**

O time de Inteligência Artificial sofre com a "deriva de ambiente" (environment drift). O Cientista de Dados constrói e testa um modelo localmente usando um Mac M1 com uma versão específica do scikit-learn e do xgboost. Ele faz o Push do código. A esteira de Integração Contínua pega esse código, instala as dependências em uma máquina Ubuntu genérica usando um `pip install` solto, e os testes passam. Semanas depois, quando o modelo é promovido para a esteira de CD (Entrega Contínua) e implantado no cluster Kubernetes, as predições diferem sutilmente dos testes locais. O motivo? O `pip install` solto no pipeline de CI baixou versões mais recentes de bibliotecas matemáticas subjacentes (como C++ compilado no backend do Numpy) que possuíam pequenas otimizações de ponto flutuante, alterando o comportamento do modelo.

A Integração Contínua deve erradicar a variabilidade computacional. O ambiente em que o(a) desenvolvedor(a) programa, o ambiente em que a CI testa o código e o ambiente de produção devem ser clones binários exatos. A missão é forçar a adoção de imagens imutáveis sem tornar a esteira de CI dolorosamente lenta (visto que construir imagens Docker com CUDA e PyTorch pode levar 40 minutos).

O GitHub Actions é configurado para orquestrar um fluxo rigoroso de conteinerização na etapa de Integração:

- **Trava de Dependências (Locking):** O CI primeiramente rejeita qualquer Pull Request que dependa apenas de um arquivo `requirements.txt` flexível (ex: `pandas>=1.0`). O pipeline exige o uso de gerenciadores de pacotes determinísticos como Poetry ou Pipenv, que geram arquivos `.lock`, garantindo as exatas hashes de todas as subdependências.
- **Build Imutável na CI:** Em todo Pull Request, o GitHub Actions executa o comando `docker build`. O código e as dependências travadas são empacotadas em uma imagem Docker isolada.
- **Estratégia de Caching:** Para resolver a lentidão, o Arquiteto de MLOps implementa a ação oficial `docker/build-push-action`. O GitHub Actions é instruído a usar o GitHub Container Registry (GHCR) como um sistema de cache de camadas externas (`cache-from` e `cache-to`). Se o(a) Cientista mudou apenas o arquivo `.py` de lógica, a CI reaproveita a camada pesada do PyTorch em segundos, reconstruindo apenas a camada superior de código.
- **Testes In-Container:** A virada de chave acontece aqui: a esteira de CI não roda os testes (como o pytest mencionado na seção anterior) no Runner do GitHub. O workflow sobe o contêiner recém-criado e roda os testes dentro dele.

O fenômeno "na minha máquina funciona" é extinto. O modelo aprovado na Integração Contínua é certificado de que rodou sobre uma fundação binária e matemática exata. Como bônus, a imagem Docker gerada e testada na CI torna-se o próprio artefato final de publicação, eliminando etapas de reconstrução no pipeline de Deploy, garantindo rastreabilidade forense entre o código testado e o artefato executado em produção.

**O treinamento de sanidade ("Dummy Training") como Fail-Fast**

Treinar modelos fundacionais ou grandes redes neurais convolucionais (CNNs) custa dezenas, às vezes centenas de dólares por execução na nuvem, levando muitas horas. Para economizar, uma equipe desativou os treinamentos na esteira de CI. Os(as) desenvolvedores(as) passaram a codificar as arquiteturas das redes e, após o código ser mesclado na main, um orquestrador agendado disparava o treinamento noturno nas GPUs pesadas. Frequentemente, a equipe chegava na manhã seguinte apenas para descobrir que o treinamento falhou no minuto 2 porque havia um erro de dimensão na última camada da rede neural, ou que rodou a noite toda, mas os pesos não atualizaram (vanishing gradients) devido a um erro na escolha do otimizador (ex.: Adam vs SGD). Dinheiro de nuvem e horas de projeto foram totalmente desperdiçados.

Inserir uma etapa de validação de Deep Learning dentro da Integração Contínua que garanta que a arquitetura do modelo compila, que o fluxo de tensores (forward pass e backward pass) ocorre sem erros e que a rede tem a capacidade matemática de aprender, tudo isso gastando apenas centavos e executando em menos de 5 minutos, antes do código ir para a main.

Implementa-se o conceito de Dummy Training (Treinamento de Sanidade) nos workflows de CI do GitHub Actions acionados por PRs.

- **Geração de Dados Sintéticos:** A esteira de CI não faz o download do terabyte de dados de produção. Um Job executa um script que gera um micro-dataset (ex.: 2 lotes/batches de dados) que imita exatamente a estrutura, os tipos e os esquemas do dado real.
- **Execução Acelerada no Runner (CPU):** O treinamento completo de produção pode exigir um cluster multi-GPU, mas a validação estrutural do código não. O GitHub Actions utiliza seus próprios Runners gratuitos (focados em CPU) para iniciar a rotina de treinamento da rede neural com este micro-dataset.
- **Validação de Grafos e Gradientes:** O script de Dummy Training é configurado para treinar por exatamente 2 a 5 Epochs (épocas) e sobreescrever temporariamente os hiperparâmetros (como forçar um `batch_size` minúsculo). O teste de CI afirma duas coisas matematicamente em código:
  - (A) O script não lança exceções de "Out of Memory" ou de "Shape Mismatch" durante o fluxo.
  - (B) A Loss Function (função de perda) do modelo no Epoch 5 é estritamente menor que a do Epoch 1. Isso prova que o grafo computacional está conectado corretamente, que os gradientes estão fluindo de volta pela rede e que o modelo está, de fato, "aprendendo" (ainda que sobrepostos/overfitting em dados minúsculos).

Implementação clássica de Shift-Left Testing (trazer os testes para o início do ciclo). A pessoa desenvolvedora que submeteu o Pull Request descobre que quebrou a arquitetura do modelo em apenas 3 minutos, lendo o log do GitHub Actions. A empresa protege seu orçamento de infraestrutura de nuvem, garantindo que execuções de GPUs noturnas só ocorram em códigos que foram provados estruturalmente viáveis pela Integração Contínua.

**Rastreamento de Experimentos e Feedback de CI no PR**

Em muitos processos tradicionais, a discussão de um Pull Request no GitHub baseia-se exclusivamente na leitura humana do código. Em Machine Learning, no entanto, um código mais "bonito" ou refatorado não significa um modelo com maior acurácia. Uma cientista alterou a função de ativação de uma rede de ReLU para LeakyReLU. No Pull Request, os revisores concordam que o código está correto, mas ninguém sabe responder a pergunta de negócio: "Essa mudança melhora o F1-Score do modelo?". Para descobrir, os revisores precisam clonar a branch localmente, baixar os dados, rodar treinamentos locais comparativos e anotar em planilhas, atrasando a aprovação do código em vários dias.

O pipeline de CI deve automatizar o levantamento das métricas estatísticas comparativas entre o código que está querendo entrar (a Feature Branch) e o código que já está estabilizado (a Main Branch). O objetivo é fornecer aos revisores de código métricas baseadas em evidências diretamente na interface nativa do Pull Request no GitHub.

O pipeline de GitHub Actions integra orquestração de nuvem com ferramentas de rastreamento de experimentos de ML.

- **Short Training Comparativo:** Em modelos mais leves (como Árvores de Decisão ou Regressões baseadas em Scikit-Learn), o CI aciona não apenas um dummy train, mas um Short Train real em uma amostra estatisticamente válida dos dados reais. O CI executa esse código na branch do PR e, silenciosamente, re-executa na versão atual da main.
- **Integração com MLflow / Weights & Biases:** Durante o treinamento acionado pela CI, o código emite os resultados estatísticos (Acurácia, Precisão, Recall, Curva ROC) para um servidor centralizado de MLflow, marcando as execuções com a tag contendo o SHA (hash) do commit do GitHub.
- **Bot de CI no Pull Request:** O workflow no GitHub Actions atinge o seu clímax. Utilizando a funcionalidade github-script ou bibliotecas como cml (Continuous Machine Learning da Iterative), o pipeline gera gráficos de comparação visual (ex: uma matriz de confusão) e uma tabela em linguagem Markdown com a diferença exata nas métricas. O GitHub Actions então atua como um usuário bot e posta esse relatório como um comentário no Pull Request.

O processo de Code Review é drasticamente evoluído. Os revisores humanos abrem o Pull Request e encontram, logo abaixo das mudanças de código, um relatório do bot de Integração Contínua afirmando: "Este PR reduz a complexidade ciclomática do código em 10%, no entanto, as métricas do MLflow indicam que o Recall na classe minoritária caiu em 4.5% no dataset de validação." As decisões de mesclagem (Merge) deixam de ser baseadas em opiniões e passam a ser puramente direcionadas por evidências estáticas (código) e evidências dinâmicas (experimento), reduzindo o atrito e garantindo uma governança de produto excepcional.

**O QUE VOCÊ VIU NESTA AULA?**

Nesta aula, compreendemos o custo da dívida técnica gerada por ambientes exploratórios. Aprendemos a automatizar a análise estática e a formatação de código Python voltado para IA diretamente no pipeline. Também resolvemos o caos do versionamento de notebooks, limpando suas execuções antes do merge. Por fim, entendemos a importância de travar dependências subjacentes de matemática utilizando Lockfiles em vez de arquivos `.txt` simples, combinando tudo isso com o uso de cache no GitHub Actions para acelerar a esteira.

**PALAVRAS-CHAVE:** Linting. Jupyter Notebooks. Poetry. Cache. Qualidade de Código.

### Código e comandos
Nenhum bloco de código nesta aula. (Elementos técnicos citados inline: gatilho `on: [pull_request]`; extensão `.ipynb`; caminho local `C:\Users\Cientista\dados.csv`; tipos `pd.DataFrame` / `np.ndarray`; ação `actions/github-script`; asserção `assert tensor.shape == (batch_size, channels, height, width)`; arquivo `requirements.txt` com `pandas>=1.0`; comando `docker build`; ação `docker/build-push-action` com `cache-from` / `cache-to`; parâmetro `batch_size`.)

### Ferramentas / serviços citados
Jupyter Notebook, Git, Black, Ruff, Flake8, nbstripout, jq, nbqa, mypy, `actions/github-script`, Hypothesis, PyTest, PyTorch, TensorFlow, Great Expectations, NumPy, scikit-learn, xgboost, Poetry, Pipenv, Docker, CUDA, `docker/build-push-action`, GitHub Container Registry (GHCR), MLflow, Weights & Biases, cml (Continuous Machine Learning da Iterative).

### Aplicabilidade ao Tech Challenge Fase 3
- Aplica linting (Black/Ruff/Flake8) e tipagem (mypy) na CI do classificador de laudos — reforça a nota de CI/CD (15%) e a qualidade de código.
- Property-Based Testing (Hypothesis + PyTest) e testes de shape endereçam diretamente os testes automatizados exigidos para o pipeline NLP.
- Trava de dependências com lockfiles + build Docker na CI dá base para containerização reprodutível do serviço de inferência.

### REFERÊNCIAS (Aula 2)
- GITHUB. *Poetry Python*. Disponível em: https://github.com/python-poetry/poetry. Acesso em: 22 fev. 2026.
- HEINZ, Martin. *Setting up Python CI/CD*. 2020. Disponível em: https://martinheinz.dev/blog/50. Acesso em: 6 abr. 2026.
- NBQA. *nbQA documentation*. Disponível em: https://nbqa.readthedocs.io/en/latest/. Acesso em: 6 abr. 2026.

---

## Aula 3 — Pipeline CI/CD com GitHub Actions (Projeto de ML)
**Arquivo fonte:** `9MLET - Fase 3 - Integração com CICD - Aula 3.pdf` (14 páginas)
**Título na ementa:** Pipeline CI/CD com GitHub Actions (Projeto de ML)

### Conceitos-chave
- Modularização de pipelines (fim do YAML monolítico) e passagem de estado via artifacts.
- Diretivas `workflow_call` e `workflow_run`; ações `upload-artifact` / `download-artifact`.
- Model Registry (MLflow) como Fonte Única da Verdade.
- Estratégias de rollout: Shadow Mode e Canary Deployments.
- Continuous Training (CT) e fechamento do ciclo (Closed-Loop) via `repository_dispatch`.
- Governança / SecOps e fim das senhas em pipelines (Zero-Trust / OIDC).

### Conteúdo

**O QUE VEM POR AÍ?**

Escrever todo o ciclo de vida de um modelo em um único arquivo de automação gigantesco é uma receita para o desastre. Se a etapa final de deploy falhar, você não deve ter que gastar horas retreinando o modelo desde o início.

Nesta aula, vamos elevar a maturidade do nosso pipeline introduzindo a modularização. Você entenderá como quebrar responsabilidades, criar o conceito de "Treinamento de Sanidade" e interligar a nossa esteira de CI com um sistema vital para empresas baseadas em IA: o Registro de Modelos (Model Registry).

**HANDS ON**

Nesta etapa prática, vamos refatorar nosso pipeline monolítico. Você criará múltiplos Jobs e usará as Actions upload-artifact e download-artifact para passar dados processados entre as máquinas virtuais. Em seguida, criaremos um Dummy Training (treino rápido). Para finalizar, escreveremos um script que conecta o GitHub Actions ao MLflow, gravando o hash do modelo e suas métricas logo após o treinamento.

**SAIBA MAIS**

Nos encontros anteriores, consolidamos o entendimento de que a Integração Contínua (CI) em Machine Learning atua como um escudo matemático e sintático, garantindo a reprodutibilidade dos dados e a validade do grafo computacional. Agora, avançamos para a fronteira final da engenharia de IA corporativa: o Continuous Delivery e Continuous Deployment (CD). Em projetos de software tradicionais, o CD trata de pegar um binário compilado e colocá-lo em um servidor. Em um projeto de Machine Learning, o CD é o processo sociotécnico de transicionar um artefato probabilístico e não-determinístico (o modelo) do ambiente de pesquisa para o ambiente de consumo (inferência), garantindo que ele responda de forma escalável, segura e monitorada. O GitHub Actions, neste contexto, deixa de ser apenas um executor de testes e assume o papel de um maestro de infraestrutura multiplataforma.

**Modularização e Passagem de Estado no Pipeline Ponta a Ponta**

Em cenários iniciais de adoção de MLOps, é frequente encontrarmos equipes que tentam orquestrar todo o ciclo de vida do modelo — extração de dados, limpeza, treinamento, avaliação e implantação — em um único arquivo YAML gigantesco e monolítico no GitHub Actions. Essa abordagem inicial rapidamente se torna insustentável. Quando o tempo de execução do workflow ultrapassa horas, uma simples falha de rede na etapa final de implantação obriga a equipe a re-executar todo o pipeline desde o início, desperdiçando tempo e milhares de dólares em instâncias de GPU. Além disso, a manutenção de um arquivo monolítico gera conflitos constantes entre o Engenheiro de Dados (que mexe na extração), o Cientista de Dados (que altera o treino) e o Engenheiro de MLOps (que configura o deploy).

O desafio arquitetural primordial é quebrar esse monólito em componentes independentes, mas perfeitamente sincronizados. A esteira precisa ser capaz de passar o "estado" e os "artefatos" de uma etapa para a outra sem perder o contexto criptográfico (o hash que garante a linhagem do modelo), permitindo que processos paralelos ocorram de forma eficiente e que falhas em etapas finais não exijam o retreinamento do algoritmo.

Para resolver essa complexidade estrutural, a engenharia de plataforma adota uma arquitetura de Workflows Modulares utilizando as diretivas `workflow_call` e `workflow_run` do GitHub Actions, acopladas a uma rigorosa gestão de artefatos. O processo é fatiado em três pipelines distintos. O primeiro é focado puramente na preparação de dados e feature engineering; ao finalizar, ele utiliza a ação `actions/upload-artifact` para salvar o dataset processado e assinado no armazenamento temporário do GitHub. O segundo pipeline (Treinamento) é engatilhado automaticamente pelo sucesso do primeiro. Ele baixa esse artefato (`actions/download-artifact`), treina o modelo, gera os binários (arquivos `.pkl`, `.onnx` ou `.pt`) e os carrega para um Registro de Modelos externo. O terceiro pipeline (Deploy) aguarda aprovações de ambiente (environment protection rules) antes de empacotar o modelo em um contêiner e enviá-lo para o cluster.

O impacto dessa reestruturação reflete-se imediatamente na resiliência e na agilidade da equipe. A falha no momento do deploy no Kubernetes agora significa apenas re-executar o terceiro estágio em questão de segundos, reaproveitando o modelo já treinado. A organização ganha um pipeline desacoplado onde as responsabilidades são claras: problemas de dados quebram a etapa 1, degradações matemáticas quebram a etapa 2, e falhas de infraestrutura quebram a etapa 3. Essa separação de interesses (Separation of Concerns) é a base para escalar projetos de IA em nível corporativo.

**Orquestração do Registro de Modelos (Model Registry) como Fonte da Verdade**

A missão crítica nesta etapa do pipeline é fechar o abismo entre o repositório de código e o repositório de artefatos de IA. É necessário estabelecer uma Fonte Única da Verdade (Single Source of Truth) para os modelos preditivos, garantindo que a implantação só ocorra se o binário do modelo puder ser rastreado, sem margem para dúvidas, de volta ao commit exato que o originou, emparelhado aos dados específicos utilizados naquela rodada computacional.

**Estratégias Avançadas de Rollout: Shadow Mode e Canary Deployments**

Uma das falhas mais catastróficas em ciclos de vida de Inteligência Artificial ocorre durante a substituição abrupta (Big Bang Deployment) do modelo atual pelo novo. Em uma operação de concessão de crédito, por exemplo, o modelo antigo (V1) pode ser substituído da noite para o dia pelo modelo novo (V2) porque o V2 obteve resultados marginalmente superiores no dataset de teste estático durante a esteira de CI. No entanto, o mundo real apresenta dados não mapeados. O V2, devido a um erro na engenharia de features não detectado, começa a negar crédito indiscriminadamente para um segmento populacional específico. Como a troca foi total, o impacto na receita da empresa é imediato, severo e a reputação da marca sofre danos irreparáveis antes que a equipe consiga identificar o problema e realizar um rollback manual.

O imperativo técnico, portanto, é proteger o ambiente produtivo contra a arrogância estatística. O pipeline de CD precisa suportar estratégias de implantação progressivas e protetoras. O objetivo é validar o comportamento dinâmico do novo algoritmo frente aos dados reais do mundo (live data), medindo sua latência de inferência, consumo de memória e distribuição de predições, tudo isso limitando ou zerando completamente o risco financeiro caso o modelo apresente alucinações matemáticas ou vieses inesperados.

Através do GitHub Actions, a equipe operacionaliza integrações diretas com orquestradores como Kubernetes (utilizando ferramentas como ArgoCD ou Istio) para implementar políticas de Shadow Deployment e Canary Release. Na estratégia Shadow, o GitHub Actions implanta o V2 ao lado do V1 de forma invisível para o usuário final. O gateway da API duplica as requisições recebidas: o usuário recebe a resposta do modelo V1 (seguro), mas a requisição também é processada pelo V2 no fundo. As predições do V2 são salvas no banco de dados para análise de divergência, mas não afetam o negócio. Se o modelo for aprovado nessa fase fantasma, o GitHub Actions aciona a fase Canary. O pipeline atualiza as regras de roteamento da infraestrutura para que apenas 5% do tráfego real de usuários seja direcionado para o novo modelo V2. O workflow então entra em estado de pausa ou monitoramento, analisando as métricas de erro. Se o V2 se mantiver estável, o GitHub Actions aumenta a carga progressivamente (20%, 50%, 100%).

O resultado é uma operação técnica dotada de imensa segurança psicológica e resiliência de negócios. A organização passa a testar seus algoritmos em produção de forma ética e controlada. Caso o modelo V2 apresente falhas nas predições ou lentidão no processamento dos 5% do tráfego inicial, as próprias regras de telemetria acionam um Webhook de volta para o GitHub Actions, que engatilha imediatamente o rollback automatizado para o V1. O incidente torna-se um mero evento de log nos sistemas de observabilidade, e a empresa não sofre interrupção de serviço nem perdas financeiras significativas, provando o valor incomensurável de um pipeline de CD inteligente.

**O Fechamento do Ciclo: Monitoramento Contínuo e Retreinamento Automatizado (CT)**

Modelos de Inteligência Artificial, diferentemente de códigos tradicionais, possuem um "prazo de validade" invisível. O código de ordenação de um banco de dados funcionará da mesma forma hoje e daqui a dez anos; no entanto, um modelo de IA treinado para detectar fraudes em cartões de crédito começará a falhar assim que os fraudadores mudarem seu comportamento (Concept Drift) ou quando o perfil macroeconômico dos clientes mudar (Data Drift). Em operações imaturas, a equipe só descobre essa degradação passivamente, geralmente semanas após o início da queda de performance, através de reclamações do atendimento ao cliente ou por relatórios financeiros mensais de perdas. O processo de retreinamento é então iniciado como um combate a incêndios, correndo contra o tempo, de forma manual e altamente estressante para os cientistas de dados.

O ápice da maturidade arquitetural (MLOps Nível 2) exige o fechamento completo do ciclo de vida (Closed-Loop System). O pipeline de CI/CD construído no GitHub Actions não pode ser uma via de mão única que termina no momento em que o modelo é publicado. Ele deve se transformar em um sistema reativo. A meta é criar um fluxo de Continuous Training (CT), onde a degradação matemática do modelo em produção sirva como o gatilho automático para que toda a esteira de construção — da extração de novos dados à implantação do novo modelo — seja executada sem intervenção humana prioritária, atuando como um sistema imunológico corporativo.

A implementação dessa estratégia começa fora do GitHub, nas ferramentas de observabilidade e monitoramento de dados em produção (como Datadog, Evidently AI ou Arize). Essas ferramentas analisam a distribuição estatística das predições diárias. Quando uma anomalia estatística severa é detectada (por exemplo, a distância de Wasserstein entre os dados de treinamento originais e os dados de inferência atuais ultrapassa um limite crítico), o sistema de monitoramento dispara um Webhook autenticado apontando para a API do GitHub. No repositório, o GitHub Actions possui um workflow aguardando eventos do tipo `repository_dispatch` (gatilhos externos). Ao receber esse sinal de degradação, o pipeline de Retreinamento Contínuo acorda. Ele puxa uma janela móvel com os dados mais recentes dos últimos 30 dias do banco de produção, executa as feature engineerings, roda o treinamento completo, avalia se o novo modelo supera o degradado e, caso positivo, executa a implantação Canary.

A organização atinge o nível de arte na gestão de Inteligência Artificial. A manutenção de sistemas preditivos deixa de ser uma tarefa baseada em reações emocionais e crises pontuais. A equipe de Ciência de Dados não precisa mais interromper seu trabalho criativo em novos projetos para ficar atualizando modelos antigos manualmente. A infraestrutura baseada no GitHub Actions se torna autossuficiente para gerenciar a saúde de modelos triviais e de médio impacto, escalonando o alerta para intervenção humana apenas se o workflow de retreinamento automatizado falhar em encontrar uma nova convergência matemática ou não atingir o limite mínimo de acurácia aceitável imposto pelo negócio.

**Governança, Compliance SecOps e o Fim das Senhas em Pipelines de ML**

Durante a fase final de implantação, a esteira de CI/CD assume o controle e a responsabilidade sobre recursos produtivos de altíssimo risco. Para que o GitHub Actions consiga atualizar o serviço gerenciado (como empurrar uma nova imagem de contêiner para o Amazon ECR e atualizar os Pods no EKS), ele tradicionalmente exigiria chaves secretas de acesso de longa duração (Long-Lived Access Keys) com privilégios de administrador. O armazenamento dessas chaves definitivas nos Secrets do repositório, por mais seguro que pareça, gera um passivo de segurança massivo. Se o repositório for comprometido, ou se um engenheiro mal-intencionado imprimir (print) propositalmente as variáveis de ambiente nos logs da esteira, os invasores ganham as chaves do reino produtivo. Além disso, a rotação (troca) manual dessas senhas é frequentemente negligenciada.

A meta absoluta de DevSecOps para Machine Learning é eliminar permanentemente a existência física de credenciais fixas, senhas e chaves de API estáticas usadas pelos sistemas de CI/CD. A arquitetura deve evoluir para um paradigma de confiança zero (Zero).

> [NOTA — não é conteúdo FIAP]: o texto da última seção é interrompido em "(Zero)" ao final da pág. 10/14, sem conclusão no PDF; a pág. 11 já apresenta a seção "O QUE VOCÊ VIU NESTA AULA?".

**O QUE VOCÊ VIU NESTA AULA?**

Nesta aula, desmistificamos a gestão de estado dentro de pipelines efêmeros. Compreendemos como trafegar artefatos gerados entre Jobs independentes. Adotamos o conceito de Fail-Fast criando treinamentos curtos de validação. Também exploramos o papel do Registro de Modelos, utilizando o MLflow para fechar a lacuna de rastreabilidade entre o código versionado no GitHub e o artefato preditivo gerado em nuvem.

**PALAVRAS-CHAVE:** Modularização. Artefatos. Dummy Training. MLflow. Model Registry.

### Código e comandos
Nenhum bloco de código nesta aula. (Elementos técnicos citados inline: diretivas `workflow_call` e `workflow_run`; ações `actions/upload-artifact` e `actions/download-artifact`; formatos de binário `.pkl`, `.onnx`, `.pt`; evento `repository_dispatch`.)

### Ferramentas / serviços citados
GitHub Actions, MLflow (Model Registry), Kubernetes, ArgoCD, Istio, Datadog, Evidently AI, Arize, Amazon ECR, Amazon EKS, GitHub Secrets, DVC (implícito na janela de dados).

### Aplicabilidade ao Tech Challenge Fase 3
- Modularização de workflows e passagem de artefatos entre Jobs aplicam-se ao pipeline CI/CD (15%) separando dados → treino → deploy do classificador.
- Uso de MLflow como Model Registry dá rastreabilidade commit→modelo, útil para versionar o classificador NLP entregue.
- Estratégias Canary/Shadow e rollback via `repository_dispatch` orientam a entrega contínua do serviço de inferência.

### REFERÊNCIAS (Aula 3)
- GITHUB. *Store and share data with workflow artifacts*. GitHub Docs, 2026. Disponível em: https://docs.github.com/en/actions/tutorials/store-and-share-data. Acesso em: 6 abr. 2026.
- GITHUB. *Workflow artifacts*. GitHub Docs, 2026. Disponível em: https://docs.github.com/en/actions/concepts/workflows-and-actions/workflow-artifacts. Acesso em: 6 abr. 2026.
- MLFLOW. *MLflow Model Registry*. MLflow Documentation, 2026. Disponível em: https://mlflow.org/docs/latest/ml/model-registry/. Acesso em: 6 abr. 2026.

---

## Aula 4 — Testes Automatizados no Pipeline de ML
**Arquivo fonte:** `9MLET - Fase 3 - Integração com CICD - Aula 4.pdf` (15 páginas)
**Título na ementa:** Testes Automatizados no Pipeline de ML

### Conceitos-chave
- GIGO ("Garbage In, Garbage Out") como inimigo número um.
- A Tríade de Testes em IA: validação estrutural de código, contratos de dados, testes de comportamento do modelo.
- Testes unitários + Property-Based Testing (Hypothesis) e Shape Testing.
- Data Quality Gates (Great Expectations / Pandera).
- Testes Comportamentais: Invariância, Expectativa Direcional e Avaliação Fatiada (Slice-Based).
- Testes de Integração E2E com Service Containers (FastAPI, latência/SLA).

### Conteúdo

**O QUE VEM POR AÍ?**

De que adianta o código sintaticamente perfeito se a base de dados ingerida pelo modelo estiver corrompida?

Em Inteligência Artificial, o fenômeno do GIGO ("Garbage In, Garbage Out") é o inimigo número um. Nesta aula, vamos expandir a pirâmide de testes tradicionais de software para cobrir também as nuances de Machine Learning. Você entenderá que, para garantir a qualidade de um modelo, precisamos impor Contratos de Dados rígidos e garantir que a API de serviço não irá falhar por falta de memória ao inicializar.

**HANDS ON**

Nesta etapa prática, vamos adicionar "guardas de fronteira" no nosso pipeline. Utilizando a biblioteca Great Expectations (ou Pytest), o GitHub Actions vai reprovar o código se um CSV de amostra vier com colunas faltando ou formatos incorretos. Depois, utilizaremos Service Containers no GitHub para subir uma API web (FastAPI) e testar integrações ponta a ponta (E2E), garantindo que ela processa payloads JSON corretamente.

**SAIBA MAIS**

**A Tríade de Testes em Sistemas de Inteligência Artificial**

No desenvolvimento de software tradicional, a pirâmide de testes baseia-se em uma premissa determinística: se enviarmos a entrada A para a função X, devemos receber invariavelmente a saída B. Um pipeline de CI tradicional executa baterias de testes unitários, de integração e ponta a ponta (E2E) para garantir essa lógica binária (certo ou errado). Contudo, ao transicionarmos para o ecossistema de Machine Learning, essa fundação de testes colapsa se não for adaptada. Sistemas de IA são estocásticos e probabilísticos. O artefato final não é apenas o código escrito pelo desenvolvedor, mas sim o comportamento gerado pela compilação do código cruzado com a distribuição dos dados de treinamento.

Portanto, a arquitetura de qualidade em MLOps exige uma abordagem tridimensional. A esteira de Integração Contínua no GitHub Actions deixa de ser um mero validador de sintaxe e passa a atuar como um laboratório de inspeção rigoroso, dividido em três grandes pilares: validação estrutural do código de engenharia de features, garantia de contratos de dados pré-treinamento e, crucialmente, testes de comportamento do modelo treinado. Ignorar qualquer um desses pilares resulta na armadilha clássica do "falha silenciosa", onde o pipeline roda com sucesso, o deploy é feito, mas o modelo toma decisões enviesadas ou matematicamente absurdas em produção.

**Testes Unitários e a Blindagem da Engenharia de Features**

Em equipes de Ciência de Dados orientadas à experimentação ágil, é extremamente comum que a etapa de transformação de dados (a Feature Engineering) seja desenvolvida de forma orgânica e pouco estruturada. Um(a) cientista escreve uma função complexa no Pandas ou NumPy para imputar valores ausentes, calcular médias móveis ou normalizar distribuições. Durante a prova de conceito, essa função opera perfeitamente sobre o dataset estático de treinamento. No entanto, quando o código é promovido para a branch principal e um novo lote de dados de produção é processado na calada da noite, exceções não tratadas começam a surgir: uma divisão por zero oculta, um vetor que perdeu sua dimensionalidade ou um tipo de dado convertido inadvertidamente de float64 para object. Como os dados de ML fluem em pipelines longos, o erro na primeira etapa da transformação se propaga de forma destrutiva até o modelo final.

O desafio arquitetural fundamental é garantir que cada função de transformação matemática possua resiliência comprovada contra as mais severas anomalias estatísticas antes mesmo de tocar no algoritmo de aprendizado. O código que prepara os tensores precisa ser submetido a um estresse computacional que simule as imperfeições do mundo real, removendo a fragilidade típica de scripts desenvolvidos apenas para o "caminho feliz" (happy path).

Para estabelecer essa fundação, a equipe de MLOps implementa rotinas de testes rigorosas no GitHub Actions. Ao abrir um Pull Request, o workflow aciona Jobs que instalam a biblioteca pytest não apenas para verificações simples, mas combinada com frameworks de Property-Based Testing (como a biblioteca Hypothesis). Em vez de o desenvolvedor escrever testes estáticos (ex: "verifique se a função funciona para a idade 25"), a esteira de CI gera automaticamente milhares de casos de teste randômicos baseados em propriedades esperadas. O GitHub Actions injeta DataFrames sintéticos contendo valores infinitos, datas nulas, strings vazias e números negativos extremos diretamente nas funções de pré-processamento.

Simultaneamente, para arquiteturas de Deep Learning, o pipeline de testes impõe o Shape Testing. O GitHub Actions valida matematicamente a asserção das dimensões dos tensores ao passarem por cada camada da rede (`assert tensor.shape == esperado`). Como consequência direta, a robustez do repositório aumenta exponencialmente. Erros de redimensionamento de matrizes ou falhas de formatação são capturados na máquina virtual do GitHub Actions em frações de segundo, impedindo que código frágil seja fundido (merged) e evitando o desperdício de milhares de dólares em instâncias com GPU que fatalmente falhariam horas depois no meio de um treinamento produtivo.

**Contratos de Dados e Prevenção contra a "Entrada de Lixo"**

A estabilidade de um modelo de Machine Learning é refém da estabilidade de suas fontes de informação. Em ambientes corporativos dinâmicos, o banco de dados upstream é constantemente alterado por outras equipes de engenharia. Um engenheiro de backend pode decidir alterar o formato de uma coluna de data de DD/MM/YYYY para YYYY-MM-DD, ou o sistema de origem pode começar a enviar valores monetários em centavos em vez de reais. Como o código de leitura do modelo geralmente não quebra com essas mudanças (o Python continua lendo o arquivo perfeitamente), o pipeline de ML realiza o treinamento sobre dados semanticamente corrompidos. O resultado é o fenômeno do "Garbage In, Garbage Out" (GIGO). O GitHub Actions reportará uma execução de sucesso (código de saída 0), mas o artefato de IA gerado estará completamente cego para a realidade, destruindo o valor de negócio do produto.

A arquitetura do pipeline precisa ser estendida para incorporar o conceito de Data Quality Gates. A esteira de CI/CD deve atuar como um fiscal aduaneiro implacável, barrando a entrada de qualquer dataset que viole o esquema estrutural ou a distribuição estatística esperada pelo modelo. A validação deixa de ser sobre sintaxe e passa a ser estritamente sobre semântica de dados e integridade referencial.

A implementação técnica ocorre através de Contratos de Dados executáveis dentro do GitHub Actions. Antes da etapa de `model.fit()` ser sequer iniciada, o workflow faz o download do pacote de dados alvo e aciona bibliotecas de validação especializadas, como Great Expectations ou Pandera. A equipe de Ciência de Dados documenta antecipadamente as regras de negócio em arquivos YAML ou decorators Python: a coluna "Idade" deve ter 99% de seus valores entre 18 e 95; a coluna "Categoria_Cliente" não pode conter categorias novas não mapeadas no encoder; a proporção de valores nulos na coluna "Renda" não pode exceder 5%.

Quando o GitHub Actions roda essa suíte de testes de dados, ele levanta o perfil estatístico do arquivo atual e o compara com o contrato. Se o banco de dados upstream inseriu anomalias silenciosas, a validação de dados emite uma exceção de Data Drift ou Schema Violation. O pipeline é abortado imediatamente com uma marcação vermelha (falha). O desenvolvedor é notificado diretamente no Pull Request ou no canal de comunicação da equipe com um relatório HTML gerado pelo Great Expectations, detalhando exatamente qual coluna violou qual regra. Essa governança automatizada extingue as falhas silenciosas, garantindo que os algoritmos corporativos só aprendam a partir de bases de dados certificadas e matematicamente saudáveis.

**Testes Comportamentais: Além da Acurácia Global**

Um dos erros metodológicos mais perigosos em projetos de Machine Learning é a dependência exclusiva de métricas globais de performance para aprovar a liberação de um modelo. Uma rede neural projetada para análise de currículos pode apresentar uma precisão fantástica de 92% no conjunto de dados de validação. Aprovada por esse número único, ela é enviada para produção. Semanas depois, uma auditoria de negócios revela que o modelo atinge 99% de precisão para candidatos do sexo masculino, mas erra grosseiramente e rejeita 80% dos currículos de candidatas do sexo feminino. A métrica global (92%) ofuscou completamente as falhas locais graves. O modelo era sintaticamente perfeito e obedeceu aos contratos de dados, mas seu "comportamento" ético e operacional era falho e discriminatório.

O processo de Integração Contínua deve transcender a análise de perdas e acurácias médias. É imperativo adotar uma metodologia de Testes Comportamentais (frequentemente inspirada na abordagem CheckList apresentada por pesquisadores de IA). O objetivo é tratar o algoritmo treinado como uma entidade caixa-preta e submetê-lo a cenários extremos, perturbações lógicas e recortes demográficos específicos para provar que suas predições não apenas estão corretas na média, mas são justas, consistentes e invariáveis diante de ruídos insignificantes.

Logo após a etapa de Dummy Training (ou treinamento em amostra) no GitHub Actions, um Job específico de Behavioral Testing é engatilhado utilizando ferramentas como o Deepchecks ou roteiros customizados no pytest. O pipeline aplica três categorias de testes diretamente nas funções de inferência:

1. **Testes de Invariância (Invariance Tests):** O CI altera detalhes irrelevantes no dado de entrada (como mudar o nome de um cliente de "João" para "Maria" em um modelo de risco de crédito, ou adicionar erros de digitação leves em um processador de linguagem natural). O pipeline impõe uma asserção de que a predição final não deve mudar. Se mudar, o modelo está aprendendo correlações espúrias.
2. **Testes de Expectativa Direcional (Directional Expectation Tests):** O CI manipula variáveis com correlação lógica conhecida. Em um modelo de precificação de imóveis, o GitHub Actions dobra a metragem quadrada de uma casa no dataset de teste e valida se a predição de preço do modelo aumentou. Se o modelo prever um preço menor, ele violou uma lei da física do negócio e o CI quebra.
3. **Avaliação Fatiada (Slice-Based Testing):** O GitHub Actions divide o dataset de validação em subgrupos críticos para a corporação (ex: clientes VIPs vs. clientes normais, diferentes etnias ou faixas etárias) e força o cálculo das métricas de performance para cada fatia individualmente. O CI é configurado para falhar se a discrepância de performance entre os subgrupos ultrapassar a margem de tolerância estipulada pelas políticas de equidade da empresa.

Essa suíte de testes transforma o GitHub Actions em um auditor algorítmico implacável. Modelos enviesados, frágeis a ruídos textuais ou com comportamentos de regressão não linear ilógicos são interceptados e bloqueados na esteira, muito antes de causarem desastres de relações públicas ou exporem a organização a riscos regulatórios.

**Testes de Integração e a Camada de Serviço (E2E)**

Em estágios maduros de MLOps, o modelo preditivo não vive isolado; ele é encapsulado dentro de um contêiner (Docker), exposto através de um framework web (como FastAPI ou Flask) e serializado para receber requisições em formato JSON via chamadas REST ou gRPC. É assustadoramente comum que um modelo que passou em todos os testes unitários, de dados e comportamentais falhe espetacularmente no momento do Deploy. O motivo reside nas falhas de serialização: a API recebe um payload JSON, mas falha ao convertê-lo para o tensor NumPy exigido pelo modelo, ou o contêiner esgota sua memória RAM ao tentar carregar os pesos massivos da rede neural na inicialização do serviço web, resultando em um CrashLoopBackOff no cluster Kubernetes.

A arquitetura de testes precisa cobrir a milha final. O pipeline de CI/CD deve provar que não apenas a matemática do modelo funciona, mas que a infraestrutura de serviço que o envelopa consegue se comunicar com o mundo externo de forma eficiente, mantendo limites rígidos de latência (Service Level Agreements - SLAs) e lidando com requisições concorrentes sem corromper a memória alocada.

A última barreira de qualidade no GitHub Actions é configurada como uma etapa de Integração Ponta a Ponta (End-to-End). Em vez de rodar scripts Python soltos, o workflow utiliza a funcionalidade de Service Containers do próprio GitHub. O CI realiza o build da imagem Docker de produção contendo a API e o modelo treinado. Ele levanta esse contêiner dentro da máquina virtual do Runner e aguarda a porta 8000 ficar disponível. Em seguida, o GitHub Actions executa uma suíte de testes de integração (usando bibliotecas como o requests ou o locust em um segundo processo). Ele bombardeia o contêiner em execução com payloads JSON reais que simulam requisições de clientes.

O pipeline avalia três fatores críticos:

1. **Serialização Segura:** Confirma se a resposta da API retorna um HTTP 200 OK com o formato de predição esperado, validando a conversão de strings da web para matrizes matemáticas.
2. **Tratamento de Exceções na Borda:** Envia requisições com campos ausentes ou tipos de dados incorretos e verifica se a API retorna um erro claro (como HTTP 422 Unprocessable Entity), em vez de quebrar internamente e retornar um genérico HTTP 500.
3. **Latência de Inferência:** O CI mensura o tempo de resposta (Response Time). Se o contrato arquitetural define que uma recomendação deve ser gerada em menos de 200 milissegundos, e o novo modelo pesado eleva essa latência para 600 milissegundos, o GitHub Actions marca a branch com falha por violação de SLA de performance, bloqueando o merge.

Com esse fluxo perfeitamente encadeado, o abismo entre o ambiente do Cientista de Dados e o ambiente do Engenheiro de Software é completamente fechado. A organização adquire a certeza técnica de que todo artefato aprovado por esta esteira possui integridade matemática, pureza semântica, comportamento justo e prontidão operacional máxima para servir aos usuários finais com excelência.

**O QUE VOCÊ VIU NESTA AULA?**

Nesta aula, compreendemos que testes unitários tradicionais são insuficientes para sistemas de IA. Aprendemos a validar o formato, esquema e comportamento estatístico dos dados antes que o modelo tente aprender algo a partir deles (Contratos de Dados). Também vimos como envelopar nosso modelo em uma API de inferência e utilizar o próprio ambiente do GitHub Actions para testar a comunicação HTTP e a resposta do contêiner, simulando a experiência do cliente final.

**PALAVRAS-CHAVE:** GIGO. Contratos de Dados. Great Expectations. Testes E2E. FastAPI.

### Código e comandos
Nenhum bloco de código nesta aula. (Elementos técnicos citados inline: asserção `assert tensor.shape == esperado`; chamada `model.fit()`; formatos de data DD/MM/YYYY e YYYY-MM-DD; porta `8000`; códigos HTTP 200, HTTP 422 Unprocessable Entity, HTTP 500.)

### Ferramentas / serviços citados
GitHub Actions (Service Containers), Great Expectations, Pandera, pytest, Hypothesis, Deepchecks, Pandas, NumPy, Docker, FastAPI, Flask, requests, locust, Kubernetes, REST, gRPC.

### Aplicabilidade ao Tech Challenge Fase 3
- A Tríade de Testes (unitário + contratos de dados + comportamental) é diretamente aplicável aos testes automatizados exigidos para o classificador NLP de laudos médicos.
- Testes de Invariância/Slice-Based ajudam a auditar viés e robustez a ruído textual — relevante para laudos médicos em linguagem natural.
- Testes E2E com Service Containers validam a API FastAPI de inferência antes do deploy, cobrindo containerização + CI/CD (15%).

### REFERÊNCIAS (Aula 4)
- GREAT EXPECTATIONS. *GX Cloud overview*. Great Expectations Documentation, 2026. Disponível em: https://docs.greatexpectations.io/docs/cloud/overview/gx_cloud_overview. Acesso em: 6 abr. 2026.
- MCATEER, Matthew. *Machine learning testing*. 2020. Disponível em: https://matthewmcateer.me/blog/machine-learning-testing/. Acesso em: 6 abr. 2026.
- MICROSOFT. *Trabalhar com lint e testes de unidade no GitHub Actions*. Microsoft Learn, 2026. Disponível em: https://learn.microsoft.com/pt-br/training/modules/work-linting-unit-test-github-actions/. Acesso em: 22 fev. 2026.

---

## Aula 5 — Automatizando o Deploy de Modelos (Parte 1: Containerização)
**Arquivo fonte:** `9MLET - Fase 3 - Integração com CICD - Aula 5.pdf` (15 páginas)
**Título na ementa:** Automatizando o Deploy de Modelos (Parte 1: Containerização)

### Conceitos-chave
- Environment Drift e a inviabilidade do `requirements.txt` puro para ML.
- Docker como contrato de imutabilidade; Dockerfile declarando SO base + binários.
- Pesos Embutidos (baked) vs. Injetados (download em runtime a partir de S3/GCS).
- Otimização: Multi-Stage Builds (Builder vs. Runtime) e minificação de imagens.
- Caching Distribuído com `docker/build-push-action` (`cache-from` / `cache-to`) e GHCR.
- Etiquetagem Dinâmica (`docker/metadata-action`): SHA, SemVer e Ambiente; push com OIDC.

### Conteúdo

**O QUE VEM POR AÍ?**

A deriva de ambiente (Environment Drift) é a razão pela qual muitos modelos nunca saem do laboratório. Se a versão do pacote de álgebra linear do servidor de produção divergir uma única vírgula do notebook do cientista de dados, o modelo pode gerar resultados errados sem lançar erros no log. Nesta aula, vamos congelar a nossa fábrica de IA. Você entenderá por que o Docker é inegociável em MLOps e como usar o GitHub Actions para construir, otimizar e armazenar imagens imutáveis, preparando o terreno para uma infraestrutura sólida.

**HANDS ON**

Nesta etapa prática, abandonaremos a instalação de scripts soltos. Vamos redigir um Dockerfile otimizado (Multi-stage Build) que separa a compilação pesada do tempo de execução enxuto. No GitHub Actions, implementaremos a ação de Build e Push, e configuraremos o Layer Caching usando o GHCR (GitHub Container Registry) para que a construção de bibliotecas de IA não demore horas em todo PR.

**SAIBA MAIS**

**O Abismo Entre o Algoritmo e o Sistema**

Nas fases iniciais do ciclo de vida de Machine Learning, o foco da equipe de dados é quase inteiramente voltado para a experimentação matemática. Ajustam-se hiperparâmetros, testam-se diferentes arquiteturas de redes neurais e validam-se métricas como Recall e Precision em um ambiente controlado, frequentemente um Jupyter Notebook rodando na máquina local do cientista ou em uma instância de pesquisa. Quando o modelo atinge a performance desejada, surge a ilusão de que o trabalho terminou. No entanto, o código Python que invoca a predição (`model.predict()`) é apenas a ponta do iceberg.

O verdadeiro desafio da engenharia de IA começa na tentativa de transportar essa lógica probabilística para o mundo real. O ecossistema de Machine Learning é notoriamente frágil em relação às suas dependências. Um modelo treinado com uma versão específica do scikit-learn, rodando sobre uma versão exata do NumPy, que por sua vez compila contra uma biblioteca C++ específica no sistema operacional nativo, irá invariavelmente falhar — ou pior, gerar predições silenciosamente incorretas — se qualquer uma dessas camadas for ligeiramente alterada no servidor de produção.

Para cruzar o abismo entre o ambiente de pesquisa e o ambiente de serviço, a engenharia de plataforma precisa abolir a dependência do sistema operacional hospedeiro. A solução fundacional para a automatização do deploy não é copiar arquivos de código, mas sim empacotar toda a "fábrica" de predição em um artefato isolado, imutável e portável. Este é o domínio da Containerização, a primeira metade do processo de implantação moderna, orquestrada de forma rigorosa pela esteira de CI/CD.

**A Ilusão do requirements.txt e a Matriz de Dependências**

Em operações imaturas, a transição para produção geralmente ocorre através do versionamento de um arquivo `requirements.txt`. O pipeline de integração contínua (CI) clona o repositório em um servidor de produção, cria um ambiente virtual Python (venv) e executa a instalação dos pacotes. Essa abordagem, aceitável para microsserviços web simples, é catastrófica para Machine Learning.

Quando um algoritmo depende de aceleração por hardware (GPUs), a matriz de dependências extrapola a linguagem Python. O modelo exige a presença de drivers de vídeo proprietários (como os da NVIDIA), o toolkit CUDA para compilação paralela e bibliotecas de aprendizado profundo (como cuDNN). Se o cientista de dados treinou o modelo em um MacOS com arquitetura ARM (Apple Silicon) e o pipeline tenta implantar o código bruto em um servidor Linux x86 na nuvem, o processo colapsa. Bibliotecas matemáticas compiladas falham em encontrar os ponteiros de memória corretos, e a aplicação sequer inicializa. O "funciona na minha máquina" torna-se um passivo corporativo.

A intervenção arquitetural exige que o estado exato da máquina do pesquisador seja cristalizado. A equipe adota a tecnologia Docker não apenas como uma ferramenta de empacotamento, mas como um contrato de imutabilidade. O Cientista de Dados e o Engenheiro de Machine Learning colaboram para redigir um arquivo declarativo, o Dockerfile. Este arquivo não declara apenas as bibliotecas Python; ele declara o sistema operacional base (ex: Ubuntu 22.04), as variáveis de ambiente do sistema e os binários de sistema de baixo nível necessários para a inferência.

Essa mudança de paradigma transforma o artefato de release. O GitHub Actions deixa de transportar arquivos `.py` e pesos `.pkl` soltos. A responsabilidade da esteira passa a ser a construção sistemática desta imagem de contêiner. Quando a imagem é gerada, ela garante que o código dentro dela se comportará de forma absolutamente idêntica, seja rodando no laptop de um novo estagiário, no Runner do GitHub Actions ou no cluster de alta disponibilidade em produção. O problema da deriva de ambiente (environment drift) é erradicado na raiz.

**Anatomia de um Contêiner de IA: Pesos Embutidos vs. Injetados**

A transição para contêineres introduz um dilema arquitetural profundo exclusivo aos projetos de Inteligência Artificial: onde os pesos do modelo (frequentemente arquivos binários pesando gigabytes) devem residir?

Uma abordagem ingênua é simplesmente copiar o arquivo do modelo estático diretamente para dentro do Dockerfile (usando o comando `COPY model.pt /app/`). Embora isso crie um artefato perfeitamente autocontido, gera um efeito colateral severo. Toda vez que o modelo é retreinado com novos dados — mesmo que o código da API ou as dependências Python não tenham sofrido uma única alteração —, o GitHub Actions é forçado a reconstruir toda a imagem Docker, criar uma nova versão pesada de 5 Gigabytes e fazer o upload dessa massa de dados para o Container Registry. Isso satura a largura de banda da rede, consome o armazenamento da nuvem e torna a implantação dolorosamente lenta.

Para resolver esse gargalo, a engenharia de plataforma implementa o padrão de Desacoplamento de Estado e Lógica. O código da aplicação (a API FastAPI que recebe a requisição, as funções de pré-processamento de dados e as versões fixas das bibliotecas) é "assado" (baked) na imagem Docker. Esta imagem é leve e muda apenas quando a engenharia de software muda.

Os pesos do modelo, por outro lado, são tratados como configuração dinâmica. O pipeline de GitHub Actions é configurado para, ao final do treinamento, enviar o arquivo de pesos para um armazenamento de objetos (como AWS S3 ou Google Cloud Storage). O código dentro do contêiner Docker é projetado para atuar de forma reativa: no exato momento em que o contêiner é iniciado em produção, ele faz o download do binário correspondente à versão exigida a partir da nuvem, carrega na memória RAM e inicia o serviço.

Essa arquitetura eleva drasticamente a maturidade do MLOps. A equipe consegue implantar e alternar entre cinco versões diferentes do mesmo modelo de fraude em questão de segundos, apenas alterando uma variável de ambiente que diz ao contêiner qual "peso" ele deve baixar, sem nunca precisar disparar o moroso processo de build no GitHub Actions para alterações puramente estatísticas.

**Otimização Extrema: Multi-Stage Builds e Minificação de Imagens**

Mesmo com os pesos do modelo desacoplados, contêineres de Ciência de Dados tendem à obesidade crônica. Bibliotecas como TensorFlow ou PyTorch costumam adicionar centenas de megabytes ou até gigabytes ao tamanho da imagem. Além disso, para instalar certos pacotes Python a partir do código-fonte, o Dockerfile frequentemente exige a instalação de compiladores pesados como gcc ou g++. Se esses compiladores forem deixados dentro da imagem final de produção, a empresa não apenas sofrerá com o custo de armazenamento de imagens gigantescas, mas também ampliará criticamente sua superfície de ataque. Invasores adoram encontrar contêineres de produção que possuam ferramentas de compilação embutidas, pois facilitam a injeção de malwares pós-invasão.

A resolução arquitetural aplicada pelo pipeline de CI é a utilização de Docker Multi-Stage Builds (Construção em Múltiplas Etapas). O arquivo de configuração da imagem é dividido em dois blocos lógicos.

O primeiro estágio é rotulado como Builder (Construtor). O GitHub Actions levanta uma imagem base robusta contendo todas as ferramentas de desenvolvimento do sistema operacional. Neste ambiente temporário, o código é compilado, as bibliotecas matemáticas são otimizadas via Cython e os pacotes Python são instalados e agregados em um diretório virtual limpo. O segundo estágio é rotulado como Runtime (Tempo de Execução). Aqui, a esteira inicia uma imagem base minúscula e altamente segura (como o `python:3.10-slim` ou distros baseadas em Alpine/Distroless). A mágica acontece quando o Docker é instruído a copiar apenas os binários compilados e os arquivos de bibliotecas já processados do estágio Builder para o estágio Runtime, abandonando todo o lixo de compilação, cabeçalhos C e ferramentas de desenvolvimento no vazio.

O resultado dessa prática na esteira do GitHub Actions é formidável. Imagens de aprendizado profundo que originalmente pesavam 4 GB são reduzidas a 600 MB. Essa compressão drástica diminui o tempo gasto no `docker push` pela metade, acelera significativamente a velocidade com que o cluster de produção consegue puxar a nova imagem (mitigando problemas de Cold Start explorados em infraestrutura) e estabelece um padrão rigoroso de Segurança da Informação, removendo vetores de ataque desnecessários do ambiente que servirá ao cliente final.

**Caching Distribuído: Acelerando a CI de Machine Learning**

A incorporação da construção de imagens Docker no GitHub Actions traz um novo desafio temporal. Diferente da compilação de linguagens como Go ou Rust, a instalação de matrizes científicas via pip é demorada, e o pacote do Docker tem que baixar a mesma imagem do Ubuntu base repetidas vezes. Em uma equipe ativa, se dez Pull Requests são abertos no mesmo dia, o GitHub Actions passará horas repetindo a mesma instalação da biblioteca Pandas em dez máquinas virtuais (Runners) diferentes. O tempo de feedback da Integração Contínua sofre, frustrando os cientistas de dados que precisam esperar 20 minutos apenas para que a esteira valide se a imagem do contêiner consegue ser montada com sucesso.

A engenharia de eficiência ataca este problema implementando camadas de Caching Distribuído. O Docker possui um sistema interno inteligente de camadas (Layer Caching); se uma linha no Dockerfile não mudou, ele reaproveita o processamento anterior. Contudo, em CI/CD corporativo moderno, as máquinas (Runners) são efêmeras — elas são destruídas após cada job, apagando o cache local.

Para preservar a inteligência da construção entre diferentes execuções e branches, a equipe orquestra o GitHub Actions para utilizar Cache Registries externos. Utilizando a ação oficial `docker/build-push-action`, o pipeline é configurado com os parâmetros `cache-from` e `cache-to`. A mecânica ocorre da seguinte forma: quando a esteira começa a construir o artefato de IA, ela primeiro consulta o GitHub Container Registry (GHCR) corporativo. Ela verifica se existe uma versão em cache das camadas mais pesadas (ex: a instalação das dependências do sistema ou a compilação inicial dos pacotes Python). Se a esteira encontra um acerto (Cache Hit), ela simplesmente faz o download daquela camada binária em segundos, em vez de recompilá-la por minutos.

Como a maioria das alterações de código em MLOps ocorre nas últimas linhas do projeto (modificação na lógica da feature engineering ou na rota da API web), os primeiros 90% da imagem Docker são instantaneamente reaproveitados da nuvem. O tempo de execução da etapa de conteinerização despenca de 15 minutos para menos de 90 segundos. A equipe atinge o estado da arte em Developer Experience (DX), onde as aprovações de Pull Requests ocorrem de forma fluida, e o custo financeiro cobrado pelo GitHub Actions pelos minutos de computação utilizados é reduzido drasticamente.

**Orquestração Segura: Etiquetagem Semântica e Registro de Contêineres**

Com a imagem do modelo imutável, enxuta e otimizada via cache, a etapa final da automatização da conteinerização reside em como essa imagem é catalogada e guardada. Em muitas equipes em transição, a esteira do GitHub Actions gera a imagem Docker e a empurra para o registro (como AWS Elastic Container Registry - ECR ou DockerHub) utilizando permanentemente a etiqueta `latest` (a mais recente).

O uso da etiqueta `latest` é um antipadrão arquitetural letal em sistemas preditivos. Se o modelo de produção está apontando para `modelo-fraude:latest` e a esteira do GitHub Actions empurrar silenciosamente uma versão corrompida também chamada de `latest`, o sistema não tem garantia de rastreabilidade ou previsibilidade de quando a infraestrutura absorverá a mudança. Pior ainda, um rollback instantâneo torna-se impossível, pois a referência à versão funcional anterior foi sobrescrita.

A governança do pipeline exige uma rastreabilidade criptográfica contínua entre o código versionado e a imagem implantada. A etapa final do workflow de conteinerização no GitHub Actions é configurada com estratégias de Etiquetagem Dinâmica (Dynamic Tagging).

Através do módulo `docker/metadata-action`, o pipeline extrai metadados essenciais diretamente do evento do Git no momento em que a imagem é construída. O workflow nunca sobe uma única etiqueta. Em vez disso, aplica múltiplas etiquetas (Multi-tagging) simultâneas na mesma imagem binária:

1. **Etiqueta de SHA:** O GitHub Actions etiqueta a imagem com os 7 primeiros caracteres do hash criptográfico do commit (ex: `modelo-fraude:sha-a1b2c3d`). Isso garante que o Engenheiro de Operações olhando o Container Registry saiba exatamente qual linha de código gerou aquele contêiner.
2. **Etiqueta de SemVer:** Se o evento do GitHub for a publicação de um Release oficial, o pipeline lê a etiqueta da versão e marca a imagem semanticamente (ex: `modelo-fraude:v2.1.0`). Isso dita a compatibilidade da API e do negócio.
3. **Etiqueta de Ambiente:** Baseada na branch, o artefato pode receber a marcação `staging` ou `production-candidate`, auxiliando as ferramentas de Entrega Contínua a saberem o que podem ou não puxar.

Com a imagem rotulada com precisão militar, o GitHub Actions finaliza a sua fase 1. Utilizando autenticação segura OIDC (OpenID Connect, para evitar o vazamento de chaves de serviço), o Runner autentica-se contra o Container Registry corporativo e realiza o `docker push`.

O ciclo se encerra com sucesso. A corporação agora possui um cofre centralizado onde cada artefato de Inteligência Artificial reside de forma imutável, auditável, comprimida, imune a inconsistências de ambiente e matematicamente vinculada ao momento exato em que a ciência foi transformada em código. O modelo está oficialmente empacotado e pronto para a próxima fase crítica da engenharia de plataformas: a implantação contínua e a orquestração no cluster, assunto de nossa próxima aula.

**O QUE VOCÊ VIU NESTA AULA?**

Nesta aula, aprendemos a isolar o nosso modelo das dependências do sistema operacional hospedeiro através da conteinerização. Vimos que imagens de IA podem pesar gigabytes, e implementamos estratégias arquiteturais (Multi-stage Builds) para reduzir esse tamanho drásticamente. Também configuramos o GitHub Actions para compilar essas imagens e enviá-las para um repositório centralizado de forma segura, utilizando técnicas de cache distribuído para manter a esteira rápida e barata.

**PALAVRAS-CHAVE:** Docker. Imutabilidade. GHCR. Layer Caching. Multi-stage Build.

### Código e comandos
Nenhum bloco de código nesta aula. (Elementos técnicos citados inline: chamada `model.predict()`; arquivo `requirements.txt`; SO base `Ubuntu 22.04`; extensões `.py` / `.pkl`; instrução `COPY model.pt /app/`; imagem base `python:3.10-slim`; comando `docker push`; ação `docker/build-push-action` com `cache-from` / `cache-to`; ação `docker/metadata-action`; tags `modelo-fraude:latest`, `modelo-fraude:sha-a1b2c3d`, `modelo-fraude:v2.1.0`, `staging`, `production-candidate`.)

### Ferramentas / serviços citados
Docker / Dockerfile, GitHub Actions, GHCR (GitHub Container Registry), scikit-learn, NumPy, NVIDIA drivers, CUDA, cuDNN, FastAPI, AWS S3, Google Cloud Storage, TensorFlow, PyTorch, gcc, g++, Cython, Alpine, Distroless, `docker/build-push-action`, `docker/metadata-action`, AWS Elastic Container Registry (ECR), DockerHub, OIDC (OpenID Connect).

### Aplicabilidade ao Tech Challenge Fase 3
- Multi-Stage Build e imagem enxuta (`python:*-slim`) são o padrão recomendado para containerizar o serviço de inferência NLP do TC.
- Layer Caching com `docker/build-push-action` + GHCR mantém o pipeline CI/CD (15%) rápido e barato ao dockerizar o classificador.
- Etiquetagem por SHA/SemVer e push via OIDC dão rastreabilidade e segurança na publicação da imagem do modelo.

### REFERÊNCIAS (Aula 5)
- DOCKER. *Build and push Docker images with GitHub Actions*. Docker Documentation, 2026. Disponível em: https://docs.docker.com/build/ci/github-actions/. Acesso em: 6 abr. 2026.
- TOWARDS DATA SCIENCE. *Docker for data science*. Towards Data Science, 2026. Disponível em: https://towardsdatascience.com/from-chaos-to-consistency-docker-for-data-scientists-240372adff18/. Acesso em: 24 fev. 2026.
- TOWARDS DATA SCIENCE. *How to Dockerize your machine learning model*. Towards Data Science, 2026. Disponível em: https://towardsdatascience.com/how-to-dockerize-any-machine-learning-application-f78db654c601/. Acesso em: 24 fev. 2026.

---

## Aula 6 — Automatizando o Deploy de Modelos (Parte 2: Entrega Contínua)
**Arquivo fonte:** `9MLET - Fase 3 - Integração com CICD - Aula 6.pdf` (15 páginas)
**Título na ementa:** Automatizando o Deploy de Modelos (Parte 2: Entrega Contínua)

### Conceitos-chave
- Fim do "Big Bang Deploy"; Canary Release (5% → 20% → 50% → 100%).
- Shadow Deployments (Traffic Mirroring) para setores ultrarregulamentados.
- Warmup de Modelos e Smoke Tests (Readiness Probes) contra Cold Start.
- Rollback automatizado baseado em métricas via `on: repository_dispatch`.
- Governança: GitHub Environments (aprovação manual) + OIDC (Zero-Trust, tokens efêmeros).

### Conteúdo

**O QUE VEM POR AÍ?**

Substituir a versão 1.0 de um modelo pela versão 2.0 da noite para o dia para 100% dos usuários é uma aposta altíssima. Em IA, se a versão 2.0 estiver com um viés estatístico oculto, as perdas financeiras serão imediatas.

Nesta aula, entramos no reino do Continuous Delivery e da orquestração. Vamos explorar como o GitHub Actions consegue manipular rotas de rede na infraestrutura de nuvem, permitindo liberações progressivas e blindando o acesso aos servidores sem depender de senhas fixas e inseguras.

**HANDS ON**

Nesta etapa prática, configuraremos a autenticação Zero-Trust. Você fará o GitHub Actions solicitar um Token efêmero na nuvem usando OIDC (OpenID Connect). Com o acesso liberado, simularemos a implantação de um Canary Release, configurando os manifestos para enviar apenas 5% do tráfego para a nossa nova imagem de ML. Além disso, forçaremos a exigência de uma aprovação humana (Environment Approval) antes do código chegar à produção.

**SAIBA MAIS**

No paradigma de engenharia de software tradicional, a transição da Integração Contínua (CI) para a Entrega Contínua (CD) é frequentemente tratada como uma formalidade: se o código passou nos testes automatizados, basta atualizar a versão no servidor. Em projetos de Inteligência Artificial, essa premissa é profundamente perigosa. Um modelo de Machine Learning pode ter um código sintaticamente impecável, passar em todos os testes de contrato de dados e possuir uma imagem Docker otimizada, mas ainda assim falhar de forma espetacular — e silenciosa — ao ser exposto à distribuição de dados do mundo real.

A fase de Entrega Contínua em MLOps não é apenas um mecanismo de cópia de arquivos; é a última linha de defesa sociotécnica da organização. O GitHub Actions assume aqui o papel de um orquestrador de risco. Ele deve gerenciar a introdução da nova lógica preditiva na infraestrutura (como um cluster Kubernetes ou um serviço gerenciado como o AWS SageMaker), garantindo que a transição ocorra sem tempo de inatividade (Zero Downtime), que o modelo seja aquecido adequadamente para evitar picos de latência, e que o raio de explosão (Blast Radius) de uma predição ruim seja matematicamente contido. A seguir, dissecamos as táticas fundamentais dessa arquitetura de implantação.

**O Fim do "Big Bang Deploy" e a Ascensão do Canary Release**

Em operações de dados incipientes, a implantação de uma nova versão de modelo costuma seguir a estratégia do "Big Bang" (ou substituição total). A equipe de Ciência de Dados desenvolve a versão 2.0 de um algoritmo de precificação dinâmica, o pipeline de CI cria a imagem Docker e, na calada da noite, o engenheiro de operações desliga a versão 1.0 e liga a versão 2.0 para 100% dos usuários. Se o novo modelo contiver um viés não detectado — por exemplo, reduzindo drasticamente os preços em uma região específica por causa de uma variável de clima mal interpretada —, o impacto financeiro para a empresa é imediato, severo e irreversível. O pânico se instaura, a equipe de plantão é acionada de madrugada, e a reversão manual custa horas de estresse e perda de receita.

O imperativo arquitetural para sistemas críticos é a erradicação absoluta das implantações de substituição total. A confiança estatística obtida no ambiente de testes (CI) nunca se traduz perfeitamente em confiança operacional (CD). A infraestrutura deve suportar a exposição progressiva.

Para alcançar essa resiliência, a equipe de plataforma implementa a estratégia de Canary Release (Lançamento Canário) utilizando o GitHub Actions em conjunto com controladores de tráfego (como Istio Service Mesh ou AWS ALB). Nesta topologia, quando o workflow de implantação é acionado, o GitHub Actions não apaga o modelo antigo (V1). Em vez disso, ele implanta o novo contêiner (V2) lado a lado com o anterior. O pipeline então executa um comando ou atualiza um manifesto de infraestrutura alterando as regras do roteador de borda da nuvem: "Direcione 95% do tráfego de clientes para o modelo V1 estabelecido, e envie uma amostra aleatória de apenas 5% para o novo modelo V2".

O pipeline entra em um estado de espera (wait/sleep) ou monitoramento ativo. Durante um período pré-determinado (ex.: 30 minutos), ferramentas de telemetria analisam o comportamento dos 5%. A latência da API V2 está dentro do SLA de 200ms? A taxa de erros HTTP 500 está em zero? A distribuição das predições de preços faz sentido financeiro? Se os alertas permanecerem silenciosos, o GitHub Actions automatiza as próximas fases, promovendo o roteamento para 20%, 50% e, finalmente, 100%. Essa abordagem transforma implantações temidas em eventos não-eventos (rotineiros). O raio de explosão de um modelo "alucinado" é confinado a uma fração mínima da base de usuários, protegendo a reputação da marca e permitindo validação em dados reais (produtivos) com risco sistêmico quase nulo.

**Shadow Deployments: A Arte do Teste Silencioso**

Mesmo com o Canary Release, a liberação de modelos em setores ultrarregulamentados — como detecção de fraudes bancárias ou diagnósticos de saúde por imagem — pode considerar um erro em 5% das requisições um risco inaceitável. Se o modelo V2 (canário) classificar incorretamente o exame médico de um paciente real, a consequência legal e humana é devastadora. A diretoria de risco frequentemente veta o Canary, exigindo uma prova irrefutável de que o modelo funciona em produção sem que nenhum cliente real receba a resposta dessa nova versão experimental.

O desafio é como testar um contêiner no ambiente de produção, recebendo a carga de rede e a distribuição de dados do mundo real, mas isolando completamente o seu resultado do fluxo de negócios e do conhecimento do usuário final.

A engenharia de plataformas resolve esse dilema através dos Shadow Deployments (Implantações Fantasmas). Semelhante ao Canary, o GitHub Actions implanta o contêiner do modelo V2 ao lado do V1. No entanto, o pipeline instrui o API Gateway (ou Service Mesh) a iniciar o tráfego espelhado (Traffic Mirroring / Shadowing). Quando um usuário envia uma transação para análise de fraude, a requisição é interceptada pelo roteador principal. O roteador encaminha a requisição primária para o modelo V1, aguarda a resposta e a devolve ao usuário (garantindo que o negócio opere sob a lógica testada). Silenciosamente, em uma thread paralela e assíncrona, o roteador envia uma cópia exata dessa mesma requisição para o modelo V2.

O modelo V2 processa o dado real e gera a sua predição, mas essa resposta é descartada da rede e enviada diretamente para um banco de dados de telemetria analítica ou um Feature Store. O usuário nunca interage com o V2. Após dias rodando em modo fantasma, a equipe de Ciência de Dados ganha um tesouro analítico: eles podem comparar matematicamente como o modelo V1 e o V2 reagiram aos mesmos eventos exatos do mundo real. O GitHub Actions pode até mesmo ser configurado para rodar um Job noturno que lê essa tabela de divergências e posta um relatório automático na Issue do GitHub.

Quando a equipe finalmente decide mudar o tráfego principal para o V2, a decisão não é mais uma aposta estatística; é uma certeza comprovada empiricamente pelo comportamento do modelo em um ambiente de produção hostil, garantindo uma transição à prova de falhas.

**Pós-Deploy: Warmup de Modelos e Smoke Tests**

A inicialização de um aplicativo web tradicional é quase instantânea. Contudo, em Machine Learning, especialmente em redes neurais profundas que processam linguagem natural (NLP) ou visão computacional, inicializar o serviço é uma operação traumática para o servidor. Quando o contêiner Docker sobe, ele precisa carregar tensores de múltiplos gigabytes do disco rígido (ou da rede) para a memória RAM e realizar alocações custosas nos registradores da GPU. Se o pipeline de CD declarar o contêiner como "pronto" assim que a porta HTTP responder, e o balanceador de carga enviar tráfego imediatamente, as primeiras centenas de requisições falharão miseravelmente por Timeout (tempo limite excedido). A primeira inferência (o primeiro forward pass) de um modelo costuma ser ordens de grandeza mais lenta do que as subsequentes — um problema clássico conhecido como Cold Start algorítmico.

A arquitetura de entrega não pode transferir a dor do aquecimento para o primeiro usuário final que clicar no sistema. A esteira deve orquestrar procedimentos de injeção de prontidão antes que o contêiner seja exposto ao trânsito público.

A estratégia implementada no GitHub Actions envolve a criação de testes de fumaça (Smoke Tests) acoplados a sondas de prontidão (Readiness Probes). O pipeline de CD realiza o deploy do contêiner, mas não atualiza a rota de tráfego. Imediatamente, o GitHub Actions dispara um script interno que atua como um "aquecedor" de modelo (Model Warmup). Ele bombardeia o novo contêiner isolado com uma carga massiva de payloads falsos pré-gravados, forçando o modelo a carregar todos os seus pesos, compilar os grafos do TensorFlow/PyTorch e preencher as tabelas de cache na memória da GPU.

Apenas após o contêiner responder consistentemente a essas requisições sintéticas abaixo do limite de latência tolerável (ex: menos de 100ms), o teste de fumaça no GitHub Actions passa (Sinal Verde). Somente neste momento o workflow prossegue para a etapa de liberação de tráfego no orquestrador (Kubernetes ou nuvem serveless). Essa engenharia meticulosa de sequenciamento elimina as quedas de desempenho pós-implantação, oferecendo uma experiência perfeitamente contínua ao consumidor, independentemente do peso massivo da Inteligência Artificial operando nos bastidores.

**O sistema imunológico: rollback automatizado baseado em métricas**

Apesar de toda a governança preventiva (Canary, Shadow, Warmup), a Lei de Murphy atua com rigor na área de dados. Eventos exógenos imprevisíveis podem ocorrer. Uma hora após a implantação de 100% de um modelo de recomendação de notícias, uma quebra no banco de dados parceiro (Third-Party API) começa a retornar texto em nulo para a esteira de inferência. O código do modelo não quebra, mas a predição retorna "Zero" para todos os usuários, esvaziando a página inicial da corporação. Se o fluxo de recuperação for manual, um engenheiro precisa ser acordado, logar na VPN, abrir o GitHub, procurar o commit anterior seguro e forçar a re-implantação. Essa latência humana causa sangramento de reputação e receita a cada segundo de hesitação.

O ápice de um pipeline de Entrega Contínua (CD) maduro não é a sua capacidade de colocar software em produção, mas a sua velocidade autônoma de retirar software nocivo de produção. A infraestrutura deve agir como um sistema imunológico corporativo.

A engenharia abandona as reversões (Rollbacks) puramente procedimentais e cria malhas fechadas de telemetria. As ferramentas de observabilidade (como Datadog, Prometheus ou New Relic) são configuradas com limites rígidos de negócios (SLAs). Se a taxa de erros do novo modelo superar 1%, ou se a distribuição estatística das predições de preços desviar violentamente da média móvel histórica em uma janela de 10 minutos, o alarme não dispara apenas para o bipe do engenheiro; ele dispara um Webhook autenticado diretamente para a API do GitHub Actions.

O repositório possui um workflow oculto configurado com o gatilho genérico `on: repository_dispatch`. Ao receber o pacote JSON de pânico do sistema de monitoramento, este pipeline de Emergência acorda. Ele possui uma lógica imperativa e destrutiva: identifica a tag da versão atual (V2 defeituosa), busca no registro a tag da versão funcional imediatamente anterior (V1 estável), e aciona os comandos da nuvem para forçar o direcionamento de 100% do tráfego de volta para o artefato seguro. O tempo de recuperação (MTTR - Mean Time to Recovery) despenca de 45 minutos para 30 segundos. A equipe é notificada do incidente, mas não acorda para apagar um incêndio; acorda para investigar um log com tranquilidade, sabendo que a automação estancou a hemorragia financeira e blindou a disponibilidade do serviço de forma incondicional.

**Governança e Approvals: OIDC e Proteção de Ambientes**

Em organizações com alto grau de maturidade de segurança, o acesso aos servidores de produção é blindado. Historicamente, para o GitHub Actions realizar a última etapa do CD (aplicar o modelo no cluster), a equipe de plataforma precisava gerar uma credencial de administrador de longa duração na nuvem (AWS/GCP/Azure) e colá-la estaticamente na aba de Secrets do repositório no GitHub. Essa prática cria uma "Chave Mestra" solta na nuvem. Se um atacante escalar privilégios através de uma vulnerabilidade no código ou se um desenvolvedor injetar um script malicioso no `package.json`, a esteira pode ser sequestrada para extrair dados sigilosos ou minerar criptomoedas usando os recursos da empresa.

A etapa final do deploy de ML não pode depender de segredos permanentes (Long-Lived Tokens) e não pode acontecer sem um rastro humano audível, mesmo em processos altamente automatizados.

A equipe de arquitetura implementa duas barreiras inquebráveis diretamente na plataforma do GitHub. A primeira barreira é a adoção de GitHub Environments (Ambientes). A esteira não permite que um Push na main simplesmente injete código em produção. O Job de CD exige que o trânsito pelo ambiente "Production" passe por uma aprovação manual obrigatória. O workflow pausa o processamento e envia uma notificação para o grupo de Arquitetos de MLOps ou Diretores de Risco. Somente após a revisão dos logs de Shadow e Canary, um administrador clica no botão "Aprovar Deploy". Isso garante conformidade com as leis SOX (Sarbanes-Oxley) e normativas bancárias de separação de deveres (Segregation of Duties).

A segunda barreira é a implementação da autenticação sem senhas, através do OIDC (OpenID Connect). Uma vez que o humano aprova o deploy, o GitHub Actions não usa uma senha estática. Ele entra em contato com o provedor de nuvem de forma criptografada e declara: "Eu sou o repositório X, fui autorizado pela pessoa Y e preciso fazer o deploy do Modelo Z agora". A nuvem confia no GitHub e emite um token de acesso efêmero, válido por apenas 15 minutos. O GitHub Actions usa este token para atualizar a imagem no cluster Kubernetes ou publicar a Function serverless e encerra o processo. Se um atacante roubar o log da execução minutos depois, a credencial já estará desintegrada e será inútil.

A entrega contínua torna-se um processo de Fort Knox digital, combinando a agilidade insana da automação com a segurança e a governança que as diretorias executivas exigem para a era da Inteligência Artificial.

**O QUE VOCÊ VIU NESTA AULA?**

Nesta aula, compreendemos como conter o raio de explosão (Blast Radius) de um modelo falho usando estratégias de roteamento fracionado (Canary) e espelhamento (Shadow Deployments). Do ponto de vista de DevSecOps, eliminamos o uso de credenciais de longa duração (Secrets de nuvem) no GitHub, trocando-as pela autenticação federada baseada em identidade (OIDC). Também aprendemos a criar gatilhos de aprovação manual para adequação a leis de conformidade corporativa.

**PALAVRAS-CHAVE:** Canary Release. Shadow Deployment. OIDC. Zero-Trust. CD.

### Código e comandos
Nenhum bloco de código nesta aula. (Elementos técnicos citados inline: gatilho `on: repository_dispatch`; arquivo `package.json`; código HTTP 500; SLA 200ms; tokens efêmeros OIDC de 15 minutos.)

### Ferramentas / serviços citados
GitHub Actions, Kubernetes, AWS SageMaker, Istio Service Mesh, AWS ALB, API Gateway, Feature Store, Datadog, Prometheus, New Relic, TensorFlow, PyTorch, GitHub Environments, OIDC (OpenID Connect), GitHub Secrets, GitHub Issues.

### Aplicabilidade ao Tech Challenge Fase 3
- Canary/Shadow e rollback automatizado (`repository_dispatch`) fornecem o padrão de entrega contínua segura para o classificador NLP de laudos.
- GitHub Environments com aprovação manual atende governança do deploy — encaixa no requisito de CI/CD (15%) com controle humano.
- Warmup + Smoke Tests garantem que a API de inferência esteja pronta (readiness) antes de receber tráfego real.

### REFERÊNCIAS (Aula 6)
- FOWLER, Martin. *Canary release*. 25 jun. 2014. Disponível em: https://martinfowler.com/bliki/CanaryRelease.html. Acesso em: 6 abr. 2026.
- GITHUB. *About security hardening with OpenID Connect*. GitHub Docs, 2026. Disponível em: https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect. Acesso em: 6 abr. 2026.
- GITHUB. *Using OpenID Connect with reusable workflows*. GitHub Docs, 2026. Disponível em: https://docs.github.com/en/actions/how-tos/secure-your-work/security-harden-deployments/oidc-with-reusable-workflows. Acesso em: 24 fev. 2026.

---

## Aula 7 — Boas Práticas de CI/CD em ML (MLOps)
**Arquivo fonte:** `9MLET - Fase 3 - Integração com CICD - Aula 7.pdf` (15 páginas)
**Título na ementa:** Boas Práticas de CI/CD em ML (MLOps)

### Conceitos-chave
- Platform Engineering / caminhos pavimentados para cientistas de dados.
- Versionamento Híbrido (Tríade): Código (Git) + Dados (DVC) + Modelos (MLflow).
- Qualidade de Código Analítico (Mypy, Ruff/Pylint, Complexidade Ciclomática, SQLFluff).
- Reprodutibilidade determinística e fixação de estocasticidade (Random Seeds).
- FinOps: `paths` / `paths-ignore` e execução condicional via `git diff`.
- Pirâmide de Testes de ML assimétrica (Pre-Commit Hooks → PR → main); Reusable Workflows e Custom Actions.

### Conteúdo

**O QUE VEM POR AÍ?**

Quando a sua empresa escala de uma para trinta equipes de IA, o copia-e-cola de arquivos YAML se torna um pesadelo logístico. Atualizar regras de segurança exige dezenas de Pull Requests simultâneos, e pequenas alterações na documentação frequentemente acionam treinamentos gigantescos e caros com GPUs. Nesta aula, assumiremos a postura de Engenharia de Plataforma (Platform Engineering). Você entenderá como criar caminhos pavimentados para os cientistas de dados, centralizando a governança da esteira e otimizando os custos em nuvem (FinOps).

**HANDS ON**

Nesta etapa prática, vamos domar repositórios gigantes (Monorepos). Usaremos paths e paths-ignore no YAML para garantir que o build de ML só aconteça quando o código da rede neural for tocado. Em seguida, criaremos um repositório mestre de Reusable Workflows (workflow_call), e ensinaremos como encapsular lógicas sujas de script em Custom GitHub Actions baseadas em Docker.

**SAIBA MAIS**

**A Consolidação do Paradigma Sociotécnico em IA**

A implementação de Integração e Entrega Contínuas (CI/CD) para Inteligência Artificial não é apenas um desafio de plugar ferramentas como o GitHub Actions a provedores de nuvem; é uma transformação arquitetural e cultural profunda. Quando superamos a fase inicial de conseguir colocar um contêiner em produção, a organização invariavelmente se choca com a barreira da escala. Com múltiplos squads desenvolvendo modelos paralelos de Visão Computacional, Processamento de Linguagem Natural e Sistemas de Recomendação, o caos de processos customizados começa a cobrar seu preço em dívida técnica e fragilidade operacional.

A disciplina de MLOps (Machine Learning Operations) surge não como um produto que se compra, mas como um conjunto rigoroso de boas práticas de engenharia desenhadas para unir três domínios historicamente isolados: a Ciência de Dados (focada em matemática e descoberta), a Engenharia de Software (focada em determinismo e arquitetura) e as Operações de TI (focadas em estabilidade e segurança). O pipeline do GitHub Actions deve atuar como o juiz neutro e implacável que aplica essas melhores práticas de forma contínua e invisível. A seguir, exploramos os cinco pilares arquiteturais que diferenciam operações de IA amadoras de plataformas preditivas de classe mundial.

**A Tríade do Versionamento Híbrido: Código, Dados e Modelos**

Em operações de desenvolvimento de software tradicional, o Git atua como a única Fonte da Verdade (Single Source of Truth). Se a equipe de engenharia precisa auditar o sistema ou reverter para uma versão da semana anterior, basta fazer um `git checkout` para um hash específico e reconstruir o binário. Contudo, em uma equipe de Ciência de Dados, aplicar essa mesma premissa resulta em desastres de conformidade. Um cientista treina um modelo de risco de crédito, submete o código Python impecável para o GitHub e aprova o Pull Request. Seis meses depois, uma auditoria regulatória exige que a empresa reproduza exatamente como aquele modelo tomou uma decisão específica. A equipe tenta rodar o código versionado, mas descobre que o arquivo `.csv` original foi sobrescrito no banco de dados, e o arquivo `.pkl` do modelo foi apagado do armazenamento em nuvem. O código sozinho é uma entidade oca; sem os dados e os pesos originais, a reprodutibilidade é matematicamente impossível.

A governança corporativa exige que o conceito de versionamento seja expandido tridimensionalmente. Um pipeline de MLOps maduro não permite a dissociação temporal entre o código que define a arquitetura, o dado que alimenta o aprendizado e o artefato que contém a "memória" da máquina. Essa amarração deve ser sistêmica, imutável e verificada a cada execução.

Para resolver essa fragmentação, a engenharia de plataforma implementa um padrão rigoroso de Versionamento Híbrido orquestrado pelo GitHub Actions. A esteira integra o Git (para versionamento de texto/código) com ferramentas como DVC - Data Version Control (para controle de fluxo de dados em armazenamentos de objetos como o S3) e Registros de Modelos como o MLflow (para versionamento de parâmetros e binários). Quando a esteira de CI/CD finaliza um treinamento bem-sucedido, o workflow executa um ritual de amarração criptográfica. Ele coleta o hash SHA do commit do Git, o hash MD5 do dataset rastreado pelo DVC e os compila em um único manifesto de metadados. Este manifesto é injetado como variáveis rastreáveis dentro do MLflow.

O resultado é a eliminação do "achismo" operacional. Qualquer desenvolvedor ou auditor no futuro que olhar para o Registro de Modelos da corporação verá uma linha do tempo perfeita. Com apenas um clique, ele consegue extrair a imagem Docker exata, o dataset imutável e o código preciso que formaram aquele modelo específico. A empresa atinge o estado da arte em compliance algorítmico, blindando a operação contra multas regulatórias e perdas de propriedade intelectual.

**Qualidade de Código Analítico e a Erradicação da Dívida Técnica**

A cultura de experimentação em Ciência de Dados é inerentemente iterativa e não linear. Cientistas utilizam Jupyter Notebooks para testar hipóteses rapidamente, copiando, colando e reexecutando blocos de código até encontrarem um sinal estatístico válido. Essa liberdade é excelente para a descoberta, mas quando esse mesmo código transcende para a esteira de produção, ele carrega consigo um fardo massivo de dívida técnica crônica: variáveis globais não declaradas, funções com centenas de linhas sem coesão lógica, ausência total de tipagem estática e importações de bibliotecas que nunca são utilizadas. Quando os engenheiros de MLOps tentam encapsular esse código para gerar a imagem Docker (Aula 5), o tempo gasto refatorando a lógica é exaustivo, e as chances de introduzir "bugs" silenciosos durante a tradução são altíssimas.

A arquitetura do pipeline deve impor uma transição forçada da "ciência artesanal" para a "engenharia de precisão", sem que para isso a equipe de operações precise brigar diariamente com a equipe de dados durante as revisões de Pull Requests. O objetivo é que ferramentas matemáticas de análise de código assumam o papel de "vilão", rejeitando códigos de baixa qualidade antes mesmo que um humano precise revisá-los.

A organização implementa uma barreira de qualidade intransigente no primeiro minuto do workflow do GitHub Actions. Antes de qualquer máquina virtual ser alocada para treinamento, o CI/CD roda uma suíte de Linting Analítico Especializado. O pipeline não verifica apenas espaços em branco usando o Black. Ele invoca o Mypy para forçar tipagem estática profunda — garantindo que uma função que promete retornar um vetor de pontos flutuantes não retorne acidentalmente uma matriz de strings. Ele aplica o Ruff ou Pylint configurados com limites estritos de Complexidade Ciclomática (se uma função de limpeza de dados possui mais de dez ramificações if/else aninhadas, o GitHub Actions quebra e força o desenvolvedor a modularizar a lógica). Adicionalmente, caso haja lógica de feature engineering em SQL, integra-se o SQLFluff para analisar a sintaxe de extração no banco de dados.

A consequência dessa barreira é a elevação orgânica do nível de engenharia de toda a equipe de Inteligência Artificial. Os Pull Requests passam a fluir limpos. A manutenção do código a longo prazo torna-se trivial, pois os engenheiros confiam que qualquer arquivo fundido à branch principal obedece a um contrato semântico rigoroso, reduzindo o tempo de onboarding de novos cientistas de meses para dias.

**Reprodutibilidade Determinística e Fixação de Estocasticidade**

Um dos cenários mais enlouquecedores no ciclo de vida do Machine Learning ocorre quando a equipe tenta reproduzir os resultados de um modelo bem-sucedido. Um Pull Request antigo é reexecutado na esteira do GitHub Actions sem que uma única vírgula do código tenha sido alterada. Contudo, em vez de retornar os originais 95% de F1-Score, o treinamento resulta em 89%. A investigação técnica dessa anomalia geralmente revela um vazamento duplo de estocasticidade (aleatoriedade matemática). Primeiro, as dependências de software não estavam firmemente travadas (o que resolvemos com a conteinerização). Segundo — e mais traiçoeiro —, os algoritmos de inicialização de pesos de redes neurais, as partições dinâmicas de dados (Train/Test Split) e as operações de paralelismo em GPUs introduziram fatores aleatórios subjacentes durante a compilação matemática.

Em sistemas corporativos críticos, a aleatoriedade não justificada é uma falha de design inaceitável. A esteira de CI/CD deve garantir que a construção de modelos preditivos seja um processo inteiramente hermético e determinístico: as mesmas entradas (código, dependências, dados e configurações) devem, em 100% das vezes, produzir a mesma exata matriz de saída bit a bit.

A melhor prática imposta pela engenharia de MLOps é o mapeamento ativo e o controle estrito de todas as sementes aleatórias (Random Seeds) em todo o ecossistema computacional. No GitHub Actions, os workflows são modificados para injetar obrigatoriamente variáveis de ambiente globais estáticas (como `PYTHONHASHSEED=0`). Nos scripts de treinamento em Python, o linter é configurado para proibir a inicialização de modelos sem a declaração explícita de `np.random.seed()`, `torch.manual_seed()`, e instruções para forçar algoritmos determinísticos nas GPUs (`torch.backends.cudnn.deterministic = True`).

Para provar matematicamente que a reprodutibilidade foi alcançada, a equipe cria um Job avançado na esteira de CI chamado de Teste de Identidade Criptográfica. Mensalmente, ou a cada mudança drástica de infraestrutura, o GitHub Actions treina o mesmo modelo duas vezes seguidas em contêineres e Runners físicos distintos, usando o mesmo conjunto restrito de dados amostrais. O pipeline então calcula o hash SHASUM do arquivo `.pt` ou `.onnx` final gerado por cada treinamento. Se os hashes divergirem, o sistema de MLOps notifica os arquitetos informando que houve um vazamento de aleatoriedade no grafo computacional, prevenindo que inconsistências matemáticas poluam a governança de modelos da empresa.

**Governança Financeira (FinOps) e Filtros Cirúrgicos no CI/CD**

A transição de repositórios isolados para a automação total através de Monorepos ou repositórios colaborativos massivos introduz uma armadilha financeira catastrófica nas operações de nuvem. Um desenvolvedor que trabalha na camada de interface do usuário (Frontend Dashboard) ou que atualiza apenas a documentação em um arquivo `README.md` abre um Pull Request. A ferramenta de CI/CD, carecendo de inteligência contextual, enxerga um evento de atualização no repositório e desencadeia o pipeline mestre. Subitamente, instâncias monstruosas com dezenas de GPUs são alocadas via Terraform, puxam terabytes de dados do Data Lake e gastam três horas retreinando uma rede neural pesada de bilhões de parâmetros, apenas para validar uma alteração gramatical em um parágrafo de texto. Ao final do mês, a fatura de computação em nuvem explode sem que um único avanço científico real tenha sido produzido.

O crescimento sustentável de iniciativas de IA em larga escala requer que a eficiência orçamentária seja tratada como uma métrica arquitetural de primeira classe (FinOps em MLOps). O GitHub Actions não pode operar de forma reativa e cega; ele deve ser munido de sensibilidade contextual afiada, operando como um bisturi que dispara cargas de trabalho financeiramente dispendiosas exclusivamente quando o código matemático for ativamente alterado.

A solução arquitetural aplicada é o domínio implacável da Execução Condicional Avançada e Path Filtering (Filtragem de Caminhos) dentro das definições YAML. A equipe de plataforma elabora workflows onde o gatilho (`on: pull_request`) possui declarações de `paths` e `paths-ignore` minuciosamente documentadas. Mais profundamente, para projetos modulares complexos, utilizam-se Custom Actions no início da esteira que realizam a leitura cruzada do `git diff` (a árvore de diferenças entre os commits).

O pipeline avalia a árvore de arquivos e injeta saídas condicionais em formato JSON. O Job seguinte verifica: "O diretório `/models/deep_learning/` ou `/feature_store/` foi modificado?". Se a resposta for não, o GitHub Actions pula (skips) sumariamente todas as etapas de alocação de GPU, treinamento e validação adversária, marcando o CI como "Sucesso (ignorado por irrelevância)". Além de salvar dezenas de milhares de dólares mensalmente, essa inteligência estrutural reduz a fadiga e a espera dos desenvolvedores periféricos, que não precisam mais aguardar o término de tarefas matemáticas irrelevantes para terem seus trabalhos de engenharia ou documentação aprovados para a branch principal.

**Experiência de Desenvolvimento (DX) e a Pirâmide de Testes de ML**

Quando uma organização impõe todas as barreiras de segurança, linting, testes comportamentais (Aula 04), proteção contra ataques adversariais (Aula 06) e análises de conformidade na sua esteira de MLOps, ocorre um efeito colateral organizacional indesejado: o colapso da Developer Experience (Experiência de Desenvolvimento - DX). O pipeline torna-se tão burocrático e pesado que um cientista de dados aguarda de três a cinco horas para saber se o seu Pull Request foi aprovado. Essa latência operacional destrói o conceito central de metodologias ágeis (o Feedback Loop rápido). Desenvolvedores perdem o contexto mental de seus problemas matemáticos, começam a abrir Pull Requests massivos (com milhares de linhas de alteração de uma só vez para evitar passar pelo processo) e a resistência interna contra a adoção de CI/CD cresce de forma insustentável.

O pináculo das boas práticas de MLOps corporativo não é criar um pipeline impenetrável e demorado, mas sim desenhar uma Pirâmide de Testes Assimétrica e Inteligente, equilibrando com perfeição a velocidade de feedback diária com a robustez de verificação das releases de longo prazo. A arquitetura da esteira deve ser estratificada para reprovar erros estúpidos em milissegundos e demorar apenas para avaliar falhas estatísticas complexas.

A equipe de plataforma orquestra o GitHub Actions dividindo o pipeline em três engrenagens operacionais distintas. A primeira engrenagem é o Fail-Fast (Falha Rápida), frequentemente antecipada para a própria máquina do cientista via Pre-Commit Hooks. Trata-se de uma suíte de verificações (Linting, formatação e verificação de senhas em hardcode) que o desenvolvedor é obrigado a passar antes mesmo do código subir para a nuvem. Se algo falha, ele descobre em três segundos locais, não consumindo minutos gratuitos do GitHub Actions.

A segunda engrenagem ocorre no momento do Pull Request. Ao chegar na nuvem, o CI/CD roda testes unitários matemáticos, validação de esquemas de dados e o Dummy Training (Aula 02) com uso intenso de Cache Docker (Aula 05). Essa etapa roda em CPUs baratas e fornece feedback total sobre a integridade arquitetural em um tempo alvo máximo de 10 minutos. O cientista sabe quase imediatamente se quebrou o grafo de tensores ou se o código é estável para integração.

A terceira engrenagem é reservada estritamente para a branch principal (main) ou aprovação de Release Tag. Somente aqui, quando dezenas de pequenas alterações ágeis foram consolidadas e aprovadas semanticamente, o GitHub Actions aciona a infraestrutura pesada (Instâncias Efêmeras de GPU via IaC, Aula 05) para rodar o Treinamento Completo, simular os ataques de cibersegurança e atualizar a orquestração em produção.

Essa separação de interesses transforma o ecossistema corporativo. O cientista de dados usufrui da agilidade contínua de uma startup para experimentar rapidamente e validar suas hipóteses estruturais, enquanto a empresa desfruta da blindagem institucional, segurança e conformidade inegociáveis exigidas por uma corporação líder de mercado operando Inteligência Artificial no estado da arte.

**O QUE VOCÊ VIU NESTA AULA?**

Nesta aula, elevamos a discussão de um único projeto para a gestão corporativa de MLOps. Aprendemos técnicas ativas de governança financeira (FinOps), bloqueando execuções de esteira desnecessárias em Monorepos. Também vimos como erradicar a síndrome do "YAML Hell" extraindo a inteligência da esteira para fluxos de trabalho reutilizáveis e Actions proprietárias, permitindo que a equipe de plataforma atualize as regras de CI/CD para dezenas de times de uma só vez.

**PALAVRAS-CHAVE:** Platform Engineering. Reusable Workflows. FinOps. Monorepo. Custom Actions.

### Código e comandos
Nenhum bloco de código nesta aula. (Elementos técnicos citados inline: comando `git checkout`; extensões `.csv` / `.pkl` / `.pt` / `.onnx`; variável de ambiente `PYTHONHASHSEED=0`; funções `np.random.seed()`, `torch.manual_seed()`, `torch.backends.cudnn.deterministic = True`; arquivo `README.md`; gatilho `on: pull_request` com `paths` / `paths-ignore`; comando `git diff`; diretórios `/models/deep_learning/` e `/feature_store/`; diretiva `workflow_call`.)

### Ferramentas / serviços citados
GitHub Actions, Git, DVC (Data Version Control), S3, MLflow, Docker, Jupyter Notebooks, Black, Mypy, Ruff, Pylint, SQLFluff, PyTorch/NumPy (seeds), Terraform, Data Lake, Pre-Commit Hooks, Reusable Workflows (`workflow_call`), Custom Actions.

### Aplicabilidade ao Tech Challenge Fase 3
- Versionamento Híbrido (Git + DVC + MLflow) e reprodutibilidade determinística (seeds) reforçam a rastreabilidade do classificador de laudos.
- Path Filtering (`paths` / `paths-ignore`) evita builds/treino desnecessários no repositório do TC, otimizando os minutos de GitHub Actions.
- Pirâmide de Testes assimétrica (Pre-Commit → PR → main) estrutura os testes automatizados e o pipeline CI/CD (15%) do projeto.

### REFERÊNCIAS (Aula 7)
- GITHUB. *Events that trigger workflows*. GitHub Docs, 2026. Disponível em: https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows. Acesso em: 6 abr. 2026.
- GITHUB. *Reusing workflows*. GitHub Docs, 2026. Disponível em: https://docs.github.com/en/actions/using-workflows/reusing-workflows. Acesso em: 6 abr. 2026.
- GITHUB. *Workflow syntax for GitHub Actions: paths*. GitHub Docs, 2026. Disponível em: https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#onpushpull_requestpull_request_targetpathspaths-ignore. Acesso em: 6 abr. 2026.

---

## Aula 8 — Aprendizado Contínuo e Monitoramento de Modelos
**Arquivo fonte:** `9MLET - Fase 3 - Integração com CICD - Aula 8.pdf` (15 páginas)
**Título na ementa:** Aprendizado Contínuo e Monitoramento de Modelos

### Conceitos-chave
- Entropia algorítmica: Data Drift e Concept Drift; degradação silenciosa.
- Falha das métricas de infraestrutura tradicionais; Observabilidade de IA (KL / Wasserstein).
- Telemetria assíncrona (Kafka/Kinesis) e ponte com GitHub Actions via `repository_dispatch`.
- Continuous Training (CT) como sistema imunológico (janela móvel, treino reativo, reavaliação).
- Champion vs. Challenger (evitar Catastrophic Forgetting).
- Human-in-the-Loop: Pull Request autônomo aberto pelo `github-actions[bot]`.

### Conteúdo

**O QUE VEM POR AÍ?**

Diferente de sistemas web normais, quando um modelo de Machine Learning vai para a produção, ele imediatamente começa a "apodrecer". Mudanças no comportamento dos clientes e na economia fazem com que o algoritmo perca acurácia ao longo do tempo, um fenômeno conhecido como Drift.

Nesta última aula, fecharemos o ciclo (Closed-Loop MLOps). Vamos abandonar o retreinamento manual caótico. Você entenderá como conectar as métricas do ambiente de produção de volta ao GitHub Actions para que o sistema reaja sozinho às degradações da vida real.

**HANDS ON**

Nesta etapa prática final, vamos fazer o nosso pipeline ser acordado por robôs, não por humanos. Configuraremos o evento repository_dispatch. Simularemos um sistema de monitoramento (como Evidently AI) detectando Data Drift e enviando um Webhook JSON para o GitHub. A esteira vai acordar sozinha, executar o pipeline de Continuous Training (CT) e, usando scripts, abrirá automaticamente um Pull Request sugerindo a atualização do modelo para a diretoria.

**SAIBA MAIS**

**A Entropia Algorítmica e o Fechamento do Ciclo de Vida**

Ao longo das disciplinas anteriores, construímos uma formidável esteira de Integração e Entrega Contínuas (CI/CD). Padronizamos o código, automatizamos os testes matemáticos, conteinerizamos o artefato, blindamos a segurança da cadeia de suprimentos e orquestramos a implantação progressiva. Sob a ótica da engenharia de software tradicional, o sistema estaria "pronto". Contudo, a Inteligência Artificial obedece a leis de entropia distintas. Um binário compilado de um aplicativo de calculadora funcionará exatamente da mesma forma hoje e daqui a cinquenta anos. Um modelo preditivo de ponta, por sua vez, é um ativo em constante estado de decaimento.

Modelos de IA são aproximações estatísticas de um momento específico no tempo. Assim que o modelo é exposto à produção, o comportamento do usuário final, a macroeconomia e as tendências de mercado começam a divergir lentamente dos dados históricos utilizados no treinamento. Essa divergência degrada a acurácia da predição dia após dia. A operação de MLOps de nível 3 (maturidade máxima) reconhece que a implantação (Deploy) não é a linha de chegada, mas sim o início de um circuito fechado (Closed-Loop System). O GitHub Actions deve evoluir de uma ferramenta de "entrega" para o motor principal do Continuous Training (CT) — orquestrando não apenas a criação inicial do modelo, mas a sua sobrevivência a longo prazo através do monitoramento reativo e do retreinamento autônomo.

**A falha das métricas tradicionais e a degradação silenciosa**

Em organizações que estão dando os primeiros passos na transição de seus modelos de IA para a produção, é comum que a responsabilidade pelo monitoramento seja transferida para a equipe de Engenharia de Confiabilidade (SRE) tradicional. Esses engenheiros conectam os contêineres de Machine Learning às suas ferramentas de telemetria padrão (como Datadog, Prometheus ou Zabbix) e configuram alertas baseados em infraestrutura. Durante meses, os painéis exibem um cenário perfeito: uso de CPU estável em 40%, consumo de memória RAM sem vazamentos, latência de resposta da API na casa dos 150 milissegundos e 100% de códigos HTTP 200 (Sucesso). A equipe técnica comemora a estabilidade, enquanto a diretoria de negócios entra em pânico pois a taxa de conversão de vendas ou a detecção de fraudes despencou violentamente.

Essa assimetria de percepção ocorre porque sistemas de IA sofrem de "falhas silenciosas". O modelo continua respondendo matematicamente, mas suas respostas perdem o valor semântico. Essa degradação ocorre por dois vetores principais:

1. **Data Drift (Desvio de Dados):** A distribuição estatística das variáveis de entrada muda. Se um modelo de aprovação de crédito foi treinado em uma época de juros baixos, e a economia subitamente entra em hiperinflação, os salários e o poder de compra dos clientes mudam radicalmente. O modelo recebe valores que ele nunca viu e começa a extrapolar cegamente.
2. **Concept Drift (Desvio de Conceito):** O mapeamento entre a entrada e a saída muda. Em um modelo de cibersegurança que detecta padrões de ataque de hackers, o adversário percebe que foi bloqueado e inventa uma nova técnica. O comportamento do usuário mudou de forma invisível.

A arquitetura de monitoramento precisa ser reconstruída. A plataforma não deve monitorar apenas os recursos computacionais do servidor; ela deve monitorar a matemática da predição. A infraestrutura passa a exigir Observabilidade de Inteligência Artificial, um sistema capaz de calcular distâncias estatísticas (como Divergência de Kullback-Leibler ou Distância de Wasserstein) em tempo real, comparando as predições de hoje com o dataset assinado na esteira de CI/CD seis meses atrás.

**Telemetria Assíncrona e a Ponte com o GitHub Actions**

A implementação da Observabilidade de IA apresenta um desafio severo de performance. Calcular distribuições estatísticas complexas em matrizes multidimensionais a cada requisição de um usuário arruinaria completamente a latência da API de inferência (o modelo demoraria segundos para responder). O monitoramento precisa ser profundo, porém estritamente assíncrono.

Para orquestrar essa visibilidade sem penalizar o usuário final, a equipe de plataforma adota uma arquitetura orientada a eventos. O contêiner em produção, orquestrado previamente pelo GitHub Actions, é modificado para atuar como um emissor silencioso. Quando ele gera uma predição, ele responde imediatamente ao cliente, mas despacha uma cópia do payload de entrada, a predição calculada e a probabilidade de confiança (Confidence Score) para um barramento de mensagens (como o Apache Kafka ou AWS Kinesis). Na outra ponta desse barramento, operam ferramentas especializadas em qualidade de modelos (como Evidently AI, NannyML ou Arize). Essas ferramentas agregam as predições em janelas temporais (ex: a cada hora ou a cada dia) e calculam o Drift estatístico daquele lote de dados.

O verdadeiro salto de maturidade arquitetural ocorre na integração dessas ferramentas com a esteira corporativa. Monitoramento passivo — aquele que apenas plota gráficos vermelhos em um painel que ninguém olha — não gera valor em larga escala. A observabilidade só tem poder quando atrelada à remediação automatizada. As plataformas de monitoramento de IA são configuradas com limites de degradação algorítmica. Se o Data Drift da variável "Renda" ultrapassar 15% em uma janela de 48 horas, o sistema não envia apenas um e-mail para o cientista de dados. O sistema aciona um Webhook Seguro apontando diretamente para a API do GitHub. Ele envia um evento do tipo `repository_dispatch` (Evento Externo) contendo o nome do modelo afetado e o metadado da falha estatística. Esta é a fagulha que acende o sistema imunológico da corporação, transicionando o problema da área de "Monitoramento" para a área de "Integração Contínua".

**Orquestração do Continuous Training (CT) como Sistema Imunológico**

Quando o Webhook atinge o repositório no GitHub, o workflow de Continuous Training (Retreinamento Contínuo) é acordado de seu estado de dormência. Este não é o pipeline tradicional acionado quando um humano altera um código (`on: push`); este é um robô de manutenção operando sob a premissa de que o código está perfeito, mas o algoritmo contido nele caducou e precisa de atualização urgente a partir do mundo exterior.

O GitHub Actions assume o controle de um workflow complexo de remediação que não envolve intervenção humana imediata. O processo é estruturado em etapas sequenciais automatizadas:

1. **Ingestão da Janela Móvel:** O Job inicial conecta-se ao Feature Store ou ao Data Lake da corporação. Ele não busca todos os dados históricos, mas simulação uma janela móvel (ex: extrai todos os dados aprovados e rotulados dos últimos 30 dias que causaram o alerta de drift).
2. **O Treinamento Reativo:** O GitHub Actions aloca a instância efêmera com GPU (usando Infraestrutura como Código, como visto na Aula 05) e inicia o processo de `model.fit()`. Ele carrega a arquitetura imutável da rede neural previamente aprovada, mas a submete à nova realidade de dados extraída no passo anterior.
3. **Avaliação Pós-Remediação:** O novo artefato gerado não ganha passe livre para a produção. O pipeline submete o modelo recém-treinado à exata mesma bateria de testes matemáticos, comportamentais e contratuais estabelecida na Aula 04.

> [NOTA — não é conteúdo FIAP]: no item 1 o PDF apresenta a frase "mas simulação uma janela móvel"; provável erro de digitação da fonte para "mas simula uma janela móvel". Transcrito exatamente como consta.

O impacto sistêmico dessa automação é o fim do "combate a incêndios" reativo. Antes, a degradação de um modelo resultava em reuniões de crise e paralisação de projetos criativos para que a equipe sênior refizesse o trabalho artesanal de retreinar o algoritmo. Com o GitHub Actions orquestrando o CT, a manutenção torna-se inerente à plataforma. O(a) cientista de dados foca na criação de novas arquiteturas matemáticas, sabendo que a esteira cuidará de manter os modelos legados "vivos", saudáveis e sintonizados com o mercado automaticamente.

**O Duelo Automatizado (Champion vs. Challenger)**

A execução autônoma de cargas de trabalho pesadas acarreta um risco brutal: o "Esquecimento Catastrófico" (Catastrophic Forgetting) ou o retreinamento cego. O monitoramento alertou sobre um desvio nos dados, o GitHub Actions retreinou o modelo com os dados recentes, mas e se esse desvio for um erro transitório? Por exemplo, um sensor da fábrica quebrou, gerando uma semana de dados corrompidos. Se o sistema engolir esses dados corrompidos, retreinar o algoritmo e empurrá-lo para a produção automaticamente, a empresa substituirá um modelo ligeiramente desatualizado por um modelo estatisticamente mutilado e irracional. O remédio automático torna-se pior que a doença original.

A arquitetura do Aprendizado Contínuo precisa integrar um mecanismo de freios e contrapesos rigoroso para garantir que as atualizações algorítmicas ocorram na direção do ganho monetário e científico. O GitHub Actions não pode assumir que o novo modelo é inerentemente superior apenas porque é mais recente.

A solução é a orquestração do padrão Champion vs. Challenger (Campeão vs. Desafiante). Quando o retreinamento reativo (Challenger) é concluído, o pipeline do GitHub Actions interrompe a marcha rumo à implantação. Ele faz o download dos pesos do modelo atual de produção (o Champion) a partir do Registro de Modelos. A esteira levanta dois contêineres na máquina virtual e aplica um dataset de validação (completamente cego para ambos) sobre os dois modelos simultaneamente. O pipeline atua como o árbitro. Ele avalia métricas combinadas (ex: se o F1-Score do Challenger for estatisticamente superior ao Champion, e a latência de inferência permanecer dentro do orçamento de tempo). Se o modelo retreinado não provar matematicamente a sua superioridade com uma margem de segurança predefinida (para evitar substituições por ganhos pífios de 0,01%), o GitHub Actions reprova o Challenger, destrói o artefato recém-criado, registra o evento na ferramenta de observabilidade informando "Retreinamento abortado por ineficácia" e encerra o fluxo com sucesso (já que ele bloqueou um possível desastre). O sistema aprende a dizer "não" para os próprios dados se eles não trouxerem valor.

**Human-in-the-Loop e a Governança do Pull Request Autônomo**

Em corporações sujeitas a escrutínio regulatório intenso — como companhias aéreas, indústrias farmacêuticas e bancos —, a ideia de um robô atualizar a lógica de decisão de um sistema crítico em produção na madrugada de um domingo, sem nenhuma intervenção humana, é terminantemente proibida por Compliance (Leis de governança e SOX). A esteira não pode simplesmente empurrar a nova imagem Docker no cluster e redirecionar o tráfego, por mais rigorosos que tenham sido os testes matemáticos. No entanto, se exigirmos que um cientista execute o retreinamento manualmente sempre que o alerta tocar, anularemos os milhões investidos na plataforma de MLOps.

A engenharia de ponta reconcilia essa dicotomia através do paradigma Human-in-the-Loop (Humano no Ciclo), utilizando o GitHub Actions para orquestrar o trabalho pesado, mas deixando o veredito legal para o especialista humano. Em vez de acionar a etapa final de Continuous Deployment (CD) após o Duelo do Champion vs. Challenger, a esteira muda a sua rotação para atuar como um assistente analítico.

O GitHub Actions utiliza a funcionalidade avançada de `actions/github-script` ou chamadas à API do GitHub para manipular o próprio repositório de código de forma autônoma. Quando o modelo Challenger prova ser superior, o robô do GitHub abre um Pull Request Automático em nome da conta de serviço da esteira (ex: `github-actions[bot]`). A descrição deste Pull Request não é vazia; ela contém um relatório Markdown exaustivo autogerado, exibindo gráficos de divergência de dados, a matriz de confusão comparando o modelo antigo e o novo, a comprovação de ausência de vieses de minorias e a estimativa de ganho financeiro se o modelo for aprovado. Em anexo, o robô marca (tags) obrigatoriamente os diretores de Risco e os líderes de Ciência de Dados corporativa exigindo review.

Quando os especialistas chegam ao escritório na segunda-feira, eles não precisam gastar dias provisionando máquinas, buscando os dados da falha e testando o modelo. O trabalho braçal foi 100% resolvido. Eles encontram o prato pronto no GitHub. O cientista revisa a argumentação matemática construída pela esteira no painel do Pull Request. Ao clicar em "Approve" e "Merge", esse clique humano satisfaz instantaneamente a auditoria legal corporativa. O merge aciona o pipeline final de Deploy, e a versão superior substitui a degradada em produção. A organização atinge o nirvana tecnológico: velocidade de máquina para processamento matemático combinada com a prudência da cognição humana para responsabilidade legal, consagrando o verdadeiro valor do MLOps corporativo.

**O QUE VOCÊ VIU NESTA AULA?**

Nesta aula, compreendemos a entropia algorítmica causada por Data Drift e Concept Drift. Aprendemos que o fim do CD é apenas o início do CT (Continuous Training). Vimos como orquestrar gatilhos orientados a eventos, permitindo que falhas detectadas na produção disparem Webhooks de remediação automatizada. Por fim, exploramos o padrão "Human-in-the-loop", onde o robô do GitHub faz todo o trabalho de retreinamento e abre o Pull Request apenas para a auditoria final humana.

**PALAVRAS-CHAVE:** Continuous Training. Data Drift. Webhooks. Human-in-the-loop. Event-Driven.

### Código e comandos
Nenhum bloco de código nesta aula. (Elementos técnicos citados inline: gatilhos `on: push` e evento `repository_dispatch`; chamada `model.fit()`; ação `actions/github-script`; conta de serviço `github-actions[bot]`.)

### Ferramentas / serviços citados
GitHub Actions, Evidently AI, NannyML, Arize, Datadog, Prometheus, Zabbix, Apache Kafka, AWS Kinesis, Feature Store, Data Lake, MLflow (Registro de Modelos), Docker, `actions/github-script`, Infraestrutura como Código (IaC).

### Aplicabilidade ao Tech Challenge Fase 3
- Monitoramento de Data/Concept Drift + gatilho `repository_dispatch` fecha o ciclo de aprendizado contínuo do classificador de laudos médicos.
- Champion vs. Challenger dá critério objetivo para promover novas versões do modelo NLP com segurança.
- PR autônomo do `github-actions[bot]` com relatório Markdown integra retreinamento e governança dentro do fluxo CI/CD (15%) exigido.

### REFERÊNCIAS (Aula 8)
- EVIDENTLY AI. *Machine learning monitoring: data and concept drift*. Disponível em: https://evidentlyai.com/blog/machine-learning-monitoring-data-and-concept-drift. Acesso em: 6 abr. 2026.
- GITHUB. *Events that trigger workflows: repository_dispatch*. GitHub Docs, 2026. Disponível em: https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#repository_dispatch. Acesso em: 6 abr. 2026.
- GITHUB. *Workflow commands for GitHub Actions*. GitHub Docs, 2026. Disponível em: https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-commands. Acesso em: 6 abr. 2026.

---
