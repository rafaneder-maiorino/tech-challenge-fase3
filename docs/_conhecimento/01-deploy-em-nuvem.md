# Deploy em Nuvem
> Fonte: PDFs FIAP Pós Tech MLET — Fase 3 (Cloud and MLOps)
> Aulas extraídas: 6 de 6
> Data de extração: 2026-07-23

## Sumário
- [Aula 1 — Visão Geral do Deploy de ML em Nuvem](#aula-1--visão-geral-do-deploy-de-ml-em-nuvem)
- [Aula 2 — Comportamento Computacional dos Modelos e Estratégias de Deploy](#aula-2--comportamento-computacional-dos-modelos-e-estratégias-de-deploy)
- [Aula 3 — Deploy de ML na AWS: ECR, EC2, Lambda, AWS Batch e SageMaker](#aula-3--deploy-de-ml-na-aws-ecr-ec2-lambda-aws-batch-e-sagemaker)
- [Aula 4 — Deploy de ML na Azure: ACR, VM, Container Apps, Jobs, Azure ML](#aula-4--deploy-de-ml-na-azure-acr-vm-container-apps-jobs-azure-ml)
- [Aula 5 — Deploy de ML na Google: Artifact Registry, Compute Engine VM, Cloud Run, Cloud Run Jobs e Vertex AI](#aula-5--deploy-de-ml-na-google-artifact-registry-compute-engine-vm-cloud-run-cloud-run-jobs-e-vertex-ai)
- [Aula 6 — Boas Práticas de FinOps (Custos) e Segurança](#aula-6--boas-práticas-de-finops-custos-e-segurança)

---

## Aula 1 — Visão Geral do Deploy de ML em Nuvem
**Arquivo fonte:** `Aula 01.pdf` (15 páginas)
**Título na ementa:** Visão Geral do Deploy de ML em Nuvem (título inferido da capa: "AULA 01 - VISÃO GERAL DO DEPLOY DE ML EM NUVEM", coincide com a ementa)

### Conceitos-chave
- Deploy de ML como um **problema de sistemas**, não apenas como etapa final do treinamento.
- Os três principais padrões de deploy em produção: **batch**, **tempo real** e **serverless**.
- Trade-offs relacionados a **latência, custo, escalabilidade e complexidade de operação**.
- A escolha entre as três abordagens é uma **decisão de arquitetura, não de tecnologia**.
- Coexistência dos três padrões em **arquiteturas híbridas**.

### Conteúdo

#### O que vem por aí?
Nesta aula, você terá uma visão estruturada sobre o deploy de modelos de Machine Learning em ambientes de nuvem, entendendo o deploy como um problema de sistemas e não apenas como uma etapa final do treinamento. Serão apresentados os três principais padrões de deploy utilizados em produção: batch, tempo real e serverless, com foco em suas características técnicas, limitações operacionais e trade-offs relacionados a latência, custo, escalabilidade e complexidade de operação.

Ao longo da aula, esses padrões serão analisados de forma comparativa, mostrando como requisitos como urgência da resposta, volume de dados, frequência de execução e tolerância a falhas influenciam diretamente a escolha da arquitetura de deploy. Também será discutido como esses modelos podem coexistir em arquiteturas híbridas, refletindo práticas comuns em sistemas reais de Machine Learning em produção.

Por fim, a aula conecta os conceitos teóricos a cenários práticos de mercado, preparando o terreno para a implementação de um projeto simples de inferência em tempo real. Essa base conceitual é essencial para compreender as decisões técnicas que serão tomadas nas próximas aulas, quando entraremos em arquiteturas de nuvem, desenvolvimento prático e estratégias de deploy operacional de modelos de Machine Learning.

#### Hands On
Nesta etapa prática, será apresentado um projeto introdutório de deploy de um modelo de Machine Learning em tempo real, com o objetivo de demonstrar como um modelo treinado pode ser disponibilizado como um serviço acessível externamente. A proposta é consolidar os conceitos discutidos anteriormente, conectando teoria e prática por meio de uma arquitetura simples, porém representativa de um cenário real de produção.

O projeto utiliza um modelo de classificação treinado sobre o dataset MNIST, exposto por meio de uma API desenvolvida com FastAPI e empacotada em um container Docker. Essa API será responsável por receber dados de entrada, executar a inferência do modelo e retornar o resultado da predição. O uso de containers permite padronizar o ambiente de execução, facilitando a reprodutibilidade e reduzindo problemas relacionados a dependências e configuração de infraestrutura.

Para simular o acesso externo a esse serviço, será utilizado um túnel de rede que permite expor a aplicação local para a internet, reproduzindo o comportamento de um endpoint em nuvem. Essa abordagem possibilita observar, de forma prática, os desafios e cuidados envolvidos no deploy de modelos em tempo real, como exposição de endpoints, comunicação entre sistemas e preparação do modelo para consumo por aplicações externas. Link para os arquivos utilizados em aula: https://github.com/FIAP/MLET_Deploy_EM_NUVEM/tree/main/Aula_1.

#### Saiba Mais

**Deploy batch.** O deploy batch é uma das formas mais consolidadas de disponibilização de modelos em produção. O modelo é aplicado sobre conjuntos de dados previamente armazenados, processando grandes volumes de uma só vez ou em execuções programadas. O foco não está na resposta imediata, mas na eficiência do processamento e na geração de resultados que serão consumidos posteriormente.

Arquiteturalmente, o deploy batch se encaixa em pipelines de dados existentes, com etapas bem definidas de ingestão e inferência. Isso permite maior controle sobre o fluxo, facilitando auditorias e reprocessamentos. Como não há requisições individuais em tempo real, a infraestrutura pode ser provisionada apenas durante a execução dos jobs, reduzindo o custo operacional.

A previsibilidade é uma característica central: execuções ocorrem em janelas definidas, permitindo planejar recursos e evitar picos inesperados. A ausência de requisitos rígidos de latência também torna essa abordagem mais tolerante a falhas pontuais, desde que mecanismos de retry estejam bem definidos. Pensando em evolução do modelo, atualizações são realizadas de forma controlada, substituindo o artefato na próxima execução, sem impactar sistemas em uso contínuo.

Em termos de manutenção e evolução do modelo, o deploy batch permite a atualização de versões de forma controlada, substituindo o modelo utilizado na próxima execução sem impactar sistemas em uso contínuo. Essa característica reduz riscos associados a mudanças em produção e facilita a realização de testes comparativos entre versões do modelo, como análises offline de desempenho ou validação de métricas em conjuntos históricos de dados.

Outro aspecto relevante do deploy batch é que, como as execuções são discretas e bem delimitadas no tempo, métricas de desempenho, consumo de recursos e qualidade das previsões podem ser coletadas e analisadas de forma estruturada após cada processamento, facilitando a identificação de desvios, degradação do modelo ou problemas nos dados de entrada, permitindo intervenções antes que os resultados sejam consumidos em larga escala.

A principal limitação do batch é a latência elevada, que o torna inadequado para decisões imediatas. A defasagem entre coleta e inferência pode reduzir a relevância dos resultados em contextos dinâmicos.

**Deploy em tempo real.** Já o deploy em tempo real exige que o modelo esteja continuamente disponível para receber dados e retornar previsões de forma imediata. Geralmente materializado como APIs, cada requisição de inferência é tratada individualmente, tornando a latência um requisito crítico. Isso impõe restrições rigorosas sobre a infraestrutura, que precisa ser dimensionada para atender a picos de carga e manter níveis estáveis de desempenho.

Diferentemente do batch, a infraestrutura precisa estar ativa permanentemente, o que eleva os custos operacionais. Mecanismos de escalabilidade automática tornam-se essenciais, assim como estratégias avançadas de tolerância a falhas e balanceamento de carga — já que falhas no serviço podem interromper operações críticas.

Atualizações de versão exigem cuidados especiais: técnicas como versionamento de endpoints, deploy gradual e testes em paralelo evitam interrupções.

No que diz respeito à manutenção e evolução do modelo, o deploy em tempo real necessita que atualizações de versão sejam realizadas de forma controlada para evitar interrupções ou inconsistências nas respostas, utilizando técnicas como versionamento de endpoints, deploy gradual e testes em paralelo.

O monitoramento contínuo de latência, taxa de erro e qualidade das previsões é indispensável, pois qualquer degradação afeta diretamente o sistema. Essa abordagem é indispensável onde a decisão depende da imediaticidade da resposta, mas vem acompanhada de custo elevado e maior complexidade operacional.

Apesar de sua complexidade, o deploy em tempo real é indispensável em cenários onde a relevância da decisão depende da imediaticidade da resposta. Essa abordagem permite a construção de sistemas interativos e responsivos, nos quais o modelo de Machine Learning atua como um componente ativo da lógica de negócio. No entanto, essa capacidade vem acompanhada de um custo maior e de um aumento significativo na complexidade operacional, o que reforça a importância de uma análise criteriosa antes de sua adoção.

**Deploy serverless.** Também temos o deploy serverless, que elimina o gerenciamento direto de servidores. O modelo é executado sob demanda, acionado por eventos, enquanto provisionamento e escalabilidade são delegados ao provedor de nuvem. Ele costuma ser implementado por meio de funções ou serviços gerenciados que encapsulam a lógica de inferência do modelo, que são instanciados apenas quando necessários e desativados em seguida, tornando o modelo de custo mais eficiente para demandas intermitentes.

A escalabilidade automática é a principal vantagem: o provedor instancia múltiplas execuções simultâneas conforme o volume cresce, sem configuração manual. No entanto, funções serverless possuem restrições de tempo de execução, memória e acesso a recursos especializados, o que pode inviabilizar modelos mais complexos.

O cold start é outro ponto de atenção: quando a função não é invocada com frequência, a inicialização do ambiente impacta o tempo de resposta. A observabilidade depende fortemente das ferramentas do provedor, com visibilidade limitada sobre o ambiente subjacente.

O versionamento e a atualização de modelos em ambientes serverless tendem a ser mais simples do que em arquiteturas de tempo real tradicionais, pois as funções são independentes e efêmeras, permitindo que novas versões sejam implantadas rapidamente, reduzindo o risco de conflitos. Ainda assim, é necessário cuidado na gestão de dependências e na compatibilidade entre versões do modelo e do ambiente de execução fornecido pelo provedor.

Analisando monitoramento e observabilidade, o deploy serverless depende fortemente das ferramentas disponibilizadas pela plataforma de nuvem, mas métricas de execução, consumo de recursos e falhas são geralmente coletadas automaticamente. Com isso, a compreensão das capacidades e restrições do provedor escolhido e a adoção de boas práticas de instrumentação e logging dentro das funções são essenciais.

O serverless é especialmente eficaz para workloads orientados a eventos, prototipagem e demanda variável, mas deve ser avaliado com atenção às limitações de runtime e variabilidade de latência.

**Comparativo: batch × tempo real × serverless.** A escolha entre as três abordagens é uma decisão de arquitetura, não de tecnologia. Cada uma define como o modelo executa, como os dados circulam e quais custos e riscos são assumidos.

| Eixo | Batch | Tempo Real | Serverless |
| --- | --- | --- | --- |
| Latência | Alta, tolerada | Baixa, determinística | Variável (cold start) |
| Custo | Menor, sem ociosidade | Maior, infraestrutura permanente | Eficiente em cargas intermitentes |
| Escalabilidade | Planejada, orientada a throughput | Automática, orientada a concorrência | Automática, mas com limites do provedor |
| Complexidade operacional | Baixa | Alta | Média (migra para o provedor) |
| Tolerância a falhas | Alta — reexecução do job | Baixa — impacto imediato no negócio | Média — retries automáticos |
| Atualização do modelo | Controlada, rollback simples | Exige deploy gradual e versionamento | Rápida, mas atenção a dependências |

Tabela 1 – Tabela comparativa entre tipos de deploy
Fonte: Elaborado pelo autor (2025)

> [NOTA — não é conteúdo FIAP]: no dump, o texto da Tabela 1 aparece fragmentado devido à quebra de colunas na extração do PDF (pág. 8). A reconstrução acima segue a correspondência lógica entre células. Verifique com o material original em caso de dúvida sobre as células "Complexidade operacional / Média (migra para o provedor)".

Em ambientes reais, as três estratégias coexistem: batch é geralmente a melhor escolha quando o objetivo é throughput, custo controlado e previsibilidade, aceitando latência. Tempo real é a escolha adequada quando a decisão precisa ocorrer imediatamente e existe justificativa para sustentar custos e complexidade de alta disponibilidade. Serverless atende bem cenários sob demanda, com variação de carga e necessidade de reduzir operação de infraestrutura, desde que as limitações de runtime e a variabilidade de latência sejam aceitáveis.

Com isso, um mesmo produto pode ter batch para geração de insights e relatórios, tempo real para decisões críticas e serverless para tarefas eventuais orientadas a eventos.

A pergunta correta não é "qual é o melhor tipo de deploy", mas "qual atende aos requisitos do componente específico do sistema de ML em produção".

#### Mercado, Cases e Tendências
O deploy de modelos de Machine Learning em nuvem deixou de ser uma preocupação exclusiva de times de dados e passou a ocupar um papel central nas estratégias de tecnologia das empresas. À medida que organizações avançam no uso de ML em produtos e processos internos, a forma como esses modelos são operacionalizados se torna um fator determinante para escalabilidade, custo e confiabilidade das soluções. No mercado, observa-se uma clara maturação na adoção de diferentes padrões de deploy, alinhados a necessidades específicas de negócio e a restrições técnicas cada vez mais bem compreendidas.

Em empresas com forte orientação analítica, o deploy batch continua sendo amplamente utilizado, especialmente em contextos de Business Intelligence, planejamento estratégico e análise preditiva offline. Grandes organizações dos setores financeiro, varejo e telecomunicações utilizam pipelines batch para gerar previsões periódicas, segmentações e indicadores que alimentam dashboards e sistemas de apoio à decisão. Essa abordagem permanece relevante devido à sua previsibilidade, facilidade de auditoria e integração natural com plataformas de dados já consolidadas, como data warehouses e data lakes.

Por outro lado, o crescimento de produtos digitais interativos impulsionou fortemente o uso de deploy em tempo real. Plataformas de e-commerce, serviços financeiros digitais, sistemas de recomendação e soluções de detecção de fraude dependem cada vez mais de inferência imediata para manter competitividade. Nesse cenário, observa-se uma tendência de arquiteturas baseadas em microsserviços e APIs especializadas para inferência, frequentemente combinadas com mecanismos avançados de monitoramento e escalabilidade automática. O mercado também tem investido em ferramentas que facilitam o versionamento e a observabilidade de modelos em tempo real, reduzindo riscos operacionais.

O deploy serverless ganhou destaque nos últimos anos como resposta à necessidade de reduzir esforço operacional e otimizar custos em workloads irregulares. Empresas que lidam com eventos esporádicos, automações orientadas a eventos e integrações entre sistemas passaram a adotar essa abordagem para executar modelos apenas quando necessário. A tendência é que o uso de serverless se expanda em cenários de automação inteligente, pipelines event-driven e prototipação rápida, especialmente quando o custo de manter infraestrutura ativa continuamente não se justifica.

Uma tendência clara no mercado é a adoção de arquiteturas híbridas, nas quais batch, tempo real e serverless coexistem dentro do mesmo ecossistema de Machine Learning. Em vez de buscar uma abordagem única, as organizações estão estruturando seus sistemas para que cada componente utilize o tipo de deploy mais adequado ao seu papel. Essa estratégia reflete uma visão mais madura de ML em produção, na qual decisões arquiteturais são tomadas com base em requisitos específicos e não em modismos tecnológicos.

Além disso, observa-se um movimento crescente em direção à padronização e governança do deploy de modelos. Práticas como definição de pipelines reprodutíveis, controle de versões, monitoramento de desempenho e rastreabilidade de previsões estão se tornando requisitos comuns, impulsionados tanto por demandas regulatórias quanto pela necessidade de confiabilidade operacional. Essas tendências reforçam a importância de compreender profundamente os diferentes tipos de deploy e seus trade-offs, preparando profissionais para projetar soluções de Machine Learning alinhadas às exigências reais do mercado.

#### O que você viu nesta aula?
Nesta aula, você compreendeu o deploy de modelos de Machine Learning como uma decisão arquitetural fundamental dentro de sistemas em produção. Foram apresentados os três principais tipos de deploy utilizados em ambientes de nuvem: batch, tempo real e serverless, destacando como cada abordagem define a forma de execução do modelo, o fluxo de dados e o comportamento do sistema como um todo.

Ao longo do conteúdo, foram analisadas em profundidade as características técnicas de cada tipo de deploy, incluindo requisitos de latência, custo operacional, escalabilidade, complexidade de manutenção e impacto no ciclo de vida do modelo. Também foram discutidos os trade-offs envolvidos na escolha entre essas estratégias, reforçando que não existe uma solução única e que a decisão deve ser guiada pelas restrições e necessidades do problema a ser resolvido.

Por fim, a aula apresentou uma visão de mercado e tendências atuais, mostrando como organizações reais combinam diferentes estratégias de deploy para construir arquiteturas híbridas de Machine Learning em produção. Esse conjunto de conceitos estabelece a base necessária para as próximas aulas, nas quais serão exploradas arquiteturas de nuvem, implementação prática e aspectos operacionais do deploy de modelos de Machine Learning.

#### Referências
- BAYLOR, D. et al. *TFX: A TensorFlow-Based Production-Scale Machine Learning Platform*. 2017. Disponível em: https://dl.acm.org/doi/10.1145/3097983.3098021. Acesso em: 16 jun. 2026.
- HUYEN, C. *Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications*. [s.l.]: O'Reilly Media, 2022.
- SCULLEY, D. et al. *Hidden Technical Debt in Machine Learning Systems*. 2015. Disponível em: https://proceedings.neurips.cc/paper_files/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf. Acesso em: 16 jun. 2026.

#### Palavras-chave
Batch. Tempo Real. Serverless. Trade-off. Latência. Escalabilidade. Custo Operacional. Machine Learning.

### Código e comandos
Nenhum bloco de código nesta aula. (O material desta aula é conceitual; o único código referenciado está no repositório externo https://github.com/FIAP/MLET_Deploy_EM_NUVEM/tree/main/Aula_1, não transcrito no dump.)

### Ferramentas / serviços citados
- MNIST (dataset)
- FastAPI
- Docker (container)
- Túnel de rede para exposição de aplicação local (simulação de endpoint em nuvem)
- Data warehouses e data lakes (contexto de mercado)

### Aplicabilidade ao Tech Challenge Fase 3
- A definição dos três padrões (batch, tempo real, serverless) e a Tabela 1 de trade-offs (latência × custo × escalabilidade) fornecem o vocabulário para justificar a **decisão arquitetural de nuvem no README** do TC.
- O cenário Hands On (modelo de classificação exposto via **FastAPI + Docker** como serviço de inferência em tempo real) é o molde direto para o **classificador NLP de laudos médicos** servido como API.
- A ênfase em "latência como requisito crítico" no deploy em tempo real conecta-se ao requisito de **otimização de latência (ONNX/quantização)** do TC.

---

## Aula 2 — Comportamento Computacional dos Modelos e Estratégias de Deploy
**Arquivo fonte:** `Aula 02.pdf` (15 páginas)
**Título na ementa:** Comportamento computacional dos modelos e estratégias de deploy (título inferido da capa: "AULA 02 - COMPORTAMENTO COMPUTACIONAL DOS MODELOS E ESTRATÉGIAS DE DEPLOY", coincide com a ementa)

### Conceitos-chave
- A escolha entre batch/tempo real/serverless depende também das **características computacionais do algoritmo**, não apenas da arquitetura de nuvem.
- **Custo de inferência**: modelos com custo constante escalam horizontalmente com facilidade; modelos com custo proporcional ao volume de dados exigem planejamento de capacidade.
- **Dependência de dados auxiliares** e **pré-processamento** como parte integrante do deploy.
- Deploy orientado ao algoritmo: modelos lineares → APIs de baixa latência; KMeans/baseados em distância → pipelines batch.
- Aproximação entre ML e engenharia de software (versionamento, testes, rastreabilidade, governança).

### Conteúdo

#### O que vem por aí?
Nesta aula, será aprofundada a relação entre os modelos de Machine Learning e as estratégias de deploy discutidas anteriormente, mostrando que a escolha entre batch, tempo real ou serverless não depende apenas da arquitetura de nuvem, mas também das características computacionais do algoritmo utilizado. O conteúdo parte da análise de como diferentes tipos de modelos realizam inferência, quais recursos consomem e como seu comportamento matemático impacta diretamente latência, custo e escalabilidade em produção.

Ao longo da aula, serão exploradas categorias clássicas de modelos, como regressões lineares, algoritmos probabilísticos, árvores de decisão, métodos baseados em distância e técnicas de clusterização, observando como cada uma dessas abordagens se comporta quando operacionalizada. A proposta é demonstrar que modelos aparentemente simples podem ser altamente eficientes para inferência sob demanda, enquanto outros exigem processamento mais estruturado e são naturalmente mais adequados a execuções em lote.

A aula também introduz dois cenários práticos que simulam comportamentos reais de deploy. Um deles apresenta a exposição de um modelo leve por meio de uma API para inferência em tempo real, enquanto o outro demonstra a execução de um processo analítico que precisa avaliar todo o conjunto de dados antes de gerar resultados. Esses exemplos servirão como base para compreender, de forma concreta, como decisões técnicas sobre modelos influenciam diretamente o desenho da solução em nuvem e sua forma de operação.

#### Hands On
Nesta etapa prática, serão desenvolvidos dois projetos distintos com o objetivo de demonstrar, na prática, como o comportamento de diferentes modelos de Machine Learning influencia diretamente a forma de deploy adotada. A proposta é evidenciar que não existe uma única estratégia universal de disponibilização de modelos, mas sim abordagens que precisam ser compatíveis com a natureza computacional de cada algoritmo e com o tipo de processamento exigido.

O primeiro projeto utiliza um modelo de regressão linear simples, treinado a partir de dados sintéticos, que será exposto por meio de uma API para realização de inferências sob demanda. Essa implementação simula um cenário de execução leve, no qual o modelo pode ser carregado rapidamente, responder a requisições individuais e operar de forma semelhante a serviços serverless ou aplicações em tempo real. A construção da API permite observar como modelos de baixa complexidade podem ser integrados diretamente a sistemas consumidores sem necessidade de processamento prévio em larga escala.

O segundo projeto apresenta um cenário baseado em clusterização com KMeans, no qual os dados precisam ser analisados em conjunto para que o resultado seja produzido. Nesse caso, o processamento ocorre como um job executado de forma controlada, simulando o comportamento de pipelines batch. A execução desse fluxo evidencia como determinados algoritmos dependem da visão global do dataset e não são naturalmente adequados para inferência isolada por requisição, exigindo uma estratégia de execução diferente da adotada no primeiro projeto.

Ambos os projetos serão organizados, executados e posteriormente containerizados, permitindo observar como a estrutura de código, a forma de execução e o empacotamento da aplicação variam conforme o tipo de modelo e o padrão de deploy escolhido.

Link com os códigos: https://github.com/FIAP/MLET_Deploy_EM_NUVEM/tree/main/Aula_2.

#### Saiba Mais
A escolha da estratégia de deploy em Machine Learning não está apenas relacionada à infraestrutura disponível, mas também ao comportamento computacional do modelo utilizado. Cada algoritmo possui características próprias de treinamento e inferência que influenciam diretamente o tempo de resposta, o consumo de memória, a necessidade de acesso ao conjunto completo de dados e a forma como o modelo pode ser operacionalizado. Compreender essas diferenças é essencial para evitar arquiteturas inadequadas, nas quais o modelo é tecnicamente funcional, mas operacionalmente ineficiente ou inviável.

Modelos lineares, por exemplo, tendem a apresentar inferência extremamente rápida e com baixo custo computacional, pois a predição se resume a operações matemáticas simples sobre um conjunto reduzido de parâmetros. Isso permite que sejam carregados facilmente em memória e executados sob demanda, sendo naturalmente compatíveis com APIs de baixa latência e ambientes que priorizam elasticidade. Já modelos mais estruturados, como aqueles baseados em árvores ou métodos probabilísticos, podem introduzir maior complexidade de cálculo ou necessidade de percorrer estruturas mais profundas, o que impacta diretamente o tempo de execução e a previsibilidade da resposta.

Algoritmos baseados em distância, como KMeans, apresentam um comportamento distinto, pois dependem da comparação entre pontos e centros previamente calculados, exigindo frequentemente acesso ao conjunto completo de dados para atualização ou validação dos agrupamentos. Essa característica torna sua execução mais adequada a pipelines controlados, nos quais o processamento é realizado de forma consolidada, não como respostas isoladas a eventos individuais. O custo de recalcular estruturas internas ou de manter grandes volumes de dados acessíveis em memória pode inviabilizar sua utilização em cenários de inferência contínua.

Além da complexidade algorítmica, fatores como dimensionalidade dos dados, necessidade de pré-processamento e frequência de atualização do modelo também influenciam a decisão de deploy. Modelos que exigem transformações extensas antes da inferência podem deslocar parte significativa do custo computacional para etapas anteriores à predição, alterando o ponto de equilíbrio entre execução em lote e execução sob demanda. Da mesma forma, modelos que precisam ser reavaliados periodicamente sobre novos dados podem se beneficiar de arquiteturas que favoreçam reprocessamento estruturado, garantindo consistência entre diferentes execuções.

Ao analisar o comportamento interno dos algoritmos, torna-se possível alinhar a estratégia de deploy com as características reais do modelo, evitando soluções genéricas que não consideram limitações práticas. Esse entendimento permite projetar sistemas mais eficientes, nos quais o tipo de modelo, o padrão de execução e a infraestrutura trabalham de forma coerente, reduzindo custos operacionais e aumentando a previsibilidade do sistema em produção.

A análise do comportamento dos modelos também envolve compreender como eles utilizam recursos computacionais ao longo do tempo. Alguns algoritmos possuem custo de inferência constante, independentemente do tamanho do conjunto de dados original, enquanto outros apresentam custo proporcional ao volume de informação que precisa ser considerado durante a execução. Essa diferença influencia diretamente a previsibilidade de desempenho em produção. Modelos com custo de inferência estável são mais fáceis de escalar horizontalmente, pois cada nova requisição possui impacto semelhante sobre o sistema. Já modelos cujo custo varia conforme o tamanho do dataset ou da estrutura interna exigem maior planejamento de capacidade e, muitas vezes, estratégias específicas de particionamento ou execução controlada.

Outro aspecto relevante é o carregamento do modelo em memória e o tempo necessário para inicialização do ambiente de execução. Em aplicações onde o modelo é instanciado repetidamente, como em cenários sob demanda, o tamanho do artefato treinado e suas dependências passam a ter impacto direto na latência total. Modelos compactos, com poucos parâmetros e sem necessidade de estruturas auxiliares complexas, podem ser carregados rapidamente e responder de forma eficiente a execuções eventuais. Por outro lado, modelos que dependem de grandes matrizes, estruturas de indexação ou múltiplos componentes de pré-processamento podem apresentar tempos de inicialização mais elevados, o que altera sua adequação a determinados padrões de deploy.

A dependência de dados auxiliares durante a inferência também deve ser considerada. Alguns modelos funcionam de maneira autossuficiente após o treinamento, necessitando apenas das variáveis de entrada para gerar a predição. Outros, no entanto, requerem acesso a informações adicionais, como estatísticas globais, estruturas de agrupamento ou transformações previamente calculadas. Quando esse tipo de dependência existe, torna-se necessário garantir que tais dados estejam disponíveis de forma consistente durante a execução, o que pode favorecer arquiteturas mais centralizadas ou processos de execução planejados.

A forma como os dados são preparados antes da inferência é igualmente determinante. Em muitos casos, o modelo não recebe os dados em seu formato bruto, sendo necessário aplicar normalizações, codificações ou transformações derivadas do conjunto de treinamento. Essas etapas de pré-processamento podem representar uma parcela significativa do tempo total de execução e precisam ser consideradas como parte integrante do deploy, não como um detalhe secundário. Em ambientes produtivos, manter a equivalência entre o pré-processamento utilizado no treinamento e aquele aplicado na inferência é fundamental para garantir consistência dos resultados.

A frequência com que o modelo precisa ser atualizado ou reavaliado também influencia a estratégia de operacionalização. Modelos utilizados em contextos estáveis podem permanecer longos períodos sem necessidade de ajuste, permitindo ciclos de execução previsíveis. Já modelos aplicados a dados dinâmicos podem exigir reprocessamento recorrente, validação constante e integração com pipelines de atualização, o que altera o desenho da solução e o tipo de execução mais apropriado.

Esses fatores demonstram que o deploy de Machine Learning não pode ser tratado como uma camada isolada, mas sim como uma extensão natural do próprio modelo e de seu comportamento matemático. A eficiência operacional depende da compatibilidade entre o algoritmo escolhido, o fluxo de dados existente e a forma como o sistema executa suas tarefas. Ao compreender essa relação de maneira detalhada, torna-se possível projetar soluções mais coerentes, nas quais o modelo não apenas produz boas métricas em ambiente controlado, mas também mantém desempenho, custo e confiabilidade adequados quando inserido em um sistema real.

Além das questões relacionadas ao desempenho e ao consumo de recursos, a operacionalização de modelos também exige atenção à forma como eles são integrados ao restante do ecossistema de software. Em aplicações reais, o modelo raramente atua de maneira isolada; ele passa a fazer parte de um conjunto maior de serviços, bancos de dados, filas de processamento e interfaces de comunicação. Essa integração exige definição clara de contratos de entrada e saída, padronização de formatos de dados e mecanismos de validação que garantam que o modelo receba informações compatíveis com aquelas utilizadas durante o treinamento.

A governança do modelo em produção é outro elemento essencial. Uma vez disponibilizado, é necessário acompanhar não apenas o funcionamento técnico da aplicação, mas também o comportamento das previsões ao longo do tempo. Mudanças no perfil dos dados de entrada podem afetar a qualidade das respostas, exigindo monitoramento contínuo de métricas operacionais e estatísticas. Esse acompanhamento permite identificar situações em que o modelo deixa de representar adequadamente o fenômeno que se propõe a analisar, indicando a necessidade de reavaliação ou reprocessamento.

Também é importante considerar a rastreabilidade das execuções realizadas. Em muitos cenários, especialmente aqueles que envolvem decisões automatizadas, é necessário manter registros que permitam compreender como determinada previsão foi gerada, quais dados foram utilizados e qual versão do modelo estava em operação naquele momento. Essa capacidade de reconstrução é fundamental tanto para análise técnica quanto para requisitos de auditoria e conformidade.

Outro ponto frequentemente observado em ambientes produtivos é a necessidade de isolamento entre etapas de processamento. Separar claramente o momento de preparação dos dados, a execução do modelo e o armazenamento dos resultados contribui para maior estabilidade e facilita intervenções pontuais sem comprometer todo o sistema. Essa organização modular permite que ajustes sejam feitos de forma incremental, reduzindo riscos e tornando a evolução da solução mais controlada.

À medida que os sistemas de Machine Learning se tornam parte integrante das operações organizacionais, cresce também a necessidade de padronizar práticas de desenvolvimento, versionamento e validação. O modelo passa a ser tratado como um componente de software que precisa seguir ciclos de atualização, testes e documentação, garantindo que sua evolução ocorra de maneira coordenada com o restante da aplicação. Essa abordagem aproxima o desenvolvimento de soluções de ML das práticas tradicionais de engenharia de software, criando um ambiente mais previsível e sustentável.

Essas considerações reforçam que a decisão sobre como executar um modelo está profundamente ligada à forma como ele será mantido, observado e integrado ao longo do tempo. Mais do que escolher uma tecnologia específica, trata-se de compreender o papel do modelo dentro do sistema e alinhar sua execução aos requisitos técnicos e organizacionais que sustentam sua operação contínua.

#### Mercado, Cases e Tendências
No contexto atual, observa-se que organizações que adotam Machine Learning de forma mais madura não tratam o deploy apenas como uma etapa final do projeto, mas sim parte integrante da estratégia de engenharia de dados e software. Empresas que operam em grande escala passaram a estruturar seus ambientes de ML com pipelines bem definidos, nos quais treinamento, validação, disponibilização e monitoramento são etapas conectadas e contínuas. Essa integração reduz o tempo necessário para levar modelos à produção e aumenta a capacidade de responder rapidamente a mudanças nos dados ou nas demandas do negócio.

Outro movimento relevante é a aproximação entre práticas de Machine Learning e conceitos tradicionais de engenharia de software. A adoção de versionamento estruturado, testes automatizados, integração contínua e padronização de ambientes tornou-se comum em equipes que trabalham com modelos em produção. Essa convergência tem sido impulsionada pela necessidade de confiabilidade operacional, já que modelos utilizados em larga escala precisam atender aos mesmos requisitos de estabilidade e previsibilidade esperados de qualquer outro sistema crítico.

Casos de mercado mostram também uma tendência crescente de especialização das arquiteturas conforme o tipo de modelo e o fluxo de dados envolvido. Em vez de centralizar toda a execução em uma única plataforma, as empresas distribuem suas soluções entre diferentes padrões de processamento, combinando execuções periódicas, serviços de inferência e processos orientados a eventos. Essa fragmentação controlada permite ajustar custo, desempenho e complexidade de acordo com a finalidade de cada componente, evitando sobrecarga desnecessária em partes do sistema que não exigem alta disponibilidade ou baixa latência.

A evolução das plataformas de nuvem também tem influenciado diretamente essas decisões. Provedores passaram a oferecer serviços gerenciados que simplificam desde a execução de jobs analíticos até a disponibilização de APIs escaláveis, reduzindo a necessidade de gerenciamento manual de infraestrutura. Como consequência, equipes conseguem concentrar esforços na modelagem e na análise de dados, delegando à plataforma aspectos operacionais mais repetitivos. Essa mudança contribui para acelerar ciclos de desenvolvimento e facilitar experimentação, sem perder controle sobre custos e desempenho.

Outra tendência observada é o aumento da preocupação com governança e transparência no uso de modelos. Organizações que dependem de decisões automatizadas precisam garantir rastreabilidade, explicabilidade e controle de versões, principalmente em setores regulados. Isso tem levado à criação de processos mais formais de validação e acompanhamento, nos quais a disponibilização de um modelo passa por etapas semelhantes às de homologação de software tradicional.

Esse cenário demonstra que o mercado caminha para uma integração cada vez maior entre ciência de dados, engenharia de software e arquitetura de sistemas, consolidando o deploy como um elemento central na construção de soluções de Machine Learning sustentáveis e operacionais em larga escala.

#### O que você viu nesta aula?
Nesta aula, você aprofundou o entendimento sobre como o comportamento dos modelos de Machine Learning influencia diretamente a forma como eles devem ser disponibilizados em produção. Foi discutido que a escolha da estratégia de deploy não é determinada apenas pela infraestrutura disponível, mas pelas características matemáticas e computacionais de cada algoritmo, incluindo custo de inferência, dependência de dados, necessidade de processamento prévio e frequência de execução.

Ao longo do conteúdo, foram analisadas diferentes categorias de modelos e como suas propriedades impactam latência, consumo de recursos e previsibilidade operacional. A aula mostrou que modelos leves e com inferência simples podem ser facilmente expostos como serviços sob demanda, enquanto algoritmos que exigem análise conjunta dos dados tendem a se adaptar melhor a execuções estruturadas, nas quais o processamento ocorre de forma consolidada.

A etapa prática apresentou dois cenários complementares que permitiram observar essas diferenças de maneira concreta. A construção de um serviço de inferência com regressão linear evidenciou como modelos de baixa complexidade podem ser integrados diretamente a aplicações, enquanto a execução de um processo de clusterização demonstrou a necessidade de um fluxo mais controlado e orientado a processamento completo do conjunto de dados.

Com isso, a aula estabeleceu a conexão entre teoria algorítmica e decisões arquiteturais, reforçando que a operacionalização de Machine Learning depende do alinhamento entre modelo, dados e forma de execução, preparando o caminho para compreender, nas próximas aulas, como essas soluções são estruturadas e gerenciadas em ambientes de nuvem.

#### Referências
- BAYLOR, D. et al. *TFX: A TensorFlow-Based Production-Scale Machine Learning Platform*. 2017. Disponível em: https://dl.acm.org/doi/10.1145/3097983.3098021. Acesso em: 16 jun. 2026.
- HUYEN, C. *Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications*. [s.l.]: O'Reilly Media, 2022.
- SCULLEY, D. et al. *Hidden Technical Debt in Machine Learning Systems*. 2015. Disponível em: https://proceedings.neurips.cc/paper_files/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf. Acesso em: 16 jun. 2026.

#### Palavras-chave
Comportamento Computacional de Modelos. Inferência de Machine Learning. Serverless. Custo de Inferência. Deploy Orientado ao Algoritmo.

> [NOTA — não é conteúdo FIAP]: na pág. 14 do dump, a lista de palavras-chave aparece sem o ponto separador entre "Comportamento Computacional de Modelos" e "Inferência de Machine Learning" ("...Modelos Inferência de Machine Learning."). Reconstruído com o separador para clareza.

### Código e comandos
Nenhum bloco de código nesta aula. (Aula conceitual; os dois projetos — regressão linear via API e KMeans batch — estão no repositório externo https://github.com/FIAP/MLET_Deploy_EM_NUVEM/tree/main/Aula_2, não transcrito no dump.)

### Ferramentas / serviços citados
- Regressão linear (modelo leve, API sob demanda)
- KMeans / clusterização (job batch)
- API / Docker (containerização)
- Integração continua e testes automatizados (contexto de mercado)

### Aplicabilidade ao Tech Challenge Fase 3
- A regra "modelos com custo de inferência constante escalam horizontalmente; modelos com custo variável exigem planejamento" ajuda a justificar a estratégia de serving do **classificador NLP de laudos** e a **otimização de latência (ONNX/quantização)** no TC.
- A ênfase em "manter equivalência entre pré-processamento de treino e de inferência" é diretamente aplicável ao pipeline NLP (tokenização/normalização de texto dos laudos).
- A aproximação ML ↔ engenharia de software (versionamento, testes automatizados, integração contínua, rastreabilidade) fundamenta os requisitos de **CI/CD com GitHub Actions** e **Airflow** do TC.

---

## Aula 3 — Deploy de ML na AWS: ECR, EC2, Lambda, AWS Batch e SageMaker
**Arquivo fonte:** `Aula 03 - Deploy de ML na Aws - Ecr, Ec2, Lambda, Aws Batch e Sagemaker.pdf` (13 páginas)
**Título na ementa:** Deploy de ML na AWS: ECR, EC2, Lambda, AWS Batch e Sagemaker

### Conceitos-chave
- Três camadas principais da arquitetura AWS para ML: **armazenamento**, **distribuição de artefatos** e **execução do processamento**.
- **Amazon ECR** como repositório central de imagens Docker, com integração via IAM.
- Quatro estratégias de execução: **EC2** (tempo real), **AWS Lambda** (serverless/eventos, com cold start), **AWS Batch** (lote), **Amazon SageMaker** (gerenciado, endpoints de inferência).
- **Amazon S3** como ponto central de armazenamento de datasets, modelos e resultados.

### Conteúdo

#### O que vem por aí?
Nesta aula, o foco está na aplicação prática dos padrões de deploy de modelos de Machine Learning utilizando serviços da AWS. Após compreender como diferentes tipos de modelos influenciam a escolha entre tempo real, processamento em lote ou execuções sob demanda, o objetivo é demonstrar como essas abordagens são implementadas em um ambiente de nuvem real por meio de serviços específicos da plataforma.

Será apresentada uma visão geral dos principais componentes da AWS envolvidos no deploy de aplicações com containers e modelos de Machine Learning, incluindo serviços de computação, armazenamento e gerenciamento de imagens. O Amazon Elastic Container Registry (ECR) é destacado como elemento central para armazenar e distribuir as imagens de container utilizadas nos diferentes serviços de execução.

A aula aborda quatro estratégias principais de execução de modelos na AWS: inferência em tempo real com instâncias EC2, oferecendo maior controle do ambiente; execução serverless com AWS Lambda, permitindo processamento sob demanda; processamento em lote com AWS Batch, voltado para cargas analíticas escaláveis; e o uso do Amazon SageMaker como solução gerenciada para treinamento e deploy de modelos.

Além dos conceitos, a aula inclui atividades práticas em que projetos anteriores são adaptados para a infraestrutura da AWS. Isso permite compreender como aplicações locais podem ser transformadas em soluções na nuvem, utilizando containers e serviços gerenciados, e ao final proporciona uma visão clara sobre como selecionar o melhor padrão de deploy conforme os requisitos da aplicação.

#### Hands On
Até este ponto, foram apresentados os principais serviços da AWS para operacionalizar modelos de Machine Learning em diferentes arquiteturas. Embora conceitos como batch, tempo real e serverless sejam independentes de provedor, cada plataforma oferece ferramentas específicas para viabilizar essas estratégias em produção. Na AWS, esses recursos são organizados de forma modular, permitindo a construção de soluções por meio da combinação de serviços de computação, armazenamento e containers.

Antes da parte prática, destaca-se a importância do uso de containers. Essa abordagem permite empacotar o modelo, suas dependências e o código em uma única imagem, garantindo consistência de execução em diferentes ambientes. Assim, o mesmo artefato pode ser reutilizado tanto localmente quanto em múltiplos serviços de nuvem, facilitando a portabilidade e a padronização dos deployments.

Outro elemento essencial é o registro de imagens de container. Na AWS, o Amazon Elastic Container Registry (ECR) desempenha esse papel, permitindo armazenar e distribuir imagens Docker de forma centralizada. Com isso, uma única imagem pode ser compartilhada entre diversos serviços, como instâncias de computação, funções serverless e aplicações de processamento em lote.

Na prática, o deploy envolve etapas recorrentes, como a organização do projeto, a criação da imagem de container e o envio dessa imagem para um repositório na nuvem. A partir disso, diferentes estratégias de execução podem ser aplicadas, reutilizando o mesmo artefato. Nos exercícios, projetos simples são adaptados para a AWS, permitindo visualizar como aplicações locais se transformam em soluções em nuvem baseadas em diferentes padrões de execução.

#### Saiba Mais
A operacionalização de modelos de Machine Learning em ambientes de nuvem envolve muito mais do que simplesmente disponibilizar um algoritmo treinado para uso externo. Quando um modelo passa do ambiente de desenvolvimento para um sistema real, ele se torna parte de uma infraestrutura distribuída que precisa lidar com variáveis como escalabilidade, confiabilidade, custo e segurança. Nesse contexto, plataformas de nuvem oferecem um conjunto de serviços que abstraem parte da complexidade da infraestrutura física, permitindo que equipes de engenharia concentrem seus esforços na construção e manutenção das aplicações.

No caso da AWS, a disponibilização de aplicações baseadas em Machine Learning costuma envolver três camadas principais: armazenamento, distribuição de artefatos e execução do processamento. O armazenamento é responsável por manter dados de entrada, modelos treinados e resultados gerados pelos processos analíticos. A distribuição de artefatos garante que aplicações e dependências estejam disponíveis para os serviços de execução, normalmente por meio de imagens de container. Já a camada de execução é responsável por rodar efetivamente o código da aplicação, podendo assumir diferentes formas dependendo do padrão de processamento escolhido.

O uso de containers tornou-se uma prática comum nesse tipo de arquitetura porque oferece portabilidade e consistência entre ambientes. Em vez de instalar manualmente bibliotecas, frameworks e dependências em cada servidor, todo o ambiente necessário para executar o modelo pode ser empacotado em uma única imagem. Essa imagem pode então ser armazenada em um repositório central e reutilizada em diferentes serviços da nuvem. Essa abordagem reduz problemas relacionados a diferenças de configuração entre ambientes e facilita a replicação da aplicação em múltiplas instâncias quando necessário.

Dentro da AWS, o Amazon Elastic Container Registry desempenha um papel importante nesse fluxo ao funcionar como um repositório de imagens Docker integrado ao restante da plataforma. Quando uma imagem é enviada para o ECR, ela se torna acessível para diversos serviços da AWS que suportam execução baseada em containers. Isso permite que a mesma aplicação seja utilizada em diferentes arquiteturas de execução, mantendo consistência no ambiente e simplificando o processo de atualização quando novas versões do modelo ou do código são disponibilizadas.

Outro aspecto relevante é a diversidade de serviços de computação disponíveis na plataforma. Em vez de oferecer apenas um tipo de ambiente de execução, a AWS disponibiliza diferentes modelos de computação que atendem a necessidades distintas. Instâncias virtuais, por exemplo, permitem controle total sobre o sistema operacional e os recursos de hardware, sendo adequadas para aplicações que precisam permanecer em execução continuamente. Já serviços orientados a eventos executam código apenas quando necessário, permitindo que aplicações sejam acionadas de forma pontual sem a necessidade de manter infraestrutura ativa durante todo o tempo.

Essa diversidade possibilita implementar arquiteturas de Machine Learning que combinam múltiplos padrões de execução dentro de um mesmo sistema. Uma aplicação pode, por exemplo, utilizar um serviço persistente para atender requisições de baixa latência enquanto tarefas analíticas mais pesadas são executadas periodicamente em processos independentes. Esse tipo de organização permite separar responsabilidades dentro do sistema, garantindo que operações críticas não sejam impactadas por atividades que exigem maior volume de processamento.

Além da camada de execução, o armazenamento de dados também desempenha papel central na arquitetura de soluções baseadas em Machine Learning. Serviços de armazenamento de objetos são frequentemente utilizados para manter datasets, artefatos de treinamento e resultados gerados pelos processos analíticos. Esses dados podem ser compartilhados entre diferentes serviços da plataforma, permitindo que pipelines de treinamento, inferência e processamento em lote utilizem as mesmas fontes de informação de maneira consistente.

A integração entre armazenamento, distribuição de artefatos e execução de aplicações cria a base necessária para construir pipelines completos de Machine Learning na nuvem. A partir dessa estrutura, torna-se possível desenvolver soluções que vão desde serviços de inferência em tempo real até sistemas de processamento massivo de dados, utilizando os recursos da plataforma de forma coordenada e escalável.

Ao analisar com mais profundidade os serviços utilizados nesta aula, é possível perceber que cada um deles foi projetado para atender um tipo específico de carga de trabalho dentro da arquitetura de aplicações em nuvem. Em vez de oferecer apenas uma infraestrutura genérica de computação, a AWS organiza seus serviços de forma que diferentes padrões de execução possam ser implementados de maneira mais eficiente, permitindo escolher o ambiente mais adequado para cada tipo de aplicação ou modelo de Machine Learning.

**Amazon ECR.** O Amazon Elastic Container Registry (ECR) ocupa um papel central nesse fluxo ao funcionar como um repositório gerenciado para imagens de container. Em arquiteturas modernas baseadas em containers, o código da aplicação, o modelo treinado e todas as dependências necessárias para execução são empacotados em uma imagem Docker. Essa imagem é então enviada para um registro centralizado, onde pode ser versionada, armazenada e distribuída para diferentes serviços de execução. O ECR oferece integração nativa com outros serviços da AWS, controle de acesso via IAM e mecanismos de autenticação que permitem que apenas recursos autorizados façam download das imagens. Na prática, isso transforma o ECR em um ponto de distribuição seguro e escalável para aplicações containerizadas.

**Amazon EC2.** No cenário de inferência em tempo real apresentado na aula, o serviço responsável pela execução da aplicação é o Amazon EC2 (Elastic Compute Cloud). O EC2 oferece instâncias virtuais configuráveis que permitem controle completo sobre o ambiente de execução, incluindo sistema operacional, capacidade de CPU, memória e, quando necessário, aceleração por GPU. Esse nível de controle torna o EC2 particularmente útil para aplicações que precisam permanecer ativas continuamente, como APIs de inferência que respondem a requisições externas. Ao executar containers em instâncias EC2, é possível manter o modelo carregado em memória, reduzindo o tempo de resposta e garantindo latência previsível para aplicações que exigem processamento imediato.

**AWS Lambda.** Em contraste com esse modelo de execução contínua, o AWS Lambda representa uma abordagem orientada a eventos, na qual o código é executado apenas quando acionado por uma requisição ou evento específico. Esse modelo elimina a necessidade de gerenciar servidores ou manter instâncias permanentemente ativas, pois a infraestrutura é provisionada automaticamente pela plataforma no momento da execução. Para workloads de Machine Learning leves, especialmente aqueles que utilizam modelos pequenos e com inferência rápida, o Lambda pode ser uma alternativa eficiente, pois permite escalar automaticamente conforme a demanda e cobrar apenas pelo tempo efetivo de execução. Entretanto, essa abordagem também introduz desafios como o chamado cold start, que ocorre quando a função precisa inicializar o ambiente antes de executar o código.

**AWS Batch.** Para cenários de processamento em lote, a AWS oferece o AWS Batch, um serviço projetado para executar jobs computacionais de forma escalável e gerenciada. Diferentemente de aplicações que respondem a requisições em tempo real, workloads batch geralmente processam grandes volumes de dados de maneira periódica ou programada. O AWS Batch permite que esses jobs sejam executados em ambientes containerizados, gerenciando automaticamente a alocação de recursos computacionais necessários para cada tarefa. Isso significa que o sistema pode escalar dinamicamente conforme o tamanho da carga de trabalho, executando múltiplos jobs em paralelo quando necessário. Em pipelines de Machine Learning, esse tipo de serviço é frequentemente utilizado para tarefas como reprocessamento de dados, geração de previsões em massa ou execução de modelos que precisam analisar grandes datasets antes de produzir resultados.

**Amazon S3.** Outro componente importante nesse tipo de arquitetura é o Amazon S3 (Simple Storage Service), responsável pelo armazenamento de objetos na nuvem. O S3 é amplamente utilizado em pipelines de Machine Learning para armazenar datasets, artefatos de treinamento, modelos serializados e resultados gerados pelos processos analíticos. Por ser altamente durável e escalável, ele permite que diferentes serviços da AWS acessem os mesmos dados de forma consistente, funcionando como um ponto central de armazenamento dentro da arquitetura. Em cenários batch, por exemplo, um job pode ler dados diretamente de um bucket S3, processá-los e escrever os resultados novamente no mesmo serviço, permitindo que outras aplicações ou pipelines consumam essas informações posteriormente.

**Amazon SageMaker.** Por fim, a aula também aborda o Amazon SageMaker, um serviço gerenciado voltado especificamente para o ciclo de vida de modelos de Machine Learning. O SageMaker oferece ferramentas que facilitam desde o treinamento até a disponibilização de modelos em produção, reduzindo a necessidade de configurar manualmente infraestrutura para essas etapas. Um dos pontos mais relevantes desse serviço é a possibilidade de criar endpoints de inferência gerenciados, nos quais o modelo pode ser implantado e escalado automaticamente de acordo com a demanda. Além disso, o SageMaker suporta tanto frameworks populares de Machine Learning quanto containers customizados, permitindo que modelos desenvolvidos em diferentes ambientes sejam integrados à plataforma.

A utilização combinada desses serviços ilustra como arquiteturas modernas de Machine Learning na nuvem são compostas por múltiplos componentes especializados. Em vez de depender de uma única infraestrutura para todas as tarefas, os sistemas passam a distribuir responsabilidades entre serviços que foram projetados para lidar com tipos específicos de processamento. Esse modelo permite construir soluções mais flexíveis, nas quais inferência em tempo real, processamento em lote e execução sob demanda podem coexistir dentro do mesmo ecossistema, aproveitando os recursos da nuvem de forma mais eficiente.

#### O que você viu nesta aula?
Nesta aula, você compreendeu como diferentes estratégias de deploy de modelos de Machine Learning podem ser implementadas utilizando serviços da AWS. Após compreender nas aulas anteriores os conceitos de execução em tempo real, processamento em lote e arquiteturas serverless, o foco passou a ser a aplicação desses padrões dentro de um ambiente de nuvem real, utilizando recursos específicos da plataforma.

Ao longo do conteúdo, foram apresentados serviços fundamentais da AWS que permitem construir essas arquiteturas. Inicialmente, foi discutido o papel dos containers no empacotamento de aplicações de Machine Learning e como o Amazon Elastic Container Registry pode ser utilizado para armazenar e distribuir imagens utilizadas pelos serviços de execução. Em seguida, foram explorados diferentes ambientes de computação disponíveis na plataforma, destacando como cada um deles se adapta a diferentes tipos de workloads.

A aula também apresentou quatro abordagens de execução de modelos na AWS. A primeira envolveu o uso de instâncias EC2 para disponibilizar aplicações de inferência em tempo real, permitindo maior controle sobre o ambiente de execução. Em seguida, foi demonstrado como funções AWS Lambda podem ser utilizadas para executar modelos de forma serverless, acionadas apenas quando necessário. Também foi apresentado o uso do AWS Batch para executar jobs de processamento em lote, adequado para tarefas que analisam grandes volumes de dados. Por fim, foi discutido o Amazon SageMaker como uma alternativa gerenciada para treinamento e deploy de modelos de Machine Learning.

#### Referências
- BRIKMAN, Y. *TFX: Terraform: Up and Running*. Sebastopol: O'Reilly Media, 2019.
- GÉRON, A. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. 3rd ed. Sebastopol: O'Reilly Media, 2022.
- HUYEN, C. *Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications*. Sebastopol: O'Reilly Media, 2022.

> [NOTA — não é conteúdo FIAP]: a referência de Brikman aparece no dump (pág. 11) como "BRIKMAN, Y. TFX: Terraform: Up and Running" — o prefixo "TFX:" parece ser um erro de OCR/colagem do PDF original, pois o livro de Yevgeniy Brikman é apenas "Terraform: Up and Running". Transcrito como está.

#### Palavras-chave
AWS. ECR. EC2. Batch. S3. Lambda. Sagemaker. Sagemaker Studio. CloudWatch. IAM. Logs.

### Código e comandos
Nenhum bloco de código nesta aula. (Aula conceitual descritiva; a parte prática adapta projetos anteriores para a AWS, mas nenhum comando/código é transcrito no dump.)

### Ferramentas / serviços citados
- Amazon ECR (Elastic Container Registry)
- Amazon EC2 (Elastic Compute Cloud)
- AWS Lambda
- AWS Batch
- Amazon S3 (Simple Storage Service)
- Amazon SageMaker / SageMaker Studio
- AWS IAM
- Amazon CloudWatch (logs)
- Docker

### Aplicabilidade ao Tech Challenge Fase 3
- Mapeia diretamente as opções de **decisão arquitetural de nuvem** para o TC: EC2 (API de inferência em tempo real), Lambda (serverless), Batch (lote), SageMaker (gerenciado) — útil para a justificativa no README.
- **ECR + Docker** é o padrão para empacotar o classificador NLP e integrá-lo a um pipeline de **CI/CD (GitHub Actions)** que faz build/push da imagem.
- **CloudWatch** (citado nas palavras-chave) é o análogo AWS ao stack **Prometheus/Grafana** exigido para observabilidade.

---

## Aula 4 — Deploy de ML na Azure: ACR, VM, Container Apps, Jobs, Azure ML
**Arquivo fonte:** `Aula 04 - Deploy de ML na Azure - Acr, Vm, Container Apps, Jobs, Azure ML.pdf` (13 páginas)
**Título na ementa:** Deploy de ML na Azure: ACR, VM, Container Apps, Jobs, Azure ML

### Conceitos-chave
- Mesma lógica arquitetural das aulas anteriores, mapeada para o **Microsoft Azure**.
- Três elementos fundamentais: **execução de aplicações**, **armazenamento de dados** e **distribuição de artefatos**.
- **Azure Container Registry (ACR)** como repositório central de imagens.
- Serviços de execução: **Azure VM** (controle total), **Azure Container Apps** (serverless, escala a zero), **Azure Container Apps Jobs** (batch), **Azure ML** (ciclo de vida gerenciado).
- **Azure Blob Storage** para dados de entrada/saída; **identidades gerenciadas** para acesso seguro sem expor credenciais.

### Conteúdo

#### O que vem por aí?
Nesta aula, você irá aplicar os conceitos de deploy de modelos de Machine Learning em um novo ambiente de nuvem, explorando como diferentes padrões de execução podem ser implementados utilizando os serviços do Microsoft Azure. Mantendo a mesma lógica arquitetural trabalhada anteriormente, o foco será compreender como essas estratégias se traduzem em um ecossistema diferente, analisando equivalências entre serviços e observando como cada plataforma organiza seus recursos.

Ao longo da aula, serão apresentados os principais componentes do Azure utilizados para execução de aplicações baseadas em containers, incluindo o registro de imagens, ambientes de computação e serviços orientados a eventos. A proposta é demonstrar como arquiteturas de inferência em tempo real, processamento em lote e execuções sob demanda podem ser construídas utilizando combinações específicas desses serviços, respeitando as características de cada tipo de workload.

Também serão exploradas as diferenças operacionais entre as abordagens, destacando aspectos como controle de infraestrutura, escalabilidade automática, latência e custo. Essa análise permitirá entender não apenas como implementar as soluções, mas também como tomar decisões arquiteturais considerando o comportamento esperado da aplicação em produção.

Na parte prática, projetos desenvolvidos anteriormente serão adaptados para execução no Azure, permitindo observar como aplicações locais podem ser empacotadas, distribuídas e executadas na nuvem utilizando containers. Esse processo evidencia como o mesmo modelo pode ser reutilizado em diferentes contextos de deploy, reforçando a importância da padronização e da portabilidade em soluções de Machine Learning.

Ao final da aula, você terá uma visão clara de como estruturar deploys de modelos no Azure e compreenderá como mapear arquiteturas entre diferentes provedores de nuvem, mantendo consistência nos conceitos e adaptando a implementação às ferramentas disponíveis em cada plataforma.

#### Hands On
Nesta etapa prática, serão implementadas diferentes estratégias de deploy de modelos de Machine Learning utilizando serviços do Microsoft Azure com o objetivo de aplicar, em um ambiente real, os conceitos de execução em tempo real, serverless e processamento em lote. A proposta é demonstrar como esses padrões podem ser reproduzidos na plataforma, evidenciando as adaptações necessárias em relação à arquitetura.

O fluxo prático será baseado em aplicações já desenvolvidas, que serão empacotadas em imagens Docker. Esse processo garante que o modelo, o código e as dependências estejam organizados em um único artefato reutilizável, que será armazenado em um repositório de containers. Assim, diferentes serviços do Azure poderão acessar e executar essas imagens conforme a estratégia de deploy adotada.

No cenário de tempo real, a aplicação será executada em uma máquina virtual, mantendo o modelo constantemente disponível para responder a requisições. Em contraste, o modelo serverless será configurado para execução sob demanda, sendo ativado apenas quando necessário, com escalabilidade automática gerenciada pela plataforma. Já o processamento em lote será implementado por meio de jobs programados que processam grandes volumes de dados armazenados na nuvem.

Por fim, será apresentado o uso de um serviço gerenciado de Machine Learning do Azure, no qual o modelo é registrado e disponibilizado por meio de um endpoint controlado pela plataforma. Ao longo da prática, observa-se como os mesmos projetos podem ser reutilizados em diferentes estratégias, reforçando a importância da padronização com containers e da separação entre modelo, código e infraestrutura.

#### Saiba Mais
A utilização de plataformas de nuvem para deploy de modelos de Machine Learning envolve compreender não apenas os serviços disponíveis, mas também como esses serviços se organizam dentro de uma arquitetura distribuída. No caso do Azure, assim como em outros provedores, a construção de soluções em produção se apoia na combinação de três elementos fundamentais: execução de aplicações, armazenamento de dados e distribuição de artefatos. A forma como esses componentes se conectam determina a eficiência, a escalabilidade e o custo da solução.

Um dos pilares dessa arquitetura é o uso de containers como unidade padrão de execução. Ao empacotar o modelo, o código de inferência e todas as dependências em uma imagem Docker, torna-se possível garantir consistência entre ambientes locais e ambientes de produção. Essa abordagem reduz problemas de compatibilidade e facilita a replicação da aplicação em diferentes serviços da nuvem. Além disso, o uso de containers permite que a mesma aplicação seja reutilizada em múltiplos cenários de deploy, desde execução contínua em máquinas virtuais até execução sob demanda em serviços serverless.

Dentro do Azure, o Azure Container Registry desempenha o papel de repositório central para essas imagens. Ele permite armazenar, versionar e distribuir containers de forma segura, utilizando mecanismos de autenticação e controle de acesso integrados ao restante da plataforma. Ao centralizar as imagens em um registry, diferentes serviços de execução podem consumir o mesmo artefato, garantindo que a aplicação mantenha o mesmo comportamento independentemente de onde esteja sendo executada.

Outro aspecto importante é a diversidade de modelos de computação oferecidos pela plataforma. O Azure disponibiliza ambientes que variam desde máquinas virtuais com controle total até serviços totalmente gerenciados que abstraem a infraestrutura. Essa variedade permite escolher o nível de controle e abstração mais adequado para cada aplicação. Em cenários que exigem maior previsibilidade de desempenho e customização, ambientes mais controlados são preferíveis. Já em situações nas quais a prioridade é reduzir esforço operacional e otimizar custos, serviços gerenciados e orientados a eventos tendem a ser mais adequados.

A escolha entre esses modelos de execução está diretamente relacionada ao comportamento do workload. Aplicações que precisam responder rapidamente a requisições externas se beneficiam de ambientes que mantêm o modelo carregado em memória, reduzindo o tempo de resposta. Por outro lado, workloads que ocorrem de forma intermitente podem aproveitar ambientes que escalam automaticamente, sendo ativados apenas quando necessário. Já tarefas que processam grandes volumes de dados de forma periódica se adaptam melhor a execuções controladas, nas quais o processamento ocorre de forma assíncrona.

O armazenamento de dados também exerce um papel fundamental nesse tipo de arquitetura. Em soluções de Machine Learning, é comum que dados de entrada, modelos treinados e resultados de processamento sejam compartilhados entre diferentes componentes do sistema. Serviços de armazenamento de objetos permitem organizar esses dados de forma estruturada, facilitando o acesso por diferentes aplicações e garantindo persistência mesmo quando os serviços de execução são efêmeros.

Além disso, a separação entre camadas de armazenamento, processamento e distribuição contribui para a construção de sistemas mais modulares e escaláveis. Cada componente pode ser dimensionado de forma independente, permitindo que alterações em uma parte da arquitetura não impactem diretamente as demais. Essa modularidade é especialmente importante em aplicações de Machine Learning, onde modelos podem ser atualizados com frequência e diferentes partes do pipeline podem evoluir de maneira independente.

Outro ponto relevante é a forma como a plataforma gerencia identidade e acesso aos recursos. Em ambientes de nuvem, serviços precisam se autenticar para acessar outros componentes, como registros de container ou serviços de armazenamento. O Azure oferece mecanismos integrados para gerenciar essas permissões, permitindo que aplicações acessem recursos de forma segura sem a necessidade de expor credenciais diretamente no código. Esse modelo de segurança é essencial para garantir que apenas serviços autorizados possam consumir imagens, acessar dados ou executar operações dentro da infraestrutura.

A combinação desses elementos permite construir arquiteturas completas de Machine Learning na nuvem, nas quais modelos podem ser treinados, armazenados, distribuídos e executados utilizando serviços especializados. A compreensão dessas camadas e de como elas se relacionam é fundamental para projetar soluções eficientes, garantindo que o deploy não apenas funcione, mas também seja sustentável do ponto de vista operacional e econômico.

Aprofundando a análise dos serviços utilizados nesta aula, é possível observar que cada componente do Azure foi projetado para atender a um tipo específico de comportamento de execução, o que reforça a importância de alinhar o serviço ao padrão de workload do modelo de Machine Learning.

**Azure Virtual Machines (VM).** O Azure Virtual Machines (VM) representa o modelo mais tradicional de execução, no qual a aplicação é executada em uma máquina virtual com controle total sobre o ambiente. Nesse cenário, o container é apenas uma camada dentro de uma infraestrutura maior, que inclui sistema operacional, rede, regras de segurança e gerenciamento de recursos. Essa abordagem permite alta flexibilidade, sendo possível configurar desde o tipo de hardware até bibliotecas específicas e drivers necessários para execução do modelo. No entanto, esse nível de controle traz consigo a responsabilidade de gerenciar atualizações, segurança, disponibilidade e escalabilidade, o que aumenta a complexidade operacional.

**Azure Container Apps.** Por outro lado, o Azure Container Apps introduz um modelo mais abstrato, no qual o(a) desenvolvedor(a) não precisa gerenciar diretamente a infraestrutura subjacente. Nesse serviço, o container é executado em um ambiente gerenciado que escala automaticamente conforme a demanda, podendo inclusive reduzir sua execução a zero quando não há requisições. Esse comportamento é particularmente útil para aplicações com carga variável, pois permite otimizar custos ao evitar a manutenção de recursos ociosos. Entretanto, essa abstração reduz o controle sobre o ambiente, limitando a personalização e introduzindo características como latência variável em cenários de baixa utilização.

**Azure Container Apps Jobs.** Para workloads batch, o Azure Container Apps Jobs oferece uma abordagem orientada à execução programada ou sob demanda, na qual containers são executados como tarefas independentes, sem necessidade de manter serviços ativos continuamente. Esse modelo é adequado para o processamento de grandes volumes de dados ou execução de pipelines analíticos, pois permite controlar quando e como os jobs serão executados, além de possibilitar paralelismo e escalabilidade conforme a carga de trabalho. A execução desacoplada do fluxo principal da aplicação garante que tarefas pesadas não impactem diretamente sistemas que exigem baixa latência.

**Azure Blob Storage.** O Azure Blob Storage complementa essa arquitetura ao fornecer um mecanismo eficiente para armazenamento de dados utilizados pelos processos de Machine Learning. Em cenários batch, por exemplo, os dados de entrada podem ser armazenados em um prefixo específico, processados por um job e, em seguida, os resultados são escritos em outro prefixo dentro do mesmo container. Essa organização permite que diferentes etapas do pipeline compartilhem dados de forma consistente, mantendo separação lógica entre entrada e saída. Além disso, o Blob Storage oferece alta durabilidade e escalabilidade, características essenciais para aplicações que lidam com grandes volumes de informação.

**Azure Machine Learning.** Já o Azure Machine Learning representa uma camada mais especializada, voltada para o ciclo de vida completo de modelos de ML. Diferentemente dos demais serviços, que executam containers de forma mais genérica, o Azure ML introduz conceitos específicos como registro de modelos, definição de ambientes de execução e criação de endpoints gerenciados. Essa abordagem permite maior controle sobre versionamento, reprodutibilidade e governança, sendo especialmente útil em cenários nos quais modelos precisam ser monitorados, atualizados e auditados de forma contínua. No entanto, essa estrutura mais completa também exige maior configuração inicial e um entendimento mais aprofundado da plataforma.

Outro ponto importante é a forma como esses serviços podem ser combinados para criar arquiteturas híbridas. Em uma solução real, é comum que diferentes partes do sistema utilizem abordagens distintas de execução. Por exemplo, uma aplicação pode utilizar Container Apps para expor uma API de inferência, enquanto jobs batch processam dados periodicamente e armazenam resultados no Blob Storage. Paralelamente, modelos críticos podem ser gerenciados por meio do Azure ML, garantindo controle de versões e monitoramento contínuo. Essa combinação de serviços permite aproveitar o melhor de cada abordagem, equilibrando controle, custo e desempenho.

Além disso, a integração entre serviços no Azure é fortemente baseada em identidades e permissões, o que reduz a necessidade de configuração manual de credenciais. Serviços podem ser autorizados a acessar registros de containers ou armazenamento de dados por meio de identidades gerenciadas, aumentando a segurança e simplificando a gestão de acesso. Esse modelo é particularmente relevante em ambientes produtivos, onde a exposição de credenciais pode representar um risco significativo.

Esses aspectos demonstram que a escolha dos serviços no Azure não deve ser feita de forma isolada, mas sim como parte de uma estratégia arquitetural mais ampla. Compreender como cada componente se comporta e como eles podem ser combinados permite construir soluções de Machine Learning mais robustas, adaptáveis e alinhadas às necessidades reais de execução em produção.

#### O que você viu nesta aula?
Nesta aula, você aprendeu como implementar estratégias de deploy de modelos de Machine Learning utilizando os serviços do Microsoft Azure, compreendendo como os conceitos estudados anteriormente podem ser aplicados em um ambiente de nuvem diferente, mantendo a mesma lógica arquitetural. O foco esteve na equivalência entre serviços e na forma como diferentes padrões de execução — tempo real, serverless e batch — podem ser construídos utilizando componentes específicos da plataforma.

Ao longo do conteúdo, foram apresentados os principais serviços do Azure envolvidos no deploy de aplicações baseadas em containers. Inicialmente, foi discutido o papel do Azure Container Registry como repositório central de imagens, permitindo armazenar e distribuir aplicações empacotadas em containers. Em seguida, foram exploradas diferentes opções de execução, incluindo máquinas virtuais para aplicações persistentes, serviços serverless para execução sob demanda e jobs programados para processamento em lote.

A aula também abordou o uso do Azure Blob Storage como camada de armazenamento para dados de entrada e saída, destacando sua importância na integração entre diferentes componentes da arquitetura. Além disso, foi apresentado o Azure Machine Learning como uma solução gerenciada para deploy de modelos, permitindo maior controle sobre versionamento, ambientes de execução e exposição de endpoints de inferência.

Na parte prática, você acompanhou a adaptação de projetos já desenvolvidos para execução no Azure, utilizando containers como unidade de deploy. Esse processo evidenciou como aplicações locais podem ser empacotadas, enviadas para um registro de imagens e executadas por diferentes serviços, de acordo com o padrão de workload desejado. Também foi possível observar as etapas necessárias para configurar recursos na nuvem, incluindo autenticação, criação de ambientes, definição de permissões e exposição de endpoints.

#### Referências
- GÉRON, A. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. 3rd ed. Sebastopol: O'Reilly Media, 2022.
- HUYEN, C. *Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications*. Sebastopol: O'Reilly Media, 2022.
- MURPHY, K. P. *Machine Learning: A Probabilistic Perspective*. Cambridge: MIT Press, 2012.

#### Palavras-chave
ACR. VM. Azure ML. Job. Blob.

### Código e comandos
Nenhum bloco de código nesta aula. (Aula conceitual descritiva; a parte prática adapta projetos anteriores para o Azure, mas nenhum comando/código é transcrito no dump.)

### Ferramentas / serviços citados
- Azure Container Registry (ACR)
- Azure Virtual Machines (VM)
- Azure Container Apps
- Azure Container Apps Jobs
- Azure Blob Storage
- Azure Machine Learning (Azure ML)
- Identidades gerenciadas (managed identities)
- Docker

### Aplicabilidade ao Tech Challenge Fase 3
- Fornece o equivalente Azure para a **decisão arquitetural de nuvem** do TC: Container Apps (serverless, escala a zero — bom para custo) vs VM (controle total, latência previsível) para servir o classificador NLP.
- **ACR + Docker + managed identities** oferecem o padrão de build/distribuição de imagem seguro que se encaixa em **CI/CD (GitHub Actions)** sem expor credenciais.
- **Azure ML** (registro/versionamento/endpoints gerenciados) é referência para governança e versionamento do modelo, útil caso o TC opte por serviço gerenciado.

---

## Aula 5 — Deploy de ML na Google: Artifact Registry, Compute Engine VM, Cloud Run, Cloud Run Jobs e Vertex AI
**Arquivo fonte:** `Aula 05 - Deploy de ML na Google - Artifact Registry, Compute Engine Vm, Cloud Run, Cloud Run Jobs e Vertex Ai.pdf` (14 páginas)
**Título na ementa:** `Título no PDF: Deploy de ML na Google: Artifacft Registry, Compute Engine VM, Cloud Run Jobs e Vertex AI | Título na ementa: Deploy de ML na Google: Artifact Registry, Compute Engine VM, Cloud Run Jobs e Vertex AI`

> [NOTA — não é conteúdo FIAP]: no cabeçalho/marca d'água do PDF (pág. 1-14), o serviço aparece grafado como "Artifacft Registry" (typo de OCR). A grafia correta é "Artifact Registry". O nome do arquivo em disco inclui "Cloud Run" além de "Cloud Run Jobs"; o título interno do PDF omite "Cloud Run". Ambos os serviços (Cloud Run e Cloud Run Jobs) são efetivamente abordados no conteúdo.

### Conceitos-chave
- Deploy no **Google Cloud Platform (GCP)** fortemente orientado a **containers** como unidade padrão.
- **Artifact Registry** como repositório de imagens e ponto de integração build↔execução.
- **Compute Engine** (controle total/tempo real), **Cloud Run** (serverless, URL pública, porta dinâmica via variável de ambiente), **Cloud Run Jobs + Cloud Storage** (batch), **Vertex AI** (gerenciado: registro, versionamento, endpoints).
- **Service accounts** para workloads automatizados; **IAM** para controle de acesso.
- Otimização de imagens, latência de acesso a recursos, escolha de região, idempotência em pipelines.

### Conteúdo

#### O que vem por aí?
Nesta aula, será apresentada a aplicação prática das estratégias de deploy de modelos de Machine Learning no ecossistema do Google Cloud Platform, explorando como diferentes serviços da plataforma suportam padrões de execução em tempo real, serverless e batch. O conteúdo parte da mesma base conceitual das aulas anteriores, porém adaptando a implementação para os serviços específicos do Google Cloud, evidenciando suas particularidades operacionais e arquiteturais.

Ao longo da aula, você irá compreender como o uso de containers se mantém como elemento central no processo de deploy, sendo armazenados no Artifact Registry e utilizados por diferentes serviços de execução. Serão exploradas abordagens de deploy em tempo real utilizando máquinas virtuais com o Compute Engine, execução serverless com o Cloud Run e processamento batch por meio do Cloud Run Jobs integrado ao Cloud Storage.

A aula também introduz o uso do Vertex AI como plataforma gerenciada para deploy de modelos, destacando seu papel em cenários mais estruturados de Machine Learning em produção

#### Hands On
A etapa prática consiste na adaptação dos projetos já desenvolvidos anteriormente para o ambiente do Google Cloud Platform, permitindo observar como os mesmos padrões de deploy são implementados utilizando serviços específicos da plataforma. O objetivo é demonstrar, de forma concreta, como containers, modelos de Machine Learning e APIs de inferência se comportam quando executados em diferentes contextos dentro do ecossistema do Google Cloud.

No contexto serverless, o mesmo container será utilizado no Cloud Run, evidenciando a diferença no modelo de execução. A aplicação passa a ser disponibilizada automaticamente por meio de uma URL pública, com escalabilidade gerenciada pela própria plataforma. Ajustes específicos no container, como o uso da variável de ambiente de porta e a exposição correta do serviço HTTP, tornam-se essenciais para garantir o funcionamento adequado nesse tipo de ambiente.

Para o processamento batch, será construído um fluxo baseado em Cloud Run Jobs integrado ao Cloud Storage, no qual os dados de entrada são armazenados em um bucket e processados por um job executado sob demanda. O projeto é ajustado para buscar arquivos diretamente do armazenamento, realizar o processamento completo e persistir os resultados no mesmo ambiente, simulando um pipeline de dados mais próximo de cenários reais.

Adicionalmente, será apresentado o fluxo de deploy utilizando o Vertex AI, demonstrando como um modelo pode ser registrado, versionado e exposto.

#### Saiba Mais
A operacionalização de modelos de Machine Learning em ambientes de nuvem envolve mais do que a simples execução de código em infraestrutura remota. Cada provedor introduz um conjunto próprio de abstrações, serviços e restrições que influenciam diretamente a forma como os modelos são empacotados, distribuídos e executados. No caso do Google Cloud Platform, essa operacionalização é fortemente orientada ao uso de containers como unidade padrão de deploy, o que estabelece um ponto de convergência entre diferentes estratégias de execução, independentemente do tipo de workload.

O uso de containers permite encapsular não apenas o modelo treinado, mas também todo o ambiente necessário para sua execução, incluindo bibliotecas, dependências e código de inferência. Essa abordagem reduz inconsistências entre ambientes e facilita a migração entre diferentes serviços dentro da própria plataforma. No Google Cloud, esse papel é centralizado no Artifact Registry, que atua como repositório de imagens e ponto de integração entre as etapas de build e execução. A padronização do formato de imagens e do fluxo de autenticação permite que um mesmo container seja consumido por múltiplos serviços, reforçando a separação entre a construção da aplicação e sua forma de execução.

**Compute Engine.** Ao analisar a execução em infraestrutura tradicional, observa-se que o Compute Engine oferece um modelo baseado em controle total do ambiente. Nesse cenário, o(a) desenvolvedor(a) define explicitamente o sistema operacional, instala dependências, configura rede e executa o container manualmente. Essa abordagem garante flexibilidade máxima, permitindo adaptações específicas, uso de bibliotecas não suportadas por serviços gerenciados e controle detalhado sobre recursos computacionais. Em contrapartida, esse nível de controle implica maior responsabilidade operacional, incluindo gerenciamento de atualizações, segurança, escalabilidade e disponibilidade da aplicação.

**Cloud Run (serverless).** Quando se avança para um modelo serverless, representado pelo Cloud Run, a responsabilidade sobre a infraestrutura é significativamente reduzida. O container passa a ser executado em um ambiente gerenciado, no qual aspectos como provisionamento de instâncias, balanceamento de carga e escalabilidade são tratados automaticamente pela plataforma. Essa abstração altera a forma como a aplicação deve ser construída, exigindo conformidade com padrões específicos de execução, como a exposição de um servidor HTTP acessível externamente e a utilização de portas dinâmicas definidas pelo ambiente. O comportamento da aplicação deixa de ser controlado diretamente pelo indivíduo desenvolvedor e passa a ser influenciado pelo ciclo de vida gerenciado pelo serviço.

Esse modelo introduz mudanças importantes na forma como recursos são utilizados. Em vez de manter instâncias continuamente ativas, o ambiente serverless aloca recursos apenas durante a execução das requisições, o que reduz custos em cenários de baixa utilização. Entretanto, essa mesma característica pode introduzir variações de desempenho, especialmente em situações em que o serviço precisa inicializar novas instâncias para atender à demanda. A gestão desse comportamento exige compreensão do padrão de uso da aplicação e de como a plataforma reage a variações de carga.

**Cloud Run Jobs + Cloud Storage (batch).** No contexto de processamento batch, o uso de Cloud Run Jobs combinado com Cloud Storage representa uma abordagem orientada a eventos e dados persistidos. Diferentemente da execução contínua ou sob demanda por requisição, o processamento ocorre de forma controlada, geralmente acionado por eventos externos ou por agendamento. O armazenamento em buckets permite desacoplar a origem dos dados da execução do processamento, criando uma separação clara entre ingestão, processamento e persistência de resultados. Essa arquitetura favorece reprocessamento, rastreabilidade e controle de fluxo, sendo amplamente utilizada em pipelines de dados.

A interação com o armazenamento de objetos introduz um modelo diferente de acesso a dados. Em vez de leitura direta de arquivos locais ou de bancos de dados tradicionais, o processamento passa a depender de operações de leitura e escrita em serviços distribuídos, o que exige atenção a aspectos como latência de acesso, consistência de dados e organização lógica dos objetos. A utilização de prefixos para simular diretórios e a necessidade de ordenação de arquivos com base em metadados são exemplos de adaptações necessárias nesse contexto.

**Vertex AI.** Além dessas abordagens, a utilização de plataformas gerenciadas de Machine Learning adiciona uma camada adicional de abstração ao processo de deploy. O Vertex AI introduz conceitos como registro de modelos, endpoints gerenciados e integração com pipelines, deslocando parte da complexidade de infraestrutura para serviços especializados. Nesse modelo, o foco deixa de ser apenas a execução do container e passa a incluir o ciclo de vida completo do modelo, desde sua publicação até sua exposição para consumo por aplicações externas.

A utilização de serviços gerenciados como o Vertex AI altera também a forma como versionamento e governança são tratados dentro do ciclo de vida de Machine Learning. Em vez de depender exclusivamente de práticas externas ou controles manuais, o próprio serviço passa a oferecer mecanismos para registro de versões, controle de deploy e gerenciamento de endpoints. Isso permite que diferentes versões de um mesmo modelo coexistam, facilitando testes comparativos, rollback e evolução incremental da solução sem necessidade de reconfigurar toda a infraestrutura subjacente.

Outro ponto relevante é a forma como a plataforma abstrai o provisionamento de recursos computacionais para inferência. Ao realizar o deploy de um modelo em um endpoint gerenciado, o(a) desenvolvedor(a) define parâmetros como tipo de máquina, quantidade de réplicas e políticas de escalabilidade, enquanto o serviço se encarrega de alocar e gerenciar esses recursos. Essa separação reduz a necessidade de lidar diretamente com instâncias individuais, mas exige compreensão sobre como essas configurações impactam custo e desempenho, especialmente em cenários de alta demanda ou uso contínuo.

A integração com outros serviços do ecossistema também se torna um fator determinante na escolha da estratégia de deploy. O Google Cloud apresenta forte acoplamento entre seus serviços de dados, como armazenamento de objetos e processamento analítico, e as soluções de Machine Learning. Essa integração facilita a construção de pipelines completos, nos quais dados são coletados, processados, utilizados para treinamento e posteriormente consumidos por modelos em produção, mantendo consistência entre as diferentes etapas do fluxo. Essa característica reforça a importância de considerar não apenas o deploy isolado, mas o sistema como um todo.

A padronização do uso de containers ao longo de todas essas abordagens contribui para a portabilidade entre serviços e até mesmo entre provedores de nuvem. O mesmo artefato pode ser executado em uma máquina virtual, em um ambiente serverless ou em uma plataforma gerenciada, com ajustes mínimos relacionados ao contexto de execução. Essa flexibilidade reduz o acoplamento a uma única solução e permite que decisões arquiteturais sejam tomadas com base em requisitos técnicos e operacionais, não apenas em limitações de implementação.

Ao mesmo tempo, essa portabilidade não elimina a necessidade de adaptação ao ambiente de execução. Cada serviço impõe restrições específicas, como limites de tempo, formato de entrada e saída, configuração de rede e modelo de autenticação. Ignorar essas particularidades pode resultar em aplicações que funcionam corretamente em um ambiente local, mas apresentam falhas ou comportamento inesperado quando implantadas na nuvem. A compreensão dessas diferenças é essencial para garantir que o deploy ocorra de forma consistente e previsível.

A escolha entre controle direto da infraestrutura e uso de serviços gerenciados passa, portanto, a ser uma decisão estratégica. Ambientes como máquinas virtuais oferecem flexibilidade máxima e controle detalhado, mas exigem maior esforço operacional. Soluções serverless e plataformas gerenciadas reduzem a complexidade de operação, porém impõem limites e padrões que precisam ser respeitados. O equilíbrio entre esses fatores depende do tipo de aplicação, da frequência de uso, dos requisitos de desempenho e do nível de maturidade da solução.

Esse conjunto de elementos evidencia que o deploy em nuvem não é apenas uma adaptação técnica, mas uma mudança na forma de projetar e operar sistemas de Machine Learning. A escolha dos serviços, a organização dos dados e a estrutura do código passam a ser influenciadas pelas características da plataforma, exigindo uma visão integrada entre desenvolvimento, arquitetura e operação.

**Segurança e controle de acesso.** Outro aspecto que ganha relevância nesse contexto é o modelo de segurança e controle de acesso adotado pela plataforma. No Google Cloud, a gestão de permissões é realizada por meio de políticas de identidade e acesso, nas quais usuários, serviços e aplicações recebem papéis específicos para interagir com recursos como repositórios, buckets e serviços de execução. Esse modelo influencia diretamente o deploy, pois a execução de containers, acesso a imagens no registry e leitura ou escrita em armazenamento dependem de credenciais corretamente configuradas. A utilização de service accounts para workloads automatizados se torna uma prática fundamental, garantindo que aplicações possam acessar recursos de forma segura e auditável, sem depender de credenciais manuais.

**Observabilidade.** A observabilidade também assume um papel central na operação de modelos em produção. A execução distribuída em diferentes serviços exige mecanismos que permitam acompanhar logs, métricas e eventos de forma consolidada. No ambiente do Google Cloud, ferramentas de logging e monitoramento possibilitam visualizar o comportamento das aplicações, identificar falhas e analisar desempenho ao longo do tempo. Em cenários de inferência, isso inclui não apenas métricas técnicas, como tempo de resposta e uso de CPU, mas também indicadores relacionados à qualidade das previsões e à consistência dos dados processados.

Outro ponto importante está relacionado ao gerenciamento de dependências e ao tamanho dos artefatos de deploy. Como os containers são utilizados como unidade padrão de execução, a forma como a imagem é construída impacta diretamente o tempo de inicialização, o consumo de recursos e a eficiência do sistema. Imagens excessivamente grandes ou com dependências desnecessárias podem aumentar o tempo de build, dificultar o transporte entre serviços e afetar o desempenho em ambientes serverless. A otimização dessas imagens, por meio de escolhas adequadas de base, organização de camadas e exclusão de arquivos não utilizados, torna-se uma prática relevante para manter eficiência operacional.

A latência de acesso a recursos externos também deve ser considerada. Em muitos casos, a aplicação depende de leitura de arquivos em armazenamento remoto ou de comunicação com outros serviços. A localização geográfica dos recursos, a escolha da região de execução e a proximidade entre serviços influenciam diretamente o tempo de resposta e o custo de transferência de dados. A definição adequada dessas configurações contribui para reduzir atrasos e evitar custos adicionais associados à movimentação de dados entre regiões.

Além disso, a execução em ambientes distribuídos introduz desafios relacionados à consistência e à sincronização de dados. Em pipelines batch, por exemplo, é necessário garantir que os dados de entrada estejam completos e disponíveis antes do início do processamento, evitando resultados inconsistentes. Em aplicações serverless ou em tempo real, a chegada de dados pode ocorrer de forma concorrente, exigindo mecanismos que tratem possíveis duplicidades, ordem de processamento e idempotência das operações. Esses cuidados são essenciais para manter a integridade dos resultados produzidos pelo modelo.

A evolução contínua das aplicações também depende de práticas adequadas de versionamento e controle de mudanças. Cada alteração no modelo, no código de inferência ou na estrutura dos dados pode impactar o comportamento da aplicação em produção. A manutenção de versões claras dos artefatos, aliada a estratégias de deploy controlado, permite introduzir melhorias sem comprometer a estabilidade do sistema.

#### O que você viu nesta aula?
Ao longo desta aula, foi apresentada a implementação prática de estratégias de deploy de modelos de Machine Learning utilizando serviços do Google Cloud Platform, evidenciando como diferentes padrões de execução são suportados dentro de um mesmo ecossistema. O uso de containers como unidade central de empacotamento foi explorado em conjunto com o Artifact Registry, permitindo a reutilização das aplicações em múltiplos cenários de execução.

Foram analisadas abordagens de deploy em tempo real com o Compute Engine, nas quais o controle sobre a infraestrutura é direto e a aplicação é executada de forma contínua, além de cenários serverless com o Cloud Run, que abstraem a gestão de servidores e permitem escalabilidade automática baseada em demanda. Também foi apresentado o processamento batch utilizando Cloud Run Jobs integrado ao Cloud Storage, destacando a execução controlada de workloads orientados a dados persistidos.

Adicionalmente, foi introduzido o uso do Vertex AI como solução gerenciada para deploy de modelos, abordando conceitos como registro, versionamento e exposição de endpoints de inferência. A análise das diferentes abordagens permitiu compreender como fatores como latência, custo, escalabilidade e complexidade operacional influenciam a escolha da arquitetura mais adequada.

A execução prática dos cenários demonstrou como projetos previamente desenvolvidos podem ser adaptados para diferentes serviços de nuvem, mantendo consistência arquitetural e evidenciando as particularidades de cada ambiente. Esse entendimento reforça a capacidade de projetar soluções portáveis e alinhadas a diferentes provedores, considerando tanto aspectos técnicos quanto operacionais.

#### Referências
- AMERSHI, S. et al. *Software Engineering for Machine Learning: A Case Study*. Proceedings of the 41st International Conference on Software Engineering. 2019. Disponível em: https://www.microsoft.com/en-us/research/wp-content/uploads/2019/03/amershi-icse-2019_Software_Engineering_for_Machine_Learning.pdf. Acesso em: 28 mai. 2026.
- GÉRON, A. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. 3. ed. Sebastopol: O'Reilly Media, 2022.
- GOOGLE CLOUD. *Artifact Registry Documentation*. 2026. Disponível em: https://docs.cloud.google.com/artifact-registry/docs. Acesso em: 28 mai. 2026.
- HUYEN, C. *Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications*. Sebastopol: O'Reilly Media, 2022.
- SCULLEY, D. et al. *Hidden Technical Debt in Machine Learning Systems*. Advances in Neural Information Processing Systems. 2015. Disponível em: https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems. Acesso em: 28 mai. 2026.

#### Palavras-chave
Google Cloud Platform. Artifact Registry. Compute Engine. Cloud Run. Cloud Run Jobs. Vertex AI.

### Código e comandos
Nenhum bloco de código nesta aula. (Aula conceitual descritiva; a parte prática adapta projetos anteriores para o GCP. O único detalhe técnico verbatim mencionado é o requisito de o container Cloud Run usar a variável de ambiente de porta e expor um servidor HTTP — mas nenhum snippet ou comando é transcrito no dump.)

### Ferramentas / serviços citados
- Artifact Registry
- Compute Engine (VM)
- Cloud Run (serverless)
- Cloud Run Jobs
- Cloud Storage (buckets)
- Vertex AI (registro de modelos, endpoints gerenciados, pipelines)
- IAM / service accounts
- Cloud Logging / Cloud Monitoring (observabilidade)
- Docker

### Aplicabilidade ao Tech Challenge Fase 3
- **Cloud Run** (serverless, URL pública, porta dinâmica via env var, escala automática) é forte candidato para servir o **classificador NLP de laudos** com baixo custo — apoia a **decisão arquitetural de nuvem no README**.
- O parágrafo sobre **otimização de imagens** (base adequada, organização de camadas, remoção de arquivos não usados) e **latência de acesso/região** conecta-se diretamente ao requisito de **otimização de latência (ONNX/quantização)** e ao tempo de cold start do serviço.
- **Cloud Run Jobs + Cloud Storage** modela o processamento batch orquestrável por **Airflow**; **service accounts/IAM** e **Cloud Logging/Monitoring** cobrem CI/CD seguro e observabilidade (equivalente GCP ao Prometheus/Grafana).

---

## Aula 6 — Boas Práticas de FinOps (Custos) e Segurança
**Arquivo fonte:** `Aula 06 - Boas Práticas de Finops (Custos) e Segurança.pdf` (14 páginas)
**Título na ementa:** Boas práticas de FinOps (custos) e segurança

### Conceitos-chave
- **FinOps** como disciplina que conecta tecnologia, negócio e finanças; foco em **maximizar valor**, não apenas reduzir gastos.
- Variabilidade de consumo em ML (treinamento × inferência × experimentação); **visibilidade de custos** e **otimização contínua**.
- Decisão arquitetural (tempo real × serverless × batch) determina a distribuição dos custos.
- Segurança: **controle de acesso (princípio do menor privilégio)**, **proteção de dados** (criptografia em repouso e trânsito, mascaramento/anonimização), **segurança do modelo** (rate limiting, autenticação, validação de entrada).
- Ataques específicos a ML: **extração de modelo** e **envenenamento de dados**.
- Custo e segurança compartilham fundamentos: governança, visibilidade e controle. Tags, orçamentos/alertas, retenção de logs, ambientes isolados, automação de governança.

### Conteúdo

#### O que vem por aí?
Nessa aula veremos como a operação de modelos de Machine Learning em nuvem envolve decisões que impactam diretamente custo e segurança, indo além da simples implementação técnica. A discussão parte do entendimento de que sistemas já em produção consomem recursos continuamente, processam dados sensíveis e expõem interfaces de acesso, o que exige controle estruturado sobre uso, acesso e comportamento das aplicações.

Ao longo do material, veremos os conceitos de FinOps aplicados a Machine Learning, destacando como práticas de monitoramento, análise de consumo e otimização contínua permitem alinhar o uso de recursos ao valor gerado para o negócio. A análise considera diferentes tipos de execução, como tempo real, serverless e batch, evidenciando como cada escolha arquitetural influencia diretamente o custo operacional e a previsibilidade financeira dos sistemas.

Também serão explorados os principais aspectos de segurança em ambientes de Machine Learning, incluindo controle de acesso, proteção de dados, monitoramento de uso e riscos associados à exposição de modelos em produção. A abordagem considera tanto a proteção da infraestrutura quanto a integridade dos dados e do próprio modelo, tratando esses elementos como ativos críticos dentro da solução.

A análise conjunta de custo e segurança permite compreender que ambos os temas compartilham os mesmos fundamentos de governança, visibilidade e controle. A forma como os recursos são organizados, monitorados e acessados influencia simultaneamente a eficiência financeira e o nível de proteção do sistema, exigindo uma abordagem integrada na tomada de decisões técnicas.

Também teremos o entendimento para uma análise prática dos projetos já desenvolvidos, permitindo avaliar como diferentes arquiteturas se comportam em termos de custo e risco, e como ajustes técnicos podem melhorar a eficiência e a segurança de soluções de Machine Learning em produção.

#### Hands On
A atividade prática tem como foco a análise dos projetos desenvolvidos anteriormente, utilizando as ferramentas das plataformas de nuvem para avaliar comportamento de custo e exposição a riscos de segurança. Não há criação de novas aplicações ou desenvolvimento de código, o objetivo é observar e interpretar como os recursos já implantados se comportam em um ambiente real de operação.

O processo se inicia com a exploração dos serviços utilizados nos deploys, identificando quais recursos permanecem ativos e quais continuam gerando consumo mesmo sem uso direto. A análise considera máquinas virtuais, serviços serverless, jobs batch e recursos de armazenamento, permitindo compreender como diferentes arquiteturas impactam o custo ao longo do tempo. A partir dessa observação, é possível classificar os tipos de custo envolvidos e identificar pontos de desperdício relacionados a recursos ociosos ou mal dimensionados.

Em seguida, a análise se volta para o comportamento das aplicações em termos de uso. A observação de métricas como número de requisições, tempo de execução e consumo de recursos permite estimar o custo associado à inferência e avaliar se a arquitetura adotada é adequada para o volume de uso esperado. Esse processo incentiva a reflexão sobre decisões arquiteturais e como elas podem ser ajustadas para melhorar a eficiência financeira das soluções.

A atividade também inclui a exploração dos mecanismos de segurança disponíveis nas plataformas, analisando como os recursos estão configurados em relação a controle de acesso, exposição de endpoints e proteção de dados. A verificação de permissões, acessos públicos e ausência de restrições permite identificar vulnerabilidades que podem comprometer tanto a segurança quanto o custo do sistema, especialmente em cenários de uso indevido ou ataques.

Outro ponto relevante da prática é a análise de logs e monitoramento, verificando se existe visibilidade suficiente sobre o comportamento das aplicações. A capacidade de rastrear acessos, identificar padrões anômalos e compreender o uso dos recursos é essencial para detectar problemas, investigar incidentes e tomar decisões baseadas em dados.

A integração entre custo e segurança é observada de forma direta durante a atividade, evidenciando como configurações inadequadas podem gerar impacto financeiro e aumentar a exposição a riscos. A análise permite identificar situações em que a falta de controle de acesso ou de limitação de uso resulta em consumo excessivo de recursos, reforçando a necessidade de governança contínua.

#### Saiba Mais
A operação de sistemas de Machine Learning em ambientes de nuvem introduz uma camada adicional de complexidade que vai além do desenvolvimento e do deploy. Uma vez em produção, esses sistemas passam a consumir recursos continuamente, interagir com dados sensíveis e expor interfaces de acesso, o que exige uma abordagem estruturada para controle de custos e segurança. Nesse contexto, práticas de FinOps e segurança deixam de ser complementares e passam a atuar como pilares centrais da arquitetura.

**FinOps.** O conceito de FinOps surge como uma disciplina que conecta tecnologia, negócio e finanças, permitindo que o consumo de recursos em nuvem seja acompanhado, analisado e otimizado de forma contínua. Diferentemente de abordagens tradicionais de controle de custos, FinOps não se limita à redução de gastos, mas sim busca maximizar o valor gerado a partir do uso da infraestrutura. Em sistemas de Machine Learning, esse desafio se intensifica devido à natureza variável do consumo, que depende de fatores como volume de dados, frequência de inferência e complexidade dos modelos.

A variabilidade de consumo é um dos principais desafios na gestão de custos em ML. Diferentes etapas do ciclo de vida apresentam perfis distintos de uso de recursos. O treinamento de modelos tende a concentrar alto consumo em períodos específicos, enquanto a inferência pode gerar custos contínuos ao longo do tempo. Já a experimentação, comum em fases de desenvolvimento, pode gerar consumo recorrente sem necessariamente produzir valor imediato. Esse comportamento exige visibilidade detalhada sobre como e onde os recursos estão sendo utilizados.

A visibilidade de custos é um elemento essencial para qualquer estratégia de FinOps. Sem mecanismos adequados de monitoramento, torna-se difícil identificar desperdícios, prever gastos ou tomar decisões informadas. A capacidade de associar custos a serviços, aplicações, ambientes e até mesmo a modelos específicos permite uma análise mais precisa e direcionada. Esse nível de granularidade possibilita identificar quais componentes da arquitetura são responsáveis pelo maior consumo e quais podem ser otimizados.

A otimização contínua é outro princípio fundamental. Em ambientes de nuvem, recursos podem ser provisionados rapidamente, mas também podem permanecer ativos sem necessidade, gerando custos desnecessários. Instâncias não utilizadas, endpoints sem tráfego e jobs executados sem necessidade são exemplos comuns de desperdício. A identificação e remoção desses elementos fazem parte de um processo contínuo de revisão da arquitetura e do uso dos recursos.

As decisões arquiteturais desempenham um papel determinante no comportamento de custo. A escolha entre execução em tempo real, serverless ou batch não afeta apenas o desempenho da aplicação, mas também a forma como os custos são distribuídos ao longo do tempo. Modelos em tempo real tendem a gerar custos constantes devido à necessidade de manter recursos ativos, enquanto abordagens serverless permitem um consumo mais alinhado à demanda. Já o processamento batch concentra custos em execuções específicas, sendo mais adequado para cenários que não exigem resposta imediata.

**Segurança.** Além do custo, a segurança se apresenta como um fator crítico na operação de sistemas de Machine Learning. A exposição de modelos por meio de APIs, o armazenamento de dados sensíveis e a execução distribuída em diferentes serviços ampliam a superfície de ataque. Isso exige a implementação de controles que garantam a proteção dos dados, a integridade dos modelos e o uso adequado dos recursos.

O controle de acesso é um dos principais mecanismos de segurança. A definição de permissões adequadas para usuários e serviços reduz o risco de acessos indevidos e limita o impacto de possíveis falhas. O princípio do menor privilégio, no qual cada entidade possui apenas as permissões necessárias para executar sua função, é amplamente utilizado para minimizar riscos. A ausência desse controle pode resultar em acesso indevido a dados, alteração de configurações críticas ou uso não autorizado de recursos.

A proteção de dados também é um elemento central. Dados utilizados em sistemas de Machine Learning frequentemente contêm informações sensíveis, o que exige mecanismos de proteção tanto em repouso quanto em trânsito. A criptografia garante que, mesmo em caso de acesso não autorizado, as informações não possam ser interpretadas sem as chaves apropriadas. Além disso, práticas como mascaramento e anonimização contribuem para reduzir o risco associado ao uso de dados em ambientes de produção.

A segurança dos modelos é outro ponto relevante. Modelos treinados representam ativos valiosos, pois encapsulam conhecimento obtido a partir de dados e processos de treinamento. A exposição desses modelos sem controles adequados pode permitir que terceiros repliquem seu comportamento ou explorem vulnerabilidades. Técnicas como limitação de requisições, autenticação e validação de entrada ajudam a reduzir esses riscos.

Os ataques específicos a sistemas de Machine Learning também devem ser considerados. Ataques de extração de modelo, nos quais um atacante tenta reproduzir o comportamento do modelo por meio de múltiplas consultas, podem comprometer a propriedade intelectual da solução. Ataques de envenenamento de dados podem afetar a qualidade do modelo ao introduzir dados maliciosos durante o treinamento. Esses cenários evidenciam a necessidade de monitoramento constante e de mecanismos de defesa específicos para esse tipo de sistema.

O monitoramento contínuo é essencial tanto para FinOps quanto para segurança. A coleta e análise de métricas e logs permitem identificar comportamentos anômalos, como picos inesperados de consumo ou padrões incomuns de acesso. Esses sinais podem indicar problemas operacionais, falhas de configuração ou até mesmo tentativas de ataque. A capacidade de detectar e responder rapidamente a essas situações é fundamental para manter a estabilidade e a confiabilidade do sistema.

**Relação entre custo e segurança.** A relação entre custo e segurança se torna evidente quando se observa que muitas práticas contribuem simultaneamente para ambos os aspectos. A limitação de requisições, por exemplo, reduz o risco de abuso de APIs e também controla o consumo de recursos. O controle de acesso evita uso indevido e também impede a criação de recursos desnecessários que poderiam gerar custos adicionais. Essa interdependência reforça a necessidade de uma abordagem integrada na gestão de sistemas em nuvem.

A governança emerge como elemento central nesse cenário. A definição de políticas, padrões e processos permite estabelecer diretrizes claras para uso de recursos, controle de acesso e monitoramento. A separação de ambientes, a organização de recursos e a padronização de configurações contribuem para reduzir riscos e facilitar a gestão do sistema como um todo.

A organização dos recursos em ambientes distintos, como desenvolvimento, homologação e produção, contribui diretamente para a redução de riscos e para o controle de custos. Esse isolamento permite que testes e experimentações ocorram sem impactar sistemas críticos, além de facilitar a aplicação de políticas específicas para cada contexto. Em termos de FinOps, essa separação possibilita identificar com maior precisão onde os recursos estão sendo consumidos, evitando que ambientes não produtivos gerem custos desnecessários ao longo do tempo.

O uso de estratégias de identificação, como tags e nomenclaturas padronizadas, desempenha um papel relevante na gestão de recursos. A associação de informações como projeto, ambiente e responsável a cada recurso permite rastrear consumo, aplicar políticas e organizar a infraestrutura de forma consistente. Esse tipo de prática facilita tanto a análise financeira quanto a aplicação de controles de segurança, pois torna mais claro quem está utilizando determinado recurso e com qual finalidade.

A definição de limites e alertas de custo é outro mecanismo importante dentro de uma estratégia de FinOps. A configuração de orçamentos e notificações permite antecipar situações de consumo elevado, possibilitando ações corretivas antes que o impacto financeiro se torne significativo. Esse tipo de monitoramento contínuo reduz a imprevisibilidade dos gastos e incentiva uma cultura de responsabilidade no uso de recursos.

A análise de anomalias também se torna uma prática essencial. Variações inesperadas no consumo de recursos ou no volume de requisições podem indicar problemas de configuração, erros de implementação ou até mesmo atividades maliciosas. A identificação precoce desses comportamentos permite uma resposta mais rápida, reduzindo impactos tanto financeiros quanto operacionais. Em muitos casos, o aumento repentino de custo é um dos primeiros sinais de que algo não está funcionando conforme o esperado.

A retenção de logs deve ser tratada com equilíbrio entre necessidade de rastreabilidade e controle de custos. Manter registros detalhados por longos períodos pode gerar consumo significativo de armazenamento, enquanto a retenção insuficiente pode dificultar a investigação de incidentes. A definição de políticas claras para armazenamento e arquivamento de logs permite atender requisitos de auditoria sem comprometer a eficiência financeira.

A proteção de dados ao longo do pipeline de Machine Learning também exige atenção contínua. Desde a ingestão até a inferência, os dados passam por diferentes etapas e serviços, cada um com suas próprias configurações de acesso e armazenamento. Garantir que essas etapas estejam devidamente protegidas reduz o risco de vazamento e o uso indevido de informações. Esse cuidado se estende também à forma como os dados são manipulados e transformados ao longo do processo.

Outro ponto relevante está na automação de processos de governança. A utilização de mecanismos que desligam recursos automaticamente, limpam dados temporários ou revisam permissões reduz a dependência de ações manuais e minimiza a ocorrência de erros humanos. A automação contribui para manter a consistência das configurações e garante que boas práticas sejam aplicadas de forma contínua, independentemente do crescimento do ambiente.

A maturidade na operação de sistemas de Machine Learning em nuvem está diretamente relacionada à capacidade de integrar práticas de FinOps e segurança no dia a dia das equipes. Isso envolve não apenas o uso de ferramentas, mas também a adoção de uma cultura orientada a responsabilidade compartilhada, na qual indivíduos desenvolvedores, engenheiros de dados e times de operação atuam de forma alinhada. A tomada de decisão passa a considerar não apenas a viabilidade técnica, mas também o impacto financeiro e o nível de risco associado.

À medida que os sistemas evoluem, a complexidade tende a aumentar, exigindo revisões periódicas da arquitetura e das práticas adotadas. O que é adequado em um estágio inicial pode não ser suficiente em um ambiente mais maduro, com maior volume de dados e maior número de usuários. A capacidade de adaptação e melhoria contínua se torna, portanto, um fator crítico para garantir que o sistema permaneça eficiente, seguro e alinhado aos objetivos do negócio.

#### O que você viu nesta aula?
Ao longo desta aula, foi apresentada a importância de práticas de FinOps e segurança na operação de sistemas de Machine Learning em ambientes de nuvem, destacando como decisões técnicas impactam diretamente o custo e o nível de proteção das soluções. A análise considerou o comportamento dos recursos após o deploy, evidenciando que a operação contínua exige monitoramento, controle e governança.

Foram explorados os princípios de FinOps aplicados ao contexto de Machine Learning, com foco em visibilidade de custos, responsabilidade compartilhada e otimização contínua. A relação entre arquitetura e consumo foi analisada, mostrando como diferentes estratégias de execução, como tempo real, serverless e batch, influenciam o custo operacional e a eficiência do sistema.

Também foram abordados os principais aspectos de segurança em ambientes de ML, incluindo controle de acesso, proteção de dados e monitoramento de uso. A exposição de modelos por meio de APIs, o armazenamento de informações sensíveis e a execução distribuída foram discutidos como fatores que ampliam a superfície de ataque e exigem mecanismos de proteção adequados.

A integração entre custo e segurança foi apresentada como um ponto central, demonstrando que práticas como controle de acesso, limitação de requisições e monitoramento contínuo contribuem simultaneamente para reduzir riscos e otimizar o uso de recursos. Essa relação evidencia a necessidade de uma abordagem conjunta na gestão de sistemas em produção.

A análise prática dos projetos permitiu observar como diferentes arquiteturas se comportam em termos de custo e segurança, incentivando a identificação de desperdícios, vulnerabilidades e oportunidades de melhoria. Esse processo reforça a capacidade de avaliar soluções já implantadas e evoluí-las com base em critérios técnicos, financeiros e operacionais.

#### Referências
- AMERSHI, S. et al. *Software Engineering for Machine Learning: A Case Study*. Proceedings of the 41st International Conference on Software Engineering. 2019. Disponível em: https://www.microsoft.com/en-us/research/wp-content/uploads/2019/03/amershi-icse-2019_Software_Engineering_for_Machine_Learning.pdf. Acesso em: 28 mai. 2026.
- FINOPS FOUNDATION. *FinOps Framework*. 2023. Disponível em: https://www.finops.org/framework/. Acesso em: 28 mai. 2026.
- GÉRON, A. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. 3. ed. Sebastopol: O'Reilly Media, 2022.
- GOOGLE CLOUD. *Cloud Cost Management Documentation*. 2026. Disponível em: https://cloud.google.com/cost-management. Acesso em: 28 mai. 2026.
- GOOGLE CLOUD. *Cloud Logging Documentation*. 2026. Disponível em: https://cloud.google.com/logging. Acesso em: 28 mai. 2026.
- GOOGLE CLOUD. *Cloud Monitoring Documentation*. 2026. Disponível em: https://cloud.google.com/monitoring. Acesso em: 28 mai. 2026.
- GOOGLE CLOUD. *IAM Overview*. 2026. Disponível em: https://cloud.google.com/iam/docs/overview. Acesso em: 28 mai. 2026.
- GOOGLE CLOUD. *Security Best Practices*. 2026. Disponível em: https://cloud.google.com/security/best-practices. Acesso em: 28 mai. 2026.
- HUYEN, C. *Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications*. Sebastopol: O'Reilly Media, 2022.
- SCULLEY, D. et al. *Hidden Technical Debt in Machine Learning Systems*. Advances in Neural Information Processing Systems. 2015. Disponível em: https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems. Acesso em: 28 mai. 2026.

#### Palavras-chave
FinOps. Segurança em ML. Governança em Nuvem. Otimização de Custos. Controle de Acesso.

### Código e comandos
Nenhum bloco de código nesta aula. (O Hands On explicitamente não cria aplicações nem desenvolve código: "Não há criação de novas aplicações ou desenvolvimento de código".)

### Ferramentas / serviços citados
- FinOps Framework (FinOps Foundation)
- Cloud Cost Management / orçamentos e alertas de custo
- Cloud Logging / Cloud Monitoring
- IAM (controle de acesso, princípio do menor privilégio)
- Criptografia (repouso e trânsito), mascaramento, anonimização
- Tags / nomenclaturas padronizadas
- Rate limiting (limitação de requisições)

### Aplicabilidade ao Tech Challenge Fase 3
- FinOps (visibilidade de custos, orçamentos/alertas, otimização contínua, escolha tempo real × serverless × batch) sustenta a justificativa de custo na **decisão arquitetural de nuvem no README** do TC.
- Segurança do modelo (rate limiting, autenticação, validação de entrada) e proteção de dados sensíveis (criptografia, anonimização) são requisitos diretos ao expor o **classificador NLP de laudos médicos** (dados de saúde) como API.
- **Cloud Logging/Monitoring** + análise de anomalias mapeiam para o stack **Prometheus/Grafana** e para a detecção de picos anômalos de latência/consumo; automação de governança e ambientes isolados dev/hml/prod conectam-se a **CI/CD (GitHub Actions)** e orquestração via **Airflow**.

---
