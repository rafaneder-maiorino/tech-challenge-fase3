# Latência e Performance em Modelos de Dados Não Estruturados
> Fonte: PDFs FIAP Pós Tech MLET — Fase 3 (Cloud and MLOps)
> Aulas extraídas: 7 de 8 (Aula 2 não disponibilizada pela FIAP — lacuna conhecida)
> Data de extração: 2026-07-23

## Sumário
- [Aula 1 — Fundamentos de Latência e Performance em ML](#aula-1--fundamentos-de-latência-e-performance-em-ml)
- [Aula 2 — Desafios de Performance em NLP e Áudio (LACUNA)](#aula-2--desafios-de-performance-em-nlp-e-áudio)
- [Aula 3 — Desafios de Performance em Visão Computacional](#aula-3--desafios-de-performance-em-visão-computacional)
- [Aula 4 — Otimização de Modelos I – Pruning e Quantização](#aula-4--otimização-de-modelos-i--pruning-e-quantização)
- [Aula 5 — Transfer Learning Eficiente e Adaptação de Modelos](#aula-5--transfer-learning-eficiente-e-adaptação-de-modelos)
- [Aula 6 — Aceleração com Hardware - GPU, TPU e Aceleradores](#aula-6--aceleração-com-hardware---gpu-tpu-e-aceleradores)
- [Aula 7 — Inferência Distribuída e Paralelismo de Modelos](#aula-7--inferência-distribuída-e-paralelismo-de-modelos)
- [Aula 8 — Escalabilidade e Orquestração Multimodal em Produção](#aula-8--escalabilidade-e-orquestração-multimodal-em-produção)

---

## Aula 1 — Fundamentos de Latência e Performance em ML
**Arquivo fonte:** `Aula 01 - Fundamentos de Latência e Performance em ML.pdf` (13 páginas)
**Título na ementa:** `Título no PDF: Fundamentos de Latência e Performance em ML | Título na ementa: Introdução à Latência e Performance em Modelos de ML`

### Conceitos-chave
- Latência, throughput e jitter
- Métricas de percentis: P50, P95, P99, P99.9 e TTFT
- Teorema de Little; Teoria de Filas (notação de Kendall, M/M/1, M/G/1, M/M/c, redes de Jackson)
- Lei de Amdahl e limites de paralelização
- Complexidade computacional: FLOPs e MACs
- Trade-offs: acurácia, latência e custo (fronteira de Pareto)
- Pipeline de inferência end-to-end e pirâmide de otimização

### Conteúdo

**O QUE VEM POR AÍ?**

Você já se perguntou por que alguns modelos de Machine Learning respondem instantaneamente enquanto outros parecem levar uma eternidade? A diferença entre uma aplicação de ML bem-sucedida em produção e uma que frustra usuários frequentemente reside em milissegundos, e compreender esses milissegundos é o que separa engenheiros de ML iniciantes de especialistas em sistemas de alta performance.

Nesta aula, você mergulhará nos fundamentos teóricos e práticos que governam a performance de sistemas de Machine Learning. Exploraremos desde as definições formais de latência, throughput e jitter até modelos matemáticos consagrados como a Lei de Amdahl e o Roofline Model, que permitem prever e otimizar o comportamento de pipelines de inferência. Ao final, você terá as ferramentas conceituais necessárias para identificar gargalos, estabelecer baselines e tomar decisões informadas sobre trade-offs entre acurácia, latência e custo computacional.

**HANDS ON**

Nas videoaulas desta aula, você aprenderá a realizar profiling completo de um pipeline multimodal de inferência, identificando gargalos de latência e estabelecendo baselines de performance. Utilizaremos PyTorch Profiler e torch.utils.benchmark para instrumentar código, coletar métricas de tempo de execução e visualizar a distribuição de latência através de percentis P50, P95 e P99. Demonstraremos também a integração com NVIDIA Nsight Systems para análise de operações GPU e line_profiler para identificação de hotspots em código Python puro.

O repositório completo com todos os notebooks, scripts e datasets está disponível em: github.com/fiap-postech/ml-performance-profiling. Clone o repositório e certifique-se de ter Python 3.11+, PyTorch 2.x e as dependências listadas no requirements.txt instaladas antes de iniciar as videoaulas.

**SAIBA MAIS**

Latência é o tempo entre uma requisição e sua resposta, sendo crítica em ML em produção, onde é preciso equilibrar desempenho e rapidez. Ela inclui componentes como processamento, fila, transmissão e propagação, que juntos determinam a experiência do usuário. Já o throughput mede quantas requisições o sistema processa por unidade de tempo. Embora relacionados, são distintos: um sistema pode ter alta latência individual, mas alto throughput via paralelismo, e vice-versa. Esta relação foi formalizada por Little (1961) em seu teorema fundamental da teoria de filas, expresso como:

$$L = \lambda \times W$$

Onde $L$ representa o número médio de requisições no sistema, $\lambda$ denota a taxa de chegada de requisições e $W$ corresponde ao tempo médio de permanência no sistema. Esta relação elegante permite estimar qualquer uma das três variáveis conhecendo-se as outras duas, sendo particularmente útil para dimensionamento de capacidade em sistemas de inferência distribuídos.

O jitter, por sua vez, caracteriza a variabilidade na latência entre requisições consecutivas, representando um desafio frequentemente subestimado em sistemas de ML. Aplicações de tempo real, como sistemas de recomendação em streaming ou assistentes virtuais, são particularmente sensíveis a jitter elevado, pois a inconsistência na experiência do usuário pode ser tão prejudicial quanto latência média elevada. Dean e Barroso (2013), em seu influente artigo sobre tail latency, demonstraram que em sistemas distribuídos de larga escala, o jitter amplifica-se exponencialmente à medida que aumenta o número de componentes seriais no pipeline de processamento.

**Métricas de Percentis: P50, P95, P99 e TTFT**

A análise de latência vai além da média, usando percentis como P50, P95 e P99 para capturar a distribuição completa dos tempos de resposta, especialmente os extremos que mais impactam o usuário. Enquanto o P50 mostra o comportamento típico, P95 e P99 revelam problemas de performance na cauda. Focar apenas na média pode ocultar falhas relevantes: em sistemas com alto volume, 1% de requisições lentas já afeta milhares de usuários. Por isso, SLOs baseados em percentis são essenciais. Em modelos generativos, destaca-se também o Time To First Token (TTFT), que mede o tempo até a primeira resposta. Um TTFT baixo melhora a percepção do usuário, mesmo que o tempo total de geração seja igual. A Tabela a seguir ilustra a distribuição típica de latência em um sistema de inferência, destacando a distinção entre os diferentes percentis e sua interpretação prática.

| Percentil | Interpretação | Uso Típico em SLOs |
|-----------|---------------|--------------------|
| P50 | Experiência típica do usuário | Monitoramento geral |
| P95 | Experiência degradada ocasional | SLO primário |
| P99 | Piores casos frequentes | SLO crítico |
| P99.9 | Outliers extremos | Debugging |

Tabela 1 - Interpretação dos percentis de latência e seu uso em Service Level Objectives. Fonte: Elaborado pelo autor (2026).

**Complexidade Computacional: FLOPs e MACs**

A complexidade computacional é essencial para entender os limites de performance em ML. Métricas como FLOPs e MACs medem o custo das operações (≈2 FLOPs por MAC). Estudos mostram crescimento exponencial da demanda computacional, superando a Lei de Moore, o que reforça a necessidade de otimização além de hardware mais rápido.

A complexidade assintótica das operações fundamentais em redes neurais varia significativamente entre arquiteturas. Operações convolucionais apresentam complexidade $O(n \cdot k^2 \cdot c_{in} \cdot c_{out})$ para uma entrada de dimensão $n$, kernel de tamanho $k$, e canais de entrada/saída $c_{in}$ e $c_{out}$. Em contraste, a operação de self-attention em Transformers exibe complexidade $O(n^2 \cdot d)$, onde $n$ é o comprimento da sequência e $d$ a dimensionalidade do embedding.

> [NOTA — não é conteúdo FIAP]: no dump original as fórmulas de complexidade vieram com quebras de linha nos subscritos (`c` / `in` / `out` em linhas separadas). Reconstruídas aqui como $c_{in}$ e $c_{out}$ conforme o sentido do texto.

**Lei de Amdahl e Limites de Paralelização**

A Lei de Amdahl, formulada por Gene Amdahl em 1967, estabelece limites teóricos para o speedup obtível através de paralelização, constituindo um dos resultados mais fundamentais da ciência da computação. O teorema afirma que o speedup máximo de um programa está limitado pela fração do código que permanece sequencial, independentemente do número de processadores disponíveis. Matematicamente, o speedup $S$ com $n$ processadores é dado por:

$$S(n) = \frac{1}{(1 - p) + \frac{p}{n}}$$

Onde $p$ representa a fração do código que pode ser paralelizada. No limite quando $n \to \infty$, o speedup converge para $\frac{1}{1-p}$, demonstrando que mesmo frações mínimas de código sequencial impõem tetos significativos à escalabilidade.

Esta lei tem implicações profundas para otimização de pipelines de ML. Considere um pipeline de inferência onde 90% do tempo é gasto em operações paralelizáveis na GPU (convoluções, multiplicações matriciais) e 10% em operações sequenciais na CPU (pré-processamento, pós-processamento). Mesmo com GPUs infinitamente rápidas, o speedup máximo seria limitado a 10x. Hill e Marty (2008) demonstraram que este fenômeno é ainda mais pronunciado em arquiteturas heterogêneas modernas, onde a orquestração entre CPU e aceleradores introduz overheads adicionais.

A identificação de seções sequenciais (gargalos de Amdahl) é, portanto, essencial para priorização de esforços de otimização. Profilers como PyTorch Profiler e NVIDIA Nsight permitem visualizar a decomposição temporal do pipeline, identificando quais componentes limitam o speedup total. Em muitos sistemas de ML, operações aparentemente triviais — como serialização de tensores, transferências host-device, ou sincronizações de barreira — constituem gargalos de Amdahl não óbvios que dominam a latência total.

**Modelo de Filas de Kendall e Teoria de Filas**

A teoria de filas fornece o arcabouço matemático para modelar sistemas de inferência sob carga variável, permitindo prever comportamento de latência em função da taxa de chegada de requisições. A notação de Kendall, expressa como $A/S/c/K/N/D$, caracteriza completamente um sistema de filas especificando: distribuição de chegadas ($A$), distribuição de serviço ($S$), número de servidores ($c$), capacidade do sistema ($K$), tamanho da população ($N$) e disciplina de atendimento ($D$).

Para sistemas de inferência, o modelo M/M/1 (chegadas Poisson, serviço exponencial, servidor único) frequentemente serve como aproximação inicial útil. Neste modelo, o tempo médio de resposta $W$ é dado por:

$$W = \frac{1}{\mu - \lambda}$$

Onde $\mu$ é a taxa de serviço e $\lambda$ a taxa de chegada. Observa-se que quando $\lambda$ aproxima-se de $\mu$ (utilização próxima de 100%), a latência diverge para infinito. Esta característica, conhecida como "hockey stick" pela forma da curva, fundamenta a prática de manter utilização de sistemas de inferência significativamente abaixo de 100% — tipicamente entre 60-80% — para garantir latência estável.

Extensões do modelo básico capturam características realísticas de sistemas de ML. O modelo M/G/1 acomoda distribuições de tempo de serviço não-exponenciais, mais representativas de inferência de redes neurais onde a variabilidade depende do input. Modelos com múltiplos servidores (M/M/c) representam deployments com réplicas horizontais, enquanto redes de Jackson modelam pipelines multi-estágio onde a saída de um servidor alimenta a fila do próximo.

**Trade-offs: Acurácia, Latência e Custo**

A engenharia de ML em produção envolve equilibrar acurácia, latência e custo, que geralmente entram em conflito: modelos maiores tendem a ser mais precisos, porém mais lentos e caros. Esse trade-off define uma fronteira de Pareto entre desempenho e eficiência. Para orientar decisões, usam-se métricas como accuracy por FLOP e comparações com latência ou acurácia fixas. Técnicas de compressão — como quantização, pruning e destilação — ajudam a melhorar esse equilíbrio, reduzindo custo e latência com pouca perda de acurácia.

**Pipeline de Inferência End-to-End**

A latência em ML deve ser analisada ao longo de todo o pipeline de inferência, não apenas no modelo. Etapas como pré e pós-processamento, serialização e transferência de dados podem representar grande parte do tempo total (até 40–60% em sistemas não otimizados). O pipeline inclui desde a recepção da requisição até a resposta ao cliente, e cada etapa adiciona latência e variabilidade, exigindo monitoramento completo. A pirâmide de otimização organiza melhorias por impacto: começar por escolhas arquiteturais e de infraestrutura garante os maiores retornos, deixando ajustes finos de código e hardware para as etapas finais.

**O QUE VOCÊ VIU NESTA AULA?**

Nesta aula, você construiu a base teórica fundamental para análise e otimização de performance em sistemas de Machine Learning. Compreendeu as definições formais de latência, throughput e jitter, assim como a importância de métricas de percentis (P50, P95, P99) para caracterização completa do comportamento de sistema. Explorou modelos matemáticos essenciais — Lei de Amdahl, Roofline Model, Lei de Little e teoria de filas — que permitem diagnosticar gargalos e prever limites de otimização.

Você também examinou a complexidade computacional de diferentes arquiteturas de redes neurais, compreendendo a distinção entre operações compute-bound e memory-bound, e como esta classificação direciona estratégias de otimização. Os trade-offs entre acurácia, latência e custo foram formalizados, preparando-o para decisões informadas de deployment. Na próxima aula, aplicaremos estes fundamentos a técnicas específicas de otimização de inferência.

**REFERÊNCIAS**

BELTAGY, I.; PETERS, M. E.; COHAN, A. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020. DOI: 10.48550/arXiv.2004.05150.

BEYER, B.; JONES, C.; PETOFF, J.; MURPHY, N. R. Site Reliability Engineering: How Google Runs Production Systems. Sebastopol: O'Reilly Media, 2016. ISBN: 978-1491929124.

CHILD, R.; GRAY, S.; RADFORD, A.; SUTSKEVER, I. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509, 2019. DOI: 10.48550/arXiv.1904.10509.

CRANKSHAW, D. et al. Clipper: A low-latency online prediction serving system. Proceedings of NSDI '17, p. 613-627, 2017.

DAO, T. et al. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. Advances in Neural Information Processing Systems (NeurIPS), v. 35, p. 16344

**PALAVRAS-CHAVE**

Modelo ML. Latência. Lei de Little.

### Código e comandos
Nenhum bloco de código nesta aula.

### Ferramentas / serviços citados
- PyTorch Profiler
- torch.utils.benchmark
- NVIDIA Nsight Systems
- line_profiler
- Python 3.11+, PyTorch 2.x
- Repositório: github.com/fiap-postech/ml-performance-profiling

### Aplicabilidade ao Tech Challenge Fase 3
- As métricas de percentis (P50/P95/P99) e o Teorema de Little fornecem a base para definir e medir o SLO de latência do classificador NLP em produção.
- A Lei de Amdahl e a análise do pipeline end-to-end orientam onde otimizar: em NLP boa parte da latência costuma estar em pré-processamento/tokenização (CPU), não só na inferência do modelo.
- Os trade-offs acurácia × latência × custo (fronteira de Pareto) justificam formalmente aplicar quantização/pruning/distilação ao classificador de texto.

---

## Aula 2 — Desafios de Performance em NLP e Áudio
**Arquivo fonte:** — (não disponibilizada pela FIAP no portal)
**Título na ementa:** Desafios de Performance em NLP e Áudio

> **LACUNA CONHECIDA:** Esta aula não foi disponibilizada pela FIAP e portanto não foi extraída. Não é uma falha de extração. Como o Tech Challenge da Fase 3 é um classificador de texto (NLP), esta é a lacuna de maior impacto; a cobertura compensatória está mapeada em `99-MAPA-TECH-CHALLENGE.md`.

---

## Aula 3 — Desafios de Performance em Visão Computacional
**Arquivo fonte:** `Aula 03 - Desafios de Performance em Visão Computacional.pdf` (10 páginas)
**Título na ementa:** Desafios de Performance em Visão Computacional

### Conceitos-chave
- Complexidade computacional em CV: resolução, profundidade e largura
- Trade-offs CNN vs. Vision Transformer (latência, FLOPs)
- Arquiteturas eficientes: MobileNet, EfficientNet, EfficientViT
- Retorno decrescente da resolução
- Detecção em tempo real: YOLOv8/v11, RT-DETR
- Processamento de vídeo: temporal downsampling, keyframe extraction, inferência adaptativa
- Gargalos de pré-processamento (NVIDIA DALI, transforms.v2)

### Conteúdo

**O QUE VEM POR AÍ?**

Imagine processar milhares de imagens por segundo em uma linha de produção industrial ou detectar defeitos em tempo real enquanto um drone sobrevoa uma plantação. A visão computacional em produção exige muito mais do que modelos precisos — exige arquiteturas que equilibrem acurácia e velocidade sob restrições severas de hardware. A escolha entre uma CNN clássica e um Vision Transformer pode significar a diferença entre um sistema funcional e um gargalo operacional.

Nesta aula, você explorará os fundamentos teóricos e práticos que governam a performance em visão computacional. Analisaremos os trade-offs entre CNNs e Vision Transformers sob a ótica de latência e FLOPs, compreenderemos como a resolução de entrada impacta exponencialmente o custo computacional, e examinaremos arquiteturas eficientes como MobileNet, EfficientNet e EfficientViT que definem o estado-da-arte em eficiência. Abordaremos também detecção em tempo real com YOLOv8/v11 e RT-DETR, técnicas de processamento de vídeo otimizado utilizando keyframe extraction e temporal downsampling, pipelines de streaming e aumentações aceleradas com NVIDIA DALI.

**HANDS ON**

Nas videoaulas desta aula, você vai transformar os trade-offs de visão computacional em medições objetivas de latência. O objetivo não é copiar um notebook extenso, mas executar três experimentos curtos e comparar decisões de arquitetura com evidências: tempo por imagem, FPS, uso de memória e impacto de resolução.

1. Benchmark CNN vs. Vision Transformer: execute EfficientNet-B0 e DeiT-Small com o mesmo tensor de entrada, faça aquecimento da GPU e registre média, p95 e p99 de latência. Compare o efeito de batch size 1, precisão FP32 e FP16.
2. Detecção em tempo real com YOLOv8: rode uma versão nano do modelo, ajuste confiança, resolução de entrada e precisão half, e observe como esses parâmetros mudam FPS e qualidade de detecção.
3. Pipeline de vídeo otimizado: aplique redução de resolução, amostragem de frames e extração de keyframes antes da inferência. Quando houver GPU disponível, use NVIDIA DALI ou transforms aceleradas para tirar pré-processamento do gargalo da CPU.

Ao final, registre uma tabela simples com modelo, resolução, hardware, latência média, p95, FPS e principal gargalo observado. Essa tabela será usada para justificar a escolha de arquitetura na discussão técnica.

**SAIBA MAIS**

**Complexidade computacional em visão computacional**

Modelos de visão computacional são sensíveis a três dimensões principais: resolução da imagem, profundidade da rede e largura das camadas. Em CNNs, o custo das convoluções cresce com o tamanho do kernel, número de canais e dimensões espaciais do mapa de atributos. Por isso, dobrar a resolução de entrada tende a aumentar de forma relevante o número de operações e o uso de memória. Em produção, essa relação aparece como aumento de latência, queda de FPS ou necessidade de hardware mais caro.

Vision Transformers seguem outra lógica. A imagem é dividida em patches, e a atenção calcula relações entre esses tokens visuais. Como o custo da atenção cresce com o número de patches, resoluções mais altas podem tornar ViTs caros rapidamente. A escolha entre CNN e ViT, portanto, não é ideológica: depende da restrição de latência, do hardware disponível, do volume de dados e do tipo de tarefa.

**Arquiteturas eficientes para restrições reais**

MobileNet, EfficientNet e EfficientViT foram criadas para buscar melhor equilíbrio entre acurácia e custo. MobileNet reduz operações com convoluções separáveis em profundidade. EfficientNet escala largura, profundidade e resolução de forma balanceada. EfficientViT combina convoluções e mecanismos de atenção mais leves. Em sistemas embarcados ou aplicações de tempo real, arquiteturas menores costumam entregar mais valor operacional do que modelos grandes com pequeno ganho de acurácia.

**Resolução, acurácia e retorno decrescente**

Aumentar a resolução pode revelar detalhes importantes, mas o ganho de acurácia costuma ter retorno decrescente. Uma aplicação médica pode justificar maior custo por imagem; uma câmera de vigilância em tempo real talvez precise sacrificar detalhe para manter 30 FPS. A decisão correta compara impacto de negócio, risco de erro, custo de infraestrutura e experiência do usuário.

**Detecção em tempo real**

Famílias como YOLO e RT-DETR mostram que detecção eficiente depende tanto do modelo quanto do pipeline. Threshold de confiança, tamanho de entrada, pós-processamento, batch, precisão numérica e leitura do vídeo influenciam a latência percebida. Em uma câmera, não basta medir inferência isolada; é preciso medir captura, pré-processamento, inferência, pós-processamento e renderização.

**Processamento de vídeo**

Vídeo é uma sequência redundante. Processar todos os frames com o mesmo custo raramente é necessário. Estratégias como temporal downsampling, keyframe extraction e inferência adaptativa permitem priorizar quadros informativos. Um pipeline maduro pode processar frames leves continuamente e acionar análise mais pesada apenas quando detectar mudança visual relevante.

**Pré-processamento e gargalos invisíveis**

Em muitos sistemas, o modelo não é o único gargalo. Decodificação de imagem, resize, normalização, cópia CPU-GPU e leitura de disco podem consumir tempo suficiente para comprometer o SLA. Bibliotecas como NVIDIA DALI, transforms.v2 e pipelines assíncronos ajudam a mover operações para GPU, reduzir cópias e manter a inferência alimentada.

**Como escolher a arquitetura**

A escolha final deve ser guiada por evidência. Para cada alternativa, registre acurácia, latência média, p95, p99, FPS, memória, custo estimado e complexidade operacional. Em produção, a melhor arquitetura é aquela que atende ao objetivo de negócio dentro do orçamento de latência e custo, com margem para monitoramento, atualização e degradação controlada.

**O QUE VOCÊ VIU NESTA AULA?**

Nesta aula, você explorou os fundamentos teóricos e práticos que governam a performance em sistemas de visão computacional. Compreendeu a complexidade computacional de operações convolucionais e de atenção, analisando como estas diferenças fundamentam os trade-offs entre CNNs e Vision Transformers para diferentes cenários de deployment.

Você examinou arquiteturas eficientes — MobileNet, EfficientNet, EfficientViT — e os princípios de design que as tornam adequadas para aplicações com restrições de latência. A família YOLO e RT-DETR foram apresentados como soluções práticas para detecção em tempo real, com análise quantitativa de suas características de performance.

Por fim, frameworks de aumentação otimizada (DALI, transforms.v2) foram apresentados como ferramentas para eliminar gargalos de pré-processamento. Na próxima aula, aplicaremos estes conceitos a técnicas avançadas de quantização e deployment otimizado.

**REFERÊNCIAS**

CAI, H. et al. EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction. Proceedings of CVPR 2023, p. 17256-17267, 2023. DOI: 10.1109/CVPR52729.2023.01656.

CAI, H. et al. Once-for-All: Train One Network and Specialize it for Efficient Deployment. Proceedings of ICLR 2020, 2020. Disponível em: https://openreview.net/forum?id=HylxE1HKwS. Acesso em: 26 mai. 2026.

DOSOVITSKIY, A. et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. Proceedings of ICLR 2021, 2021. DOI: 10.48550/arXiv.2010.11929.

HE, K.; SUN, J. Convolutional Neural Networks at Constrained Time Cost. Proceedings of CVPR 2015, p. 5353-5360, 2015.

HOWARD, A. G. et al. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. arXiv preprint arXiv:1704.04861, 2017. DOI: 10.48550/arXiv.1704.04861.

JOCHER, G. et al. Ultralytics YOLOv8. GitHub repository, 2023. Disponível em: https://github.com/ultralytics/ultralytics. Acesso em: 26 mai. 2026

LIN, T.-Y. et al. Feature Pyramid Networks for Object Detection. Proceedings of CVPR 2017, p. 936-944, 2017. DOI: 10.1109/CVPR.2017.106.

LIU, Z. et al. A ConvNet for the 2020s. Proceedings of CVPR 2022, p. 11976-11986, 2022. DOI: 10.1109/CVPR52688.2022.01167.

LIU, Z. et al. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. Proceedings of ICCV 2021, p. 10012-10022, 2021. DOI: 10.1109/ICCV48922.2021.00986.

LV, W. et al. RT-DETR: DETRs Beat YOLOs on Real-time Object Detection. arXiv preprint arXiv:2304.08069, 2023. DOI: 10.48550/arXiv.2304.08069.

NVIDIA. DALI Documentation: Performance Tuning Guide. NVIDIA Developer Documentation, 2023. Disponível em: https://docs.nvidia.com/deeplearning/dali/user-guide/docs/. Acesso em: 26 mai. 2026.

REDMON, J. et al. You Only Look Once: Unified, Real-Time Object Detection. Proceedings of CVPR 2016, p. 779-788, 2016. DOI: 10.1109/CVPR.2016.91.

**PALAVRAS-CHAVE**

Visão computacional. MobileNet, EfficientNet, EfficientViT.

### Código e comandos
Nenhum bloco de código nesta aula.

### Ferramentas / serviços citados
- EfficientNet-B0, DeiT-Small, MobileNet, EfficientViT
- YOLOv8/v11 (Ultralytics), RT-DETR
- NVIDIA DALI, transforms.v2
- Precisões FP32/FP16 (half)

### Aplicabilidade ao Tech Challenge Fase 3
- O princípio de que pré-processamento pode ser o gargalo real (não o modelo) transfere-se diretamente ao classificador NLP: tokenização e I/O podem dominar a latência, e devem ser medidos separadamente da inferência.
- A disciplina de guiar a escolha de arquitetura por evidência (registrar latência média, p95, p99, memória, custo) é o mesmo método a aplicar ao comparar variantes do classificador de texto.

---

## Aula 4 — Otimização de Modelos I – Pruning e Quantização
**Arquivo fonte:** `Aula 04 - Otimização de Modelos - Pruning e Quantização.pdf` (12 páginas)
**Título na ementa:** Otimização de Modelos I – Pruning e Quantização

### Conceitos-chave
- Três pilares da compressão: pruning, quantização, knowledge distillation
- Redundância em redes neurais (Denil et al., 2013)
- Lottery Ticket Hypothesis e Iterative Magnitude Pruning (IMP)
- Critérios de importância: magnitude-based, gradient-based, Taylor expansion-based
- Pruning estruturado vs. não-estruturado
- Quantização INT8; compressões de 10-50x

### Conteúdo

**O QUE VEM POR AÍ?**

Você já se perguntou como empresas como Google, Meta e Microsoft conseguem executar modelos de Deep Learning com bilhões de parâmetros em dispositivos móveis, ou produzir respostas em milissegundos em serviços de alta escala? A resposta está nas técnicas de compressão de redes neurais que transformam modelos pesados em versões otimizadas, mantendo a qualidade das predições enquanto reduzem drasticamente o consumo de memória e o tempo de inferência.

Nesta aula, você mergulhará no universo da otimização de modelos, compreendendo os fundamentos teóricos e práticos que permitem essa "mágica" da engenharia de Machine Learning.

**HANDS ON**

Nesta aula prática, você implementará um pipeline completo de compressão de modelos neurais, aplicando técnicas de pruning estruturado e quantização INT8 em modelos reais de NLP e visão computacional. Nas videoaulas correspondentes, demonstraremos passo a passo a configuração do ambiente, a aplicação de cada técnica de compressão e a avaliação do impacto em métricas de latência, memória e acurácia. Utilizaremos PyTorch como framework principal, complementado por ferramentas especializadas como Intel Neural Compressor e Hugging Face Optimum.

O código completo desta aula, incluindo notebooks interativos, scripts de treinamento e configurações de deployment, está disponível no repositório GitHub da disciplina. Recomendamos que você clone o repositório antes de assistir às videoaulas para acompanhar os exemplos em tempo real. Certifique-se de ter Python 3.11+, PyTorch 2.0+ e as dependências listadas no arquivo requirements.txt instaladas em seu ambiente.

Código completo disponível no GitHub: github.com/fiap-postech/model-optimization

**SAIBA MAIS**

**Fundamentos da Compressão de Redes Neurais**

A compressão de redes neurais emerge como resposta a um paradoxo fundamental do Deep Learning moderno: enquanto modelos cada vez maiores demonstram capacidades impressionantes, sua implantação em ambientes de produção enfrenta restrições severas de latência, memória e consumo energético. Segundo LeCun, Bengio e Hinton (2015), o sucesso do Deep Learning está intrinsecamente ligado à capacidade de escalar modelos, mas essa escalabilidade conflita diretamente com as demandas de sistemas em tempo real e dispositivos com recursos limitados.

A redundância em redes neurais profundas foi identificada empiricamente por Denil et al. (2013), que demonstraram que os pesos de uma rede podem ser preditos com alta precisão a partir de um subconjunto reduzido, sugerindo que a maior parte dos parâmetros carrega informação redundante. Este insight fundamental pavimentou o caminho para técnicas de compressão que exploram sistematicamente essa redundância, removendo parâmetros desnecessários ou reduzindo sua precisão numérica sem comprometer significativamente a capacidade de generalização do modelo.

O framework conceitual da compressão moderna organiza-se em três pilares complementares. O primeiro pilar, pruning, remove conexões ou estruturas inteiras consideradas não essenciais para a função aprendida pelo modelo. O segundo pilar, quantização, reduz a precisão numérica dos pesos e ativações, tipicamente de ponto flutuante de 32 bits para inteiros de 8 ou 4 bits. O terceiro pilar, knowledge distillation, transfere o conhecimento de um modelo grande (teacher) para um modelo menor (student), permitindo que arquiteturas compactas alcancem desempenho próximo ao de modelos massivos.

**A Hipótese da Lottery Ticket: Fundamentos Teóricos**

A Lottery Ticket Hypothesis, proposta por Frankle e Carlin (2019), representa uma das contribuições teóricas mais significativas para a compreensão de por que redes neurais podem ser comprimidas. A hipótese postula que uma rede neural densa, inicializada aleatoriamente, contém uma sub-rede esparsa que, quando treinada isoladamente desde a mesma inicialização, pode atingir acurácia comparável à rede completa em um número similar ou menor de iterações. Estas sub-redes são denominadas "winning tickets" (bilhetes premiados), em alusão à natureza probabilística de sua existência.

Formalmente, considere uma rede neural com parâmetros θ₀ inicializados aleatoriamente segundo uma distribuição D. Após treinamento por t iterações, obtemos parâmetros θₜ com acurácia a. A hipótese afirma que existe uma máscara m ∈ {0, 1}ⁿ tal que a sub-rede definida por m ⊙ θ₀, quando treinada isoladamente, atinge acurácia a' ≥ a em t' ≤ t iterações, onde ⊙ denota o produto elemento a elemento e ||m||₀ << n representa a condição de esparsidade.

A identificação de winning tickets segue um procedimento iterativo denominado Iterative Magnitude Pruning (IMP). O processo inicia com o treinamento completo da rede densa, seguido pela remoção de uma fração p dos pesos com menor magnitude absoluta. Crucialmente, os pesos remanescentes são reinicializados para seus valores originais θ₀, não para os valores treinados θₜ. Este procedimento é repetido iterativamente até atingir a esparsidade desejada. Frankle e Carlin (2019) demonstraram empiricamente que este método identifica sub-redes com 10-20% dos parâmetros originais que igualam ou superam a acurácia da rede densa.

A implicação prática desta descoberta é profunda: redes neurais são dramaticamente sobreparametrizadas não porque precisam de todos os parâmetros para representar a função alvo, mas sim porque facilita a otimização durante o treinamento. Uma vez identificada a estrutura relevante (o winning ticket), a rede esparsa pode ser treinada ou implantada com recursos computacionais substancialmente reduzidos.

Assista ao Vídeo 2 para uma visualização detalhada do processo de identificação de winning tickets e análise da evolução da esparsidade durante o treinamento iterativo.

**Pruning: Critérios de Importância e Estratégias**

O pruning de redes neurais fundamenta-se na identificação e remoção de parâmetros considerados menos importantes para a função aprendida. A definição formal de "importância" varia conforme o critério adotado, e a escolha do critério impacta diretamente a eficácia da compressão e a preservação da acurácia. Han et al. (2015), no trabalho seminal "Learning Both Weights and Connections for Efficient Neural Networks", estabeleceram o paradigma magnitude-based pruning, que utiliza a magnitude absoluta dos pesos como proxy para importância.

O critério magnitude-based assume que pesos com valores próximos a zero contribuem minimamente para as ativações das camadas subsequentes e, portanto, podem ser removidos com impacto limitado. Matematicamente, dado um limiar τ, um peso wᵢⱼ é removido se |wᵢⱼ| < τ. O limiar pode ser definido globalmente (mesmo τ para toda a rede) ou localmente (τ específico por camada), com a abordagem local geralmente preservando melhor a acurácia ao evitar que camadas críticas sejam excessivamente podadas.

Critérios alternativos foram propostos para capturar noções mais sofisticadas de importância. O gradient-based pruning avalia a importância de um peso pela magnitude do gradiente da função de perda em relação a ele, sob a premissa de que pesos com gradientes pequenos têm pouca influência na otimização e podem ser removidos. O Taylor expansion-based pruning, proposto por Molchanov et al. (2017), aproxima a mudança na função de perda causada pela remoção de um peso usando expansão de Taylor de primeira ordem:

> [DIAGRAMA/FÓRMULA ilegível no dump — pág. 7]: A fórmula da expansão de Taylor de primeira ordem foi referida no texto mas não transcrita legivelmente no dump. O texto que a segue: "Onde ℒ representa a função de perda e $w_i$ o peso candidato à remoção."

Onde ℒ representa a função de perda e $w_i$ o peso candidato à remoção. Esta formulação captura tanto a magnitude do peso quanto sua sensibilidade, oferecendo um critério mais informativo que magnitude isolada.

A distinção entre pruning estruturado e não-estruturado é fundamental para aplicações práticas. O pruning não-estruturado remove pesos individuais arbitrariamente, resultando em matrizes esparsas irregulares. Embora atinja taxas de compressão elevadas (até 90% de esparsidade), os ganhos de velocidade reais são limitados pela falta de suporte eficiente para esparsidade irregular em hardware convencional. O pruning estruturado, em contraste, remove estruturas completas como filtros em CNNs ou cabeças de atenção em Transformers, resultando em arquiteturas menores que podem ser executadas eficientemente em hardware padrão sem necessidade de suporte especializado para esparsidade.

**O QUE VOCÊ VIU NESTA AULA?**

Nesta aula, você explorou os fundamentos teóricos e práticos da compressão de redes neurais, compreendendo como pruning, quantização e knowledge distillation transformam modelos pesados em versões otimizadas para produção. Você aprendeu sobre a Hipótese da Lottery Ticket e como ela fundamenta a eficácia do pruning, dominou os critérios de importância para seleção de parâmetros a serem removidos, e compreendeu a teoria de quantização uniforme com suas estratégias de calibração.

Por fim, você viu como combinar estas técnicas em pipelines integrados que alcançam compressões de 10-50x com impacto mínimo na acurácia, habilitando deployment eficiente em ambientes de produção com restrições de recursos.

**REFERÊNCIAS**

BENGIO, Yoshua; LÉONARD, Nicholas; COURVILLE, Aaron. Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation. arXiv preprint arXiv:1308.3432, 2013. DOI: 10.48550/arXiv.1308.3432.

BLALOCK, Davis et al. What is the State of Neural Network Pruning? In: Proceedings of Machine Learning and Systems (MLSys), v. 2, p. 129-146, 2020. DOI: 10.48550/arXiv.2003.03033.

CAI, Han et al. Once-for-All: Train One Network and Specialize it for Efficient Deployment. In: International Conference on Learning Representations (ICLR), 2020. DOI: 10.48550/arXiv.1908.09791.

DENIL, Misha et al. Predicting Parameters in Deep Learning. In: Advances in Neural Information Processing Systems (NeurIPS), v. 26, 2013.

**PALAVRAS-CHAVE**

Deep Learning. Lottery Ticket. Pruning.

### Código e comandos
Nenhum bloco de código nesta aula.

### Ferramentas / serviços citados
- PyTorch (framework principal), Python 3.11+, PyTorch 2.0+
- Intel Neural Compressor
- Hugging Face Optimum
- Repositório: github.com/fiap-postech/model-optimization

### Aplicabilidade ao Tech Challenge Fase 3
- Pruning estruturado e quantização INT8 são diretamente aplicáveis ao classificador NLP para reduzir latência e memória em produção (a própria aula cita modelos de NLP como alvo).
- Knowledge distillation (teacher→student) permite obter um classificador de texto compacto com acurácia próxima à de um modelo grande — caminho central para o requisito de otimização de latência do TC.
- A Lottery Ticket Hypothesis / IMP fundamenta teoricamente por que é possível comprimir o modelo do TC em 10-50x sem perda relevante de acurácia.

---

## Aula 5 — Transfer Learning Eficiente e Adaptação de Modelos
**Arquivo fonte:** `Aula 05 - Transfer Learning Eficiente e Adaptação de Modelos.pdf` (12 páginas)
**Título na ementa:** `Título no PDF: Transfer Learning Eficiente e Adaptação de Modelos | Título na ementa: Otimização de Modelos II – Transfer Learning`

### Conceitos-chave
- Fundamentos do transfer learning (Pan e Yang, 2010)
- Desafios do fine-tuning completo; catastrophic forgetting
- Hipótese de subespaços de baixa dimensionalidade (dimensão intrínseca)
- Low-Rank Adaptation (LoRA): W = W₀ + BA
- QLoRA e formato NormalFloat4 (NF4)
- PEFT (Parameter-Efficient Fine-Tuning)

### Conteúdo

**O QUE VEM POR AÍ?**

Imagine poder adaptar um modelo de linguagem com bilhões de parâmetros utilizando apenas uma fração dos recursos computacionais tradicionalmente necessários. Essa possibilidade, que há poucos anos parecia distante, tornou-se realidade com o advento de técnicas como Low-Rank Adaptation (LoRA) e QLoRA, que revolucionaram a forma como realizamos fine-tuning de grandes modelos.

Nesta aula, você mergulhará no universo do transfer learning eficiente, compreendendo não apenas como essas técnicas funcionam, mas também por que elas representam um marco na democratização do acesso a Large Language Models.

**HANDS ON**

Nesta aula prática, você implementará um pipeline completo de fine-tuning eficiente utilizando QLoRA para adaptar o modelo LLaMA-2 7B a uma tarefa específica de sumarização de documentos jurídicos. Nas videoaulas correspondentes, demonstraremos passo a passo como configurar o ambiente, aplicar quantização de 4 bits com bitsandbytes, treinar adaptadores LoRA e, finalmente, exportar o modelo para ONNX visando inferência otimizada.

O código completo está disponível no repositório GitHub do curso. Recomendamos que você clone o repositório antes de assistir às videoaulas para acompanhar a implementação em seu próprio ambiente. Certifique-se de ter Python 3.11+, PyTorch 2.0+, e uma GPU com pelo menos 8GB de VRAM para executar os exemplos. As bibliotecas Hugging Face PEFT, bitsandbytes e ONNX Runtime serão utilizadas extensivamente.

Código completo disponível no GitHub: github.com/fiap-postech/transfer-learning-eficiente

**SAIBA MAIS**

**Fundamentos do Transfer Learning**

O conceito de transfer learning, formalizado por Pan e Yang (2010), fundamenta-se na premissa de que o conhecimento adquirido ao resolver uma tarefa pode ser transferido para facilitar a solução de tarefas relacionadas. Formalmente, dado um domínio fonte D_S com tarefa T_S e um domínio alvo D_T com tarefa T_T, o objetivo do transfer learning é melhorar a função de aprendizado f_T no domínio alvo utilizando conhecimento extraído de D_S e T_S. Esta abordagem contrasta com o paradigma tradicional de Machine Learning, que assume que dados de treinamento e teste são independentes e identicamente distribuídos (i.i.d.).

No contexto de Deep Learning, o transfer learning manifesta-se predominantemente através do aproveitamento de representações aprendidas em camadas intermediárias de redes neurais profundas. Estudos seminais demonstraram que as primeiras camadas de redes convolucionais aprendem features genéricas como detectores de bordas e texturas, enquanto camadas mais profundas especializam-se em features específicas do domínio (Yosinski et al., 2014). Esta hierarquia de abstração permite que modelos pré-treinados em datasets massivos como ImageNet sirvam como inicializadores robustos para tarefas downstream com dados limitados.

A emergência de Large Language Models (LLMs) amplificou dramaticamente a importância do transfer learning. Modelos como BERT (Devlin et al., 2019), GPT-3 (Brown et al., 2020) e LLaMA (Touvron et al., 2023) são pré-treinados em corpora textuais de escala sem precedentes, desenvolvendo representações linguísticas que capturam nuances semânticas, sintáticas e pragmáticas. O paradigma de pré-treinamento seguido de fine-tuning tornou-se o padrão de facto para praticamente todas as tarefas de processamento de linguagem natural, desde classificação de sentimentos até geração de código.

**O Desafio do Fine-Tuning Completo**

Apesar de sua eficácia comprovada, o fine-tuning completo de modelos de grande escala apresenta desafios computacionais substanciais. Considere um modelo como o LLaMA-2 70B: com 70 bilhões de parâmetros em precisão float32, apenas o armazenamento dos pesos requer aproximadamente 280GB de memória. Durante o treinamento, os requisitos multiplicam-se devido à necessidade de manter gradientes, estados do otimizador (como momentos em Adam) e ativações intermediárias para backpropagation. Estima-se que o fine-tuning completo de um modelo desta escala demande múltiplas GPUs A100 de 80GB em configuração paralela.

Além das restrições de memória, o fine-tuning completo apresenta riscos de catastrophic forgetting, fenômeno no qual a adaptação a novas tarefas degrada o desempenho em tarefas previamente aprendidas (Kirkpatrick et al., 2017). Este problema é particularmente pronunciado quando os dados de fine-tuning são limitados ou significativamente diferentes do corpus de pré-treinamento. A necessidade de balancear adaptação e preservação de conhecimento motivou o desenvolvimento de técnicas que modificam seletivamente subconjuntos de parâmetros.

**A Hipótese de Subespaços de Baixa Dimensionalidade**

Um insight fundamental que pavimentou o caminho para técnicas de PEFT foi a descoberta de que o fine-tuning efetivo de modelos pré-treinados ocorre em subespaços de dimensionalidade intrínseca muito menor que o espaço total de parâmetros. Aghajanyan, Gupta e Shrivastava (2021) demonstraram empiricamente que é possível restringir atualizações de parâmetros a subespaços aleatórios de dimensão d ≪ D (onde D é o número total de parâmetros) mantendo 90% do desempenho do fine-tuning completo.

A dimensão intrínseca d de uma tarefa de fine-tuning pode ser formalmente definida como a menor dimensão de um subespaço no qual a otimização ainda converge para uma solução de qualidade comparável ao espaço completo. Experimentos revelaram que para muitas tarefas de NLP, esta dimensão é de apenas algumas centenas a poucos milhares, mesmo para modelos com bilhões de parâmetros (Li et al., 2018). Esta observação sugere que a maior parte do espaço de parâmetros é redundante para adaptação task-specific, fundamentando teoricamente abordagens como LoRA.

**Low-Rank Adaptation (LoRA): Teoria e Formalismo**

Low-Rank Adaptation, introduzido por Hu et al. (2022), operacionaliza a hipótese de baixa dimensionalidade através de decomposição matricial. Para uma matriz de pesos W₀ ∈ ℝ^(d×k), em uma camada do modelo pré-treinado, LoRA injeta uma atualização de baixo rank parametrizada como:

$$\Delta W = BA$$

> [DIAGRAMA/FÓRMULA no dump — pág. 7]: A equação de atualização de baixo rank foi referida no texto mas não transcrita explicitamente no dump. Reconstruída como ΔW = BA a partir do texto seguinte ("Onde B ∈ ℝ^(d×r) e A ∈ ℝ^(r×k)... W = W₀ + BA").

Onde B ∈ ℝ^(d×r) e A ∈ ℝ^(r×k) são matrizes treináveis de rank r ≪ min(d, k). O rank r é um hiperparâmetro que controla o trade-off entre capacidade expressiva e eficiência computacional. Na prática, valores de r entre 4 e 64 demonstram desempenho competitivo com fine-tuning completo em diversas tarefas.

A inicialização desempenha papel crucial na convergência de LoRA. Convencionalmente, a matriz A é inicializada com distribuição gaussiana e B com zeros, garantindo que ΔW = BA = 0 no início do treinamento. Esta estratégia assegura que o modelo inicie exatamente no ponto do modelo pré-treinado, preservando conhecimento acumulado durante pré-treinamento. Durante o treinamento, um fator de escalonamento α/r é aplicado à atualização, onde α é um hiperparâmetro (tipicamente igual a r ou 2r) que modula a magnitude das atualizações.

A elegância matemática de LoRA reside em sua fusão sem overhead durante inferência. Após o treinamento, as matrizes BA podem ser mescladas com W₀ através de simples adição matricial: W = W₀ + BA. O modelo resultante possui arquitetura idêntica ao original, sem camadas adicionais ou modificações estruturais. Esta propriedade distingue LoRA de abordagens como Adapters, que introduzem latência adicional devido a camadas intermediárias.

**QLoRA: Quantização para Democratização**

QLoRA, proposto por Dettmers et al. (2023), combina LoRA com técnicas avançadas de quantização para reduzir drasticamente os requisitos de memória durante fine-tuning. A inovação central é a quantização dos pesos do modelo base para 4 bits utilizando o formato NormalFloat4 (NF4), especificamente projetado para dados com distribuição normal, característica observada empiricamente em pesos de redes neurais após treinamento.

O formato NF4 estabelece níveis de quantização ótimos assumindo uma distribuição normal padrão N(0, 1) para os valores a serem quantizados. Formalmente, os 16 níveis de quantização (2^4) são definidos como os quantis que dividem a distribuição normal em 16 regiões de igual probabilidade. Esta abordagem teoricamente informada resulta em menor erro de quantização comparado a formatos uniformemente espaçados como INT4.

**O QUE VOCÊ VIU NESTA AULA?**

Nesta aula, você explorou o universo do transfer learning eficiente, compreendendo por que técnicas de Parameter-Efficient Fine-Tuning como LoRA e QLoRA representam um marco na democratização de acesso a Large Language Models. Você aprendeu que o fine-tuning efetivo ocorre em subespaços de baixa dimensionalidade, fundamentando matematicamente a decomposição W = W₀ + BA que permite adaptar modelos de bilhões de parâmetros treinando apenas frações percentuais de seus pesos.

**REFERÊNCIAS**

AGHAJANYAN, Armen; GUPTA, Sonal; SHRIVASTAVA, Luke. Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning. In: Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL), 2021. p. 7319-7328. DOI: 10.18653/v1/2021.acl-long.568.

BROWN, Tom B. et al. Language Models are Few-Shot Learners. In: Advances in Neural Information Processing Systems (NeurIPS), v. 33, p. 1877-1901, 2020. DOI: 10.48550/arXiv.2005.14165.

DETTMERS, Tim et al. QLoRA: Efficient Finetuning of Quantized LLMs. In: Advances in Neural Information Processing Systems (NeurIPS), v. 36, 2023. DOI: 10.48550/arXiv.2305.14314.

**PALAVRAS-CHAVE**

Transfer Learning. LoRA. QLoRA.

### Código e comandos
Nenhum bloco de código nesta aula.

### Ferramentas / serviços citados
- LLaMA-2 7B / 70B, BERT, GPT-3, LLaMA
- QLoRA, LoRA, formato NF4 (NormalFloat4)
- Hugging Face PEFT, bitsandbytes, ONNX Runtime
- Python 3.11+, PyTorch 2.0+, GPU ≥ 8GB VRAM, A100 80GB
- Repositório: github.com/fiap-postech/transfer-learning-eficiente

### Aplicabilidade ao Tech Challenge Fase 3
- Transfer learning (pré-treino + fine-tuning) é exatamente o paradigma do classificador NLP do TC: partir de um modelo pré-treinado (ex.: BERT) e adaptá-lo à tarefa de classificação de texto.
- LoRA/QLoRA permitem fine-tunar o classificador com pouca VRAM e sem catastrophic forgetting; como LoRA se funde a W₀ (W = W₀ + BA), não adiciona latência de inferência — alinhado ao requisito de otimização de latência.
- A aula cita explicitamente exportar o modelo para ONNX para inferência otimizada, caminho direto para acelerar a inferência do classificador em produção.

---

## Aula 6 — Aceleração com Hardware - GPU, TPU e Aceleradores
**Arquivo fonte:** `Aula 06 - Aceleração com Hardware - GPU, TPU e Aceleradores.pdf` (13 páginas)
**Título na ementa:** Aceleração com Hardware (GPU/TPU)

### Conceitos-chave
- Arquitetura de GPUs NVIDIA: Streaming Multiprocessors (SMs), CUDA cores, Tensor Cores
- Modelo de execução SIMT, warps, latency hiding
- Hierarquia de memória: registradores, shared memory/L1, L2, HBM; coalesced memory access
- Modelo Roofline; intensidade aritmética; compute-bound vs. memory-bound
- Mixed Precision Training: FP16, BF16, TF32; loss scaling
- Tensor Cores (MMA / WGMMA) e evolução por geração
- Aceleradores alternativos: TPU, NPU, AWS Inferentia

### Conteúdo

**O QUE VEM POR AÍ?**

Você já parou para pensar por que treinar um modelo de Deep Learning em uma CPU pode levar dias, enquanto uma GPU moderna completa a mesma tarefa em horas ou até minutos? A resposta está na arquitetura fundamentalmente diferente desses processadores e na forma como exploram o paralelismo massivo inerente às operações de redes neurais. Nesta aula, vamos desvendar os segredos por trás da aceleração de hardware que transformou o campo de Machine Learning.

Exploraremos desde a arquitetura interna de GPUs NVIDIA — com seus Tensor Cores e hierarquia de memória sofisticada — até técnicas avançadas como Mixed Precision Training, que permite dobrar a velocidade de treinamento mantendo a precisão dos modelos. Você também conhecerá alternativas como TPUs, NPUs e AWS Inferentia, compreendendo quando cada acelerador é a escolha ideal. Ao final, estará preparado(a) para realizar profiling detalhado de seus workloads e tomar decisões informadas sobre infraestrutura de hardware.

**HANDS ON**

Nas videoaulas desta aula, você aprenderá a implementar Mixed Precision Training utilizando PyTorch e a realizar profiling detalhado de modelos de Deep Learning em GPU. Demonstraremos como identificar gargalos de performance usando NVIDIA Nsight Systems e PyTorch Profiler, além de aplicar técnicas de otimização que podem reduzir o tempo de treinamento em até 50% sem comprometer a qualidade do modelo. Trabalharemos com um Vision Transformer como caso de estudo, analisando operações compute-bound e memory-bound.

O código completo, incluindo notebooks Jupyter com experimentos reproduzíveis, está disponível no repositório GitHub do curso. Certifique-se de ter acesso a uma GPU NVIDIA (local ou via cloud como Azure NC/ND series) com drivers atualizados, CUDA 11.8+ e PyTorch 2.x instalados. Recomendamos clonar o repositório antes de assistir às videoaulas para acompanhar os exemplos em tempo real.

Repositório completo: github.com/fiap-postech/gpu-acceleration-ml

**SAIBA MAIS**

**Fundamentos da Arquitetura de GPUs para Machine Learning**

A evolução das Graphics Processing Units (GPUs) de dispositivos especializados em renderização gráfica para aceleradores de propósito geral transformou fundamentalmente o campo de Machine Learning. Esta transformação decorre de uma convergência entre as demandas computacionais de redes neurais profundas — caracterizadas por operações matriciais massivamente paralelas — e a arquitetura intrinsecamente paralela das GPUs modernas (Nickolls & Dally, 2010).

As GPUs NVIDIA contemporâneas, particularmente as arquiteturas Ampere e Hopper, são organizadas em torno de Streaming Multiprocessors (SMs), unidades de processamento semi-independentes que coordenam a execução de milhares de threads simultâneas. Cada SM contém núcleos CUDA para operações de ponto flutuante convencionais, unidades de memória compartilhada de baixa latência, e os revolucionários Tensor Cores — hardware especializado projetado especificamente para acelerar operações de multiplicação de matrizes que constituem o núcleo computacional de redes neurais (Jia et al., 2018). A arquitetura Hopper H100, por exemplo, disponibiliza 132 SMs com capacidade agregada de 989 TFLOPS em FP16, representando um salto significativo em relação às gerações anteriores.

O modelo de execução Single Instruction Multiple Threads (SIMT) constitui o paradigma fundamental que diferencia GPUs de CPUs tradicionais. Neste modelo, grupos de 32 threads — denominados warps — executam a mesma instrução simultaneamente sobre dados diferentes, maximizando o throughput em operações vetorizadas. O scheduler de warps alterna rapidamente entre warps ativos, ocultando latências de acesso à memória através de uma técnica conhecida como latency hiding. Quando um warp aguarda dados da memória, outro warp assume a execução, mantendo as unidades computacionais continuamente ocupadas (Volkov & Demmel, 2008).

**Hierarquia de Memória: O Gargalo Crítico**

Compreender a hierarquia de memória GPU é essencial para otimização de performance, pois a largura de banda de memória frequentemente constitui o gargalo principal em workloads de Machine Learning. A hierarquia organiza-se em níveis de capacidade e latência crescentes: registradores (mais rápidos, dezenas de KB por SM), shared memory/L1 cache (SRAM on-chip, ~200 ciclos), L2 cache (compartilhado entre SMs), e HBM — High Bandwidth Memory — que constitui a memória global do dispositivo.

A HBM representa uma inovação arquitetural significativa, empilhando chips de memória verticalmente e conectando-os ao processador através de interfaces largas de alta velocidade. A arquitetura A100 disponibiliza 80GB de HBM2e com largura de banda de 2TB/s, enquanto a H100 eleva esses números para 80GB HBM3 com 3.35TB/s (NVIDIA, 2022). Apesar desses valores impressionantes, a largura de banda permanece o recurso mais escasso em muitas operações de Deep Learning.

O conceito de coalesced memory access descreve padrões de acesso à memória que maximizam a utilização da largura de banda. Quando threads adjacentes em um warp acessam posições de memória consecutivas, o hardware agrupa essas requisições em transações únicas, aproveitando eficientemente a largura do barramento. Acessos não-coalescidos, por outro lado, resultam em múltiplas transações serializadas, degradando drasticamente a performance. Segundo análises da NVIDIA, padrões de acesso otimizados podem melhorar a performance em até 10x comparado a acessos aleatórios (Harris, 2017).

**O Modelo Roofline: Caracterizando Workloads**

O Modelo Roofline, proposto por Williams, Waterman e Patterson (2009), oferece um framework analítico elegante para caracterizar a performance de workloads computacionais em função de dois parâmetros fundamentais: o pico teórico de performance computacional do hardware (em FLOPS) e a largura de banda de memória. A métrica central é a intensidade aritmética, definida como a razão entre operações de ponto flutuante e bytes transferidos da memória:

$$\text{Intensidade Aritmética} = \frac{\text{FLOPs}}{\text{Bytes Transferidos}}$$

Onde FLOPs representa o número de operações de ponto flutuante executadas e Bytes Transferidos quantifica o volume de dados movimentados entre níveis de memória.

O modelo estabelece um "teto" (roofline) de performance atingível, determinado pelo menor entre dois limites: o pico computacional do hardware ou a largura de banda multiplicada pela intensidade aritmética. Formalmente:

> [FÓRMULA no dump — pág. 6-7]: A expressão formal do teto de performance foi referida ("Formalmente:") mas não transcrita legivelmente no dump. Interpretação usual: $P_{\text{atingível}} = \min(P_{\text{pico}},\ BW \times I)$, onde $I$ é a intensidade aritmética e $BW$ a largura de banda.

Workloads com baixa intensidade aritmética são classificados como memory-bound — sua performance é limitada pela velocidade de transferência de dados e não pela capacidade computacional. Operações como batch normalization, layer normalization e softmax tipicamente apresentam baixa intensidade aritmética, passando a maior parte do tempo aguardando dados da memória. Por outro lado, multiplicações de matrizes densas (matmul) exibem alta intensidade aritmética e são compute-bound, sendo capazes de saturar as unidades computacionais disponíveis.

> [NOTA — não é conteúdo FIAP]: no dump original, na frase acima aparecia o caractere cirílico "и" no lugar do "e" ("transferência de dados и não pela capacidade computacional") — provável artefato de OCR. Corrigido para "e".

A aplicação prática do modelo roofline em Deep Learning revela insights importantes. Estudos de profiling em Vision Transformers indicam que aproximadamente 40% do tempo de execução concentra-se em operações memory-bound, apesar de representarem fração menor do total de FLOPs (Dao et al., 2022). Esta observação motivou desenvolvimentos como FlashAttention, que reorganiza computações de atenção para maximizar uso de SRAM e minimizar transferências de HBM.

| Operação | Intensidade Aritmética | Classificação |
|----------|------------------------|---------------|
| Matmul (matrizes grandes) | Alta (~100 FLOP/byte) | Compute-bound |
| Convolução 3x3 | Média (~30 FLOP/byte) | Balanceado |
| BatchNorm | Baixa (~2 FLOP/byte) | Memory-bound |
| Softmax | Baixa (~1 FLOP/byte) | Memory-bound |
| LayerNorm | Baixa (~3 FLOP/byte) | Memory-bound |

Tabela 1 - Intensidade aritmética típica de operações comuns em deep learning. Fonte: Elaborado pelo autor com base em Dao et al. (2022)

**Mixed Precision Training: Acelerando sem Comprometer**

Mixed Precision Training, formalizado por Micikevicius et al. (2018), representa uma das técnicas mais impactantes para aceleração de treinamento em GPUs modernas. A abordagem fundamenta-se na observação de que redes neurais toleram representações numéricas de precisão reduzida durante o treinamento, desde que certas salvaguardas sejam implementadas para garantir estabilidade numérica.

A técnica utiliza precisão reduzida (FP16 ou BF16) para computações de forward pass e backward pass, onde o volume de operações é massivo, enquanto mantém cópias master dos pesos em FP32 para acumulação de gradientes. Esta estratégia híbrida captura o melhor de dois mundos: a velocidade dos formatos de baixa precisão e a estabilidade numérica da precisão completa.

O formato FP16 (half precision) utiliza 1 bit de sinal, 5 bits de expoente e 10 bits de mantissa, oferecendo range dinâmico de aproximadamente 6×10⁻⁵ a 6.5×10⁴. O desafio principal reside no fenômeno de underflow: gradientes muito pequenos — comuns em camadas profundas — são arredondados para zero, impedindo atualizações de pesos. A solução proposta por Micikevicius et al. é o loss scaling: multiplicar o loss por um fator grande antes do backward pass e dividir os gradientes pelo mesmo fator após, efetivamente deslocando valores para dentro do range representável.

> [FÓRMULA no dump — pág. 8]: A equação do loss scaling foi referida ("Onde S representa o fator de escala...") mas não transcrita legivelmente no dump.

Onde S representa o fator de escala, tipicamente iniciado em valores como 2¹⁶ e ajustado dinamicamente durante o treinamento.

O formato BF16 (bfloat16), desenvolvido pelo Google para TPUs e posteriormente adotado em GPUs NVIDIA, oferece alternativa interessante. Com 1 bit de sinal, 8 bits de expoente e 7 bits de mantissa, BF16 mantém o mesmo range dinâmico do FP32, eliminando a necessidade de loss scaling na maioria dos casos, ao custo de precisão ligeiramente menor. Empiricamente, BF16 demonstra robustez equivalente a FP16 com loss scaling em grande variedade de modelos (Kalamkar et al., 2019).

A arquitetura Ampere introduziu o formato TF32 (TensorFloat-32), que combina 8 bits de expoente do BF16 com 10 bits de mantissa do FP16, oferecendo compromisso intermediário. TF32 opera transparentemente em Tensor Cores — aplicações que utilizam operações FP32 convencionais são automaticamente aceleradas sem modificação de código, alcançando speedups de até 8x em operações matriciais (NVIDIA, 2020).

Resultados experimentais documentados na literatura demonstram consistentemente a eficácia de mixed precision. Treinamento de BERT-large com mixed precision FP16 reduce tempo de treinamento em 1.8x com degradação de acurácia inferior a 0.1% em tarefas do benchmark GLUE (Micikevicius et al., 2018). ResNet-50 em ImageNet alcança convergência equivalente com speedup de 2.8x em GPUs V100 equipadas com Tensor Cores.

**Tensor Cores: Hardware Especializado para Deep Learning**

Tensor Cores constituem a inovação de hardware mais significativa para aceleração de Deep Learning na última década. Introduzidos na arquitetura NVIDIA Volta (2017), estes circuitos especializados executam operações de multiplicação-acumulação de matrizes (matrix multiply-accumulate ou MMA) em um único ciclo de clock, processando blocos de matrizes simultaneamente.

A operação fundamental executada por Tensor Cores é expressa como:

$$D = A \times B + C$$

> [FÓRMULA no dump — pág. 9]: A equação MMA foi referida ("Onde A, B, C e D são matrizes...") mas não transcrita explicitamente no dump. Reconstruída como D = A×B + C (multiply-accumulate) a partir do texto.

Onde A, B, C e D são matrizes de dimensões específicas dependentes da geração de hardware. Em Tensor Cores de terceira geração (Ampere), a operação processa blocos 8×4 × 4×8, produzindo resultado 8×8 por ciclo. A quarta geração (Hopper) expande para operações ainda maiores através de instruções WGMMA (Warp Group Matrix Multiply-Accumulate).

**O QUE VOCÊ VIU NESTA AULA?**

Nesta aula, você compreendeu os fundamentos da arquitetura de GPUs NVIDIA, incluindo a organização em Streaming Multiprocessors, o modelo de execução SIMT e a hierarquia de memória que determina performance. Aprendeu a aplicar o modelo roofline para classificar operações como compute-bound ou memory-bound, direcionando estratégias de otimização.

Explorou Mixed Precision Training como técnica para acelerar treinamento mantendo precisão, compreendendo o papel de loss scaling e formatos numéricos como FP16, BF16 e TF32. Conheceu Tensor Cores e sua evolução ao longo das gerações de hardware NVIDIA.

Estudou técnicas de otimização de memória como gradient checkpointing, comparou GPUs com aceleradores alternativos (TPUs, NPUs, AWS Inferentia), e compreendeu metodologias de profiling usando NVIDIA Nsight Systems e PyTorch Profiler para identificar e resolver gargalos de performance em seus modelos.

**REFERÊNCIAS**

BROWN, Tom B. et al. Language Models are Few-Shot Learners. In: Advances in Neural Information Processing Systems (NeurIPS), 2020. arXiv:2005.14165. DOI: 10.48550/arXiv.2005.14165.

CHEN, Tianqi et al. Training Deep Nets with Sublinear Memory Cost. arXiv preprint, 2016. DOI: 10.48550/arXiv.1604.06174.

DAO, Tri. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. In: Advances in Neural Information Processing Systems (NeurIPS), 2023. DOI: 10.48550/arXiv.2307.08691.

DAO, Tri et al. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. In: Advances in Neural Information Processing Systems (NeurIPS), 2022. DOI: 10.48550/arXiv.2205.14135.

DETTMERS, Tim et al. QLoRA: Efficient Finetuning of Quantized LLMs. In: Advances in Neural Information Processing Systems (NeurIPS), 2024. DOI: 10.48550/arXiv.2305.14314.

HARRIS, Mark. Unified Memory in CUDA 6. NVIDIA Developer Blog, 2017. Disponível em: https://developer.nvidia.com/blog/unified-memory-cuda-beginners/. Acesso em: 27 mai. 2026.

JIA, Zhe et al. Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking. arXiv preprint, 2018. DOI: 10.48550/arXiv.1804.06826.

**PALAVRAS-CHAVE**

GPUs. NVIDIA. TPUs.

### Código e comandos
Nenhum bloco de código nesta aula.

### Ferramentas / serviços citados
- GPUs NVIDIA: Ampere, Hopper (H100, A100), Volta (V100)
- Tensor Cores, CUDA cores, HBM2e/HBM3
- Formatos: FP16, BF16, TF32, FP32
- NVIDIA Nsight Systems, PyTorch Profiler, PyTorch 2.x, CUDA 11.8+
- Aceleradores alternativos: TPU, NPU, AWS Inferentia
- Azure NC/ND series (cloud)
- FlashAttention / FlashAttention-2
- Repositório: github.com/fiap-postech/gpu-acceleration-ml

### Aplicabilidade ao Tech Challenge Fase 3
- Mixed Precision (FP16/BF16) e Tensor Cores aceleram tanto o fine-tuning quanto a inferência do classificador NLP, reduzindo latência sem perda relevante de acurácia (a aula cita BERT-large com FP16: 1.8x mais rápido, <0.1% de degradação).
- O modelo Roofline diagnostica se a inferência do classificador é memory-bound ou compute-bound, orientando a estratégia de otimização correta.
- AWS Inferentia / escolha de acelerador é relevante para dimensionar custo/latência do serving do TC em cloud.

---

## Aula 7 — Inferência Distribuída e Paralelismo de Modelos
**Arquivo fonte:** `Aula 07 - Inferência Distribuída e Paralelismo de Modelos.pdf` (12 páginas)
**Título na ementa:** Inferência Distribuída e Paralelismo

### Conceitos-chave
- Taxonomia de paralelismo: data, tensor e pipeline parallelism
- Lei de Amdahl vs. Lei de Gustafson (workloads escaláveis)
- ZeRO (Zero Redundancy Optimizer): Stages 1, 2 e 3
- Tensor parallelism (Megatron-LM): particionamento por colunas e por linhas
- Pipeline parallelism, pipeline bubble e micro-batching (GPipe, PipeDream)
- PagedAttention, continuous batching, speculative decoding, KV-Cache
- Ferramentas de serving: vLLM, Triton, TGI

### Conteúdo

**O QUE VEM POR AÍ?**

Imagine servir milhões de requisições simultâneas para um modelo de linguagem com bilhões de parâmetros, mantendo latência abaixo de meio segundo. Este é o desafio central da inferência distribuída em produção, onde técnicas sofisticadas de paralelismo e gerenciamento de memória transformam modelos academicamente impressionantes em produtos comercialmente viáveis. A revolução dos Large Language Models trouxe capacidades extraordinárias, mas também demandas computacionais que excedem a capacidade de qualquer GPU individual.

**HANDS ON**

Nas videoaulas desta aula, você implementará um sistema completo de serving distribuído para o modelo LLaMA-2 13B utilizando vLLM com PagedAttention e continuous batching. Demonstraremos passo a passo a configuração de tensor parallelism em múltiplas GPUs, a instrumentação de métricas de latência e throughput, e a comparação direta com serving tradicional usando Hugging Face Transformers. Você observará na prática como técnicas de gerenciamento eficiente de KV-Cache podem multiplicar o throughput em até 8 vezes mantendo latências aceitáveis.

O ambiente prático requer Python 3.11+, vLLM 0.3+, e acesso a GPUs NVIDIA com suporte a CUDA 12.0 ou superior. Para execução local, recomendamos pelo menos 2x A10G ou equivalente. Alternativamente, demonstraremos configuração em Azure ML Managed Endpoints para quem não dispõe de hardware local. Todo o código está organizado em notebooks Jupyter com células executáveis sequencialmente.

Código completo disponível no GitHub: github.com/fiap-postech/distributed-inference-llm

**SAIBA MAIS**

**Fundamentos de Paralelismo em Deep Learning**

A demanda computacional de modelos de linguagem de grande escala transformou fundamentalmente a arquitetura de sistemas de inferência. Enquanto o GPT-2 (2019) operava confortavelmente em uma única GPU com 1.5 bilhões de parâmetros, modelos contemporâneos como LLaMA-2 70B e GPT-4 exigem dezenas ou centenas de aceleradores operando em conjunto (Brown et al., 2020). Esta evolução tornou o paralelismo não uma otimização opcional, mas um requisito arquitetural fundamental para viabilizar tanto o treinamento quanto a inferência destes modelos.

A taxonomia de estratégias de paralelismo em Deep Learning compreende três abordagens fundamentais, cada uma com características e casos de uso distintos. O data parallelism replica o modelo completo em múltiplos dispositivos, particionando o batch de dados entre eles. Cada réplica processa seu subset de dados independentemente, e os gradientes são sincronizados através de operações all-reduce antes da atualização de pesos. Esta abordagem é conceitualmente simples e escala linearmente com o número de dispositivos, desde que o modelo caiba integralmente na memória de cada acelerador (Li et al., 2020).

O model parallelism, por sua vez, particiona o próprio modelo entre dispositivos quando este excede a capacidade de memória de um único acelerador. Duas variantes principais emergem desta estratégia: o pipeline parallelism divide o modelo por camadas, onde cada dispositivo processa um subconjunto sequencial de camadas, enquanto o tensor parallelism divide operações individuais (tipicamente multiplicações matriciais) entre dispositivos. A escolha entre estas abordagens depende da arquitetura do modelo, da infraestrutura disponível e dos requisitos de latência e throughput do sistema.

A Lei de Amdahl tradicionalmente limita os ganhos de paralelização ao estabelecer que o speedup máximo é determinado pela fração sequencial do código. Se 10% do código é inerentemente sequencial, o speedup máximo possível é 10x, independentemente do número de processadores. Entretanto, Gustafson (1988) propôs uma perspectiva complementar: em workloads escaláveis, onde o tamanho do problema cresce com os recursos disponíveis, os ganhos de paralelização podem ser substancialmente maiores. Esta perspectiva é particularmente relevante para inferência de LLMs, onde aumentar o batch size proporciona ganhos próximos ao linear em throughput.

**ZeRO: Otimização de Data Parallelism**

A técnica Zero Redundancy Optimizer (ZeRO), proposta por Rajbhandari et al. (2020), representa uma evolução fundamental do data parallelism tradicional. Em data parallelism convencional, cada dispositivo mantém uma cópia completa dos estados do otimizador, gradientes e parâmetros do modelo. Para um modelo de 10 bilhões de parâmetros treinado com Adam em precisão mista, isso significa aproximadamente 160GB de memória por dispositivo — muito além da capacidade de GPUs atuais.

ZeRO elimina esta redundância através de três estágios progressivos de particionamento. O ZeRO Stage 1 particiona apenas os estados do otimizador entre dispositivos, reduzindo o consumo de memória em até 4x. O Stage 2 adiciona o particionamento de gradientes, alcançando redução de até 8x. Finalmente, o Stage 3 particiona também os parâmetros do modelo, permitindo que modelos arbitrariamente grandes sejam treinados em clusters de GPUs com memória limitada. A comunicação adicional necessária para reconstruir tensores completos quando necessário é cuidadosamente otimizada para minimizar overhead.

Formalmente, considere um modelo com Ψ parâmetros e N_d dispositivos. Em data parallelism convencional, cada dispositivo armazena K⋅Ψ bytes, onde K representa o fator de multiplicação pelos estados do otimizador (tipicamente 12-16 para Adam em precisão mista). Com ZeRO Stage 3, cada dispositivo armazena apenas $\frac{K \cdot \Psi}{N_d}$ bytes, tornando possível treinar modelos com trilhões de parâmetros em clusters de GPUs commodity.

> [NOTA — não é conteúdo FIAP]: no dump, os subscritos de N_d e a fração K⋅Ψ/N_d vieram quebrados em várias linhas (paginação). Reconstruídos aqui conforme o sentido do texto.

**Tensor Parallelism para Large Language Models**

O tensor parallelism, popularizado pelo framework Megatron-LM (Shoeybi et al., 2019), particiona operações individuais — especificamente multiplicações matriciais — entre múltiplos dispositivos. Para camadas totalmente conectadas, que dominam a computação em Transformers, duas estratégias de particionamento emergem: particionamento por colunas e particionamento por linhas.

No particionamento por colunas, uma matriz de pesos W ∈ ℝ^(m×n) é dividida ao longo da dimensão das colunas em p partições: W = [W₁ ∣ W₂ ∣...∣ W_p], onde cada W_i ∈ ℝ^(m×(n/p)). Cada dispositivo computa Y_i = XW_i, e os resultados são concatenados. No particionamento por linhas, a matriz é dividida ao longo das linhas, e uma operação all-reduce é necessária para somar as contribuições parciais. A escolha entre estas estratégias depende da posição da operação na arquitetura e dos padrões de comunicação desejados.

> [NOTA — não é conteúdo FIAP]: no dump, os índices/subscritos das matrizes (W₁, W₂, W_p, W_i, Y_i) e as dimensões (m×n, m×(n/p)) vieram quebrados em linhas separadas pela paginação. Reconstruídos aqui conforme o sentido do texto.

Para o mecanismo de self-attention em Transformers, Megatron-LM aplica uma estratégia elegante. As matrizes de projeção de Query, Key e Value (W_Q, W_K, W_V) são particionadas por colunas, permitindo que cada dispositivo compute um subconjunto das cabeças de atenção independentemente. A projeção de saída é particionada por linhas, seguida de all-reduce. Esta arquitetura minimiza comunicação ao exigir apenas duas operações all-reduce por camada Transformer, independentemente do número de dispositivos.

A eficiência do tensor parallelism depende criticamente da latência de comunicação entre dispositivos. Para interconexões de alta largura de banda como NVLink (900 GB/s bidirecional no A100), tensor parallelism é altamente eficiente em até 8 GPUs. Para clusters maiores conectados via InfiniBand ou Ethernet, a combinação de tensor parallelism intra-nó com pipeline parallelism inter-nós frequentemente oferece melhor trade-off entre throughput e eficiência de comunicação.

**Pipeline Parallelism e Micro-batching**

Pipeline parallelism divide o modelo sequencialmente por camadas, atribuindo grupos de camadas a dispositivos distintos. Esta abordagem introduz um desafio fundamental: quando o dispositivo i está processando um batch, os dispositivos i+1 a N permanecem ociosos aguardando a saída de i. Este fenômeno, conhecido como pipeline bubble, pode consumir uma fração significativa do tempo de computação.

A técnica de micro-batching, formalizada no framework GPipe (Huang et al., 2019), mitiga este problema dividindo cada batch em micro-batches menores que fluem pelo pipeline. Se um batch é dividido em M micro-batches e o pipeline tem P estágios, a fração do tempo perdida em bubbles é aproximadamente $\frac{P-1}{M+P-1}$. Para M >> P, esta fração tende a zero, mas à custa de maior latência para completar cada batch.

O framework PipeDream (Narayanan et al., 2019) propõe uma alternativa: pipeline assíncrono com weight stashing, onde diferentes micro-batches podem usar versões ligeiramente diferentes dos pesos. Esta abordagem elimina bubbles ao permitir que dispositivos processem continuamente, mas introduz complexidade na garantia de convergência do treinamento.

**O QUE VOCÊ VIU NESTA AULA?**

Nesta aula, você explorou os fundamentos e técnicas avançadas de inferência distribuída para Large Language Models. Compreendeu a taxonomia de estratégias de paralelismo — data, tensor e pipeline — e quando aplicar cada uma. Aprofundou-se em PagedAttention, a inovação que revolucionou gerenciamento de KV-Cache ao aplicar princípios de memória virtual, e em continuous batching, que maximiza throughput através de scheduling dinâmico. Você também conheceu speculative decoding como estratégia de redução de latência e as principais ferramentas de serving (vLLM, Triton, TGI). Por fim, explorou práticas de auto-scaling e definição de SLOs para sistemas de inferência em produção.

**REFERÊNCIAS**

BROWN, Tom et al. Language Models are Few-Shot Learners. In: Advances in Neural Information Processing Systems (NeurIPS), 2020. Disponível em: https://arxiv.org/abs/2005.14165. DOI: 10.48550/arXiv.2005.14165.

CHEN, Charlie et al. Accelerating Large Language Model Decoding with Speculative Sampling. arXiv preprint, 2023. DOI: 10.48550/arXiv.2302.01318.

GUSTAFSON, John L. Reevaluating Amdahl's Law. Communications of the ACM, v. 31, n. 5, p. 532-533, 1988. DOI: 10.1145/42411.42415.

HUANG, Yanping et al. GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism. In: Advances in Neural Information Processing Systems (NeurIPS), 2019. Disponível em: https://arxiv.org/abs/1811.06965. DOI: 10.48550/arXiv.1811.06965.

**PALAVRAS-CHAVE**

Large Language Models. PagedAttention. SLOs.

### Código e comandos
Nenhum bloco de código nesta aula.

### Ferramentas / serviços citados
- vLLM 0.3+ (PagedAttention, continuous batching), Triton, TGI (Text Generation Inference)
- Hugging Face Transformers
- LLaMA-2 13B / 70B, GPT-2, GPT-4
- Megatron-LM, ZeRO, GPipe, PipeDream
- NVLink, InfiniBand, Ethernet
- Python 3.11+, CUDA 12.0+, GPUs NVIDIA (2x A10G), Azure ML Managed Endpoints
- Repositório: github.com/fiap-postech/distributed-inference-llm

### Aplicabilidade ao Tech Challenge Fase 3
- Continuous batching e gerenciamento eficiente de KV-Cache (via vLLM/Triton) aumentam o throughput do serving do classificador de texto sob carga, mantendo latência aceitável.
- A distinção Amdahl vs. Gustafson orienta o dimensionamento: aumentar batch size dá ganho quase linear de throughput na inferência do classificador NLP.
- Data parallelism (réplicas horizontais) é a estratégia mais simples e direta para escalar o serving do classificador do TC, já que o modelo cabe em uma única GPU.

---

## Aula 8 — Escalabilidade e Orquestração Multimodal em Produção
**Arquivo fonte:** `Aula 08 - Escalabilidade e Orquestração Multimodal em Produção.pdf` (12 páginas)
**Título na ementa:** Escalabilidade e Orquestração de Aplicações Multimodais

### Conceitos-chave
- SRE para ML: SLIs, SLOs e Error Budgets
- Data drift e concept drift
- Arquiteturas de serving: microsserviços vs. monolito
- Teorema CAP (consistência vs. disponibilidade)
- Orquestração com Kubernetes: KServe, Seldon Core, BentoML
- Canary deployments, auto-scaling, technical debt em ML

### Conteúdo

**O QUE VEM POR AÍ?**

Você já se perguntou como empresas como Netflix, Spotify e Amazon conseguem servir milhões de predições de modelos de Machine Learning simultaneamente, mantendo latências de milissegundos e disponibilidade quase perfeita? A resposta está na intersecção entre engenharia de confiabilidade (SRE) e MLOps, onde conceitos como SLOs, error budgets e canary deployments transformam modelos de laboratório em sistemas de produção resilientes e escaláveis.

**HANDS ON**

Nas videoaulas desta aula, você acompanhará a construção completa de uma plataforma multimodal de inferência utilizando Azure ML Managed Endpoints e Kubernetes. Demonstraremos a configuração de auto-scaling baseado em métricas customizadas, a implementação de canary deployments com rollback automático, e a instrumentação de observabilidade com Azure Monitor e Application Insights. Todo o código desenvolvido está disponível em nosso repositório, permitindo que você replique e adapte a solução para seus próprios cenários.

Repositório completo disponível no GitHub: github.com/fiap-postech/multimodal-serving-azure

Para acompanhar o hands-on, certifique-se de ter Python 3.11+, Azure CLI configurado, e kubectl instalado. O repositório contém instruções detalhadas de setup, incluindo scripts de provisionamento de infraestrutura via Terraform e configurações de Kubernetes prontas para deploy.

**SAIBA MAIS**

**Fundamentos de Site Reliability Engineering para Machine Learning**

A operacionalização de modelos de Machine Learning em escala representa um dos desafios mais complexos da engenharia de software moderna. Diferentemente de sistemas tradicionais, onde o comportamento é determinístico e previsível, sistemas de ML introduzem uma camada adicional de incerteza: o modelo em si é um artefato probabilístico cujo desempenho pode degradar silenciosamente ao longo do tempo devido a fenômenos como data drift e concept drift (Sculley et al., 2015). Esta característica fundamental exige uma abordagem especializada de engenharia de confiabilidade, adaptando os princípios consagrados do Site Reliability Engineering (SRE) para o contexto específico de inferência de ML.

O framework de SRE, formalizado por Beyer et al. (2016) no livro seminal "Site Reliability Engineering: How Google Runs Production Systems", estabelece três conceitos fundamentais que se tornaram pilares da operação de sistemas em escala: Service Level Indicators (SLIs), Service Level Objectives (SLOs) e Error Budgets. No contexto de serving de modelos de ML, estes conceitos adquirem nuances específicas que merecem atenção detalhada.

Os Service Level Indicators (SLIs) representam métricas quantitativas que capturam aspectos críticos do comportamento do sistema. Para sistemas de inferência de ML, os SLIs mais relevantes incluem latência de predição (tipicamente medida em percentis P50, P95 e P99), throughput (requisições por segundo), taxa de erros, e métricas específicas de ML como accuracy em produção e drift score. A escolha adequada de SLIs é crucial, pois estas métricas servirão como base para definição de objetivos e tomada de decisões operacionais.

Os Service Level Objectives (SLOs), por sua vez, estabelecem metas específicas para cada SLI. Um SLO típico para um sistema de recomendação em e-commerce poderia ser: "99% das requisições de inferência devem completar em menos de 200ms, medido em janelas de 30 dias". A definição de SLOs envolve um equilíbrio delicado entre as expectativas dos usuários, as capacidades técnicas do sistema, e os custos operacionais. SLOs excessivamente agressivos podem resultar em custos proibitivos de infraestrutura, enquanto SLOs muito relaxados podem comprometer a experiência do usuário.

O conceito de Error Budget emerge naturalmente da definição de SLOs. Se um sistema possui um SLO de 99.9% de disponibilidade, o error budget correspondente é de 0.1%, ou aproximadamente 43 minutos de downtime permitido por mês. Este modelo mental transforma a discussão de confiabilidade de um objetivo binário (zero erros) para um recurso gerenciável que pode ser "gasto" em atividades que agregam valor, como deploys de novas features ou experimentos em produção. Paleyes et al. (2022) demonstraram que organizações que adotam error budgets conseguem balancear melhor inovação e estabilidade em seus sistemas de ML.

**Arquiteturas de Serving: Microserviços versus Monolito para ML**

A decisão arquitetural entre microsserviços e monolito para sistemas de inferência de ML não é trivial e depende de múltiplos fatores contextuais. Ambas as abordagens possuem trade-offs bem documentados na literatura, e a escolha inadequada pode resultar em custos operacionais significativos ou limitações de escalabilidade que comprometem o sucesso do projeto.

A arquitetura monolítica para ML serving consolida todos os componentes de inferência em uma única unidade de deployment. Esta abordagem oferece vantagens importantes: simplicidade de desenvolvimento e debugging, menor overhead de comunicação entre componentes, e latência reduzida devido à ausência de chamadas de rede intra-sistema. Para modelos únicos ou pipelines simples, o monolito frequentemente representa a escolha mais pragmática, evitando a complexidade operacional inerente a sistemas distribuídos.

Por outro lado, a arquitetura de microsserviços decompõe o sistema de inferência em serviços independentes, cada um responsável por uma função específica do pipeline. Esta decomposição permite escalabilidade granular, onde componentes com demandas distintas podem ser dimensionados independentemente. Em um pipeline multimodal que processa texto, imagem e áudio, o serviço de processamento de imagens pode requerer GPUs potentes, enquanto o serviço de texto pode operar eficientemente em CPUs. A arquitetura de microsserviços permite alocar recursos de forma otimizada para cada tipo de workload.

O Teorema CAP, formalizado por Brewer (2000) e posteriormente provado por Gilbert e Lynch (2002), oferece um framework teórico para compreender os trade-offs em sistemas distribuídos. O teorema estabelece que, em presença de partições de rede, um sistema distribuído deve escolher entre consistência e disponibilidade. No contexto de serving de ML, este trade-off se manifesta na gestão de versões de modelos: garantir que todas as réplicas servem exatamente a mesma versão (consistência) pode comprometer a disponibilidade durante atualizações, enquanto permitir inconsistência temporária de versões (disponibilidade) pode resultar em comportamento não-determinístico do sistema.

A Figura a seguir ilustra a arquitetura de referência para sistemas multimodais, onde um API Gateway roteia requisições para workers especializados por modalidade, e um componente Aggregator consolida as respostas.

[DIAGRAMA: Figura 1 - Arquitetura de Referência para Serving Multimodal. Um API Gateway roteia requisições para workers especializados por modalidade (texto, imagem, áudio) e um componente Aggregator consolida as respostas. Fonte: Elaborado pelo autor (2026), adaptado de Paleyes et al. (2022)]

**Orquestração com Kubernetes: KServe, Seldon Core e BentoML**

Kubernetes emergiu como o padrão de fato para orquestração de containers em produção, e o ecossistema de ferramentas para ML serving sobre Kubernetes amadureceu significativamente nos últimos anos. Três frameworks destacam-se pela maturidade e adoção: KServe (anteriormente KFServing), Seldon Core e BentoML. Cada ferramenta possui filosofias de design distintas e casos de uso preferenciais.

KServe, desenvolvido inicialmente pela Kubeflow community e posteriormente adotado como projeto da CNCF, oferece uma abstração de alto nível para serving de modelos em Kubernetes (KServe Community, 2023). O framework introduz o conceito de InferenceService, um Custom Resource Definition (CRD) que encapsula toda a complexidade de deployment, scaling e canary releases em uma especificação declarativa. KServe suporta nativamente múltiplos frameworks de ML (TensorFlow, PyTorch, scikit-learn, XGBoost, ONNX) e oferece recursos avançados como autoscaling baseado em métricas customizadas, GPU sharing via Multi-Instance GPU (MIG), e integração com service meshes como Istio.

A arquitetura de KServe baseia-se em três componentes principais: o Predictor, responsável pela inferência propriamente dita; o Transformer, que realiza pré e pós-processamento; e o Explainer, que gera explicações para as predições (útil para compliance e debugging). Esta separação de responsabilidades permite escalar cada componente independentemente e substituir implementações sem afetar o restante do pipeline.

Seldon Core, desenvolvido pela Seldon Technologies, adota uma abordagem mais flexível através do conceito de Inference Graphs (Seldon Technologies, 2023). Um Inference Graph é um DAG (Directed Acyclic Graph) de componentes que podem incluir modelos, routers (para A/B testing), combiners (para ensemble), e transformers. Esta flexibilidade torna Seldon Core particularmente adequado para pipelines complexos onde múltiplos modelos colaboram para produzir uma predição final.

**O QUE VOCÊ VIU NESTA AULA?**

Nesta aula, você compreendeu os fundamentos de Site Reliability Engineering aplicados a sistemas de ML, incluindo SLIs, SLOs e error budgets como ferramentas para balancear inovação e estabilidade. Explorou as arquiteturas de serving (microserviços vs monolito) e seus trade-offs à luz do Teorema CAP, além de frameworks de orquestração como KServe, Seldon Core e BentoML. Aprendeu a configurar Azure ML Managed Endpoints com auto-scaling e a implementar canary deployments fundamentados em testes estatísticos.

Por fim, examinou os desafios específicos de pipelines multimodais e as formas de technical debt em sistemas de ML que podem comprometer a sustentabilidade de projetos em produção.

**REFERÊNCIAS**

BENTOML TEAM. BentoML Documentation. 2023. Disponível em: https://docs.bentoml.com. Acesso em: 27 mai. 2026.

BEYER, B. et al. Site Reliability Engineering: How Google Runs Production Systems. 1. ed. Sebastopol: O'Reilly Media, 2016. ISBN: 978-1491929124.

BREWER, E. Towards Robust Distributed Systems. In: Proceedings of the 19th Annual ACM Symposium on Principles of Distributed Computing (PODC), p. 7, 2000. DOI: 10.1145/343477.343502.

BURKOV, A. Machine Learning Engineering. 1. ed. Quebec: True Positive Inc., 2020. ISBN: 978-1999579579.

CUNNINGHAM, W. The WyCash Portfolio Management System. In: OOPSLA '92 Experience Report, 1992.

**PALAVRAS-CHAVE**

Site Reliability Engineering. SLIs. SLOs. Error budgets.

### Código e comandos
Nenhum bloco de código nesta aula.

### Ferramentas / serviços citados
- Azure ML Managed Endpoints, Azure Monitor, Application Insights, Azure CLI
- Kubernetes, kubectl, Terraform
- KServe (InferenceService CRD), Seldon Core (Inference Graphs), BentoML
- Istio (service mesh), Multi-Instance GPU (MIG)
- Frameworks suportados: TensorFlow, PyTorch, scikit-learn, XGBoost, ONNX
- Repositório: github.com/fiap-postech/multimodal-serving-azure

### Aplicabilidade ao Tech Challenge Fase 3
- SLIs/SLOs/Error Budgets fornecem o arcabouço para definir e monitorar formalmente a latência e disponibilidade do serviço de classificação NLP do TC.
- KServe/BentoML permitem servir o classificador (incl. via ONNX) em Kubernetes com auto-scaling e canary deployments — infraestrutura de produção adequada ao entregável do TC.
- Monitoramento de data drift / concept drift é essencial para manter a acurácia do classificador de texto ao longo do tempo em produção.

---
