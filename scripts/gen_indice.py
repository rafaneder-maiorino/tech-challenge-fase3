"""Gera 00-INDICE.md a partir dos .md de disciplina e do inventario.json.

Cria um índice navegável (links relativos para arquivos e âncoras de aula no
formato GitHub) + o resultado do inventário do Passo 0. Somente leitura sobre os
.md; escreve apenas 00-INDICE.md.
"""
import json
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("gen_indice")

DOCS = Path(__file__).resolve().parent.parent / "docs" / "_conhecimento"
INV = Path(__file__).resolve().parent / "inventario.json"

# ordem, arquivo md e rótulo por disciplina (chave = nome da pasta no inventário)
DISCIPLINAS = [
    ("Deploy em Nuvem", "01-deploy-em-nuvem.md", "Deploy em Nuvem", 6, 6),
    ("Integração com CI:CD (GitHub Actions)", "02-cicd-github-actions.md",
     "Integração com CI/CD (GitHub Actions)", 8, 8),
    ("Pipeline de Treino e Deploy Automático", "03-pipeline-treino-deploy.md",
     "Pipeline de Treino e Deploy Automático", 8, 8),
    ("Monitoramento de Performance", "04-monitoracao-performance.md",
     "Monitoração de Performance", 8, 8),
    ("Servicos de Monitoracao", "05-servicos-monitoracao.md",
     "Serviços de Monitoração", 8, 8),
    ("Latência e Performance em Modelos de Dados Não Estruturados",
     "06-latencia-dados-nao-estruturados.md",
     "Latência e Performance em Modelos de Dados Não Estruturados", 7, 8),
]


def anchor(heading: str) -> str:
    s = heading.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    return s.replace(" ", "-")


def headings_de(md_path: Path) -> list[str]:
    if not md_path.exists():
        return []
    out = []
    for linha in md_path.read_text(encoding="utf-8").splitlines():
        if linha.startswith("## Aula"):
            out.append(linha[3:].strip())
    return out


def norm_pasta(nome: str) -> str:
    """Casa a chave DISCIPLINAS com a chave do inventário (que pode ter espaços
    finais ou ':' no lugar de '/')."""
    return nome.strip()


def main() -> None:
    inv = json.loads(INV.read_text(encoding="utf-8"))
    # index do inventário por nome normalizado
    inv_norm = {norm_pasta(k): v for k, v in inv.items()}

    linhas = []
    linhas.append("# 00 — Índice da Base de Conhecimento (Fase 3)")
    linhas.append("> Base de conhecimento consolidada dos PDFs de aula da FIAP Pós Tech MLET — "
                  "Fase 3 (Cloud and MLOps).")
    linhas.append("> Aluno: RM 373042 · Data de extração: 2026-07-23")
    linhas.append("")
    linhas.append("Esta pasta reúne a transcrição fiel do conteúdo das 6 disciplinas da fase, "
                  "o enunciado do Tech Challenge e o mapa que cruza cada requisito do TC com as "
                  "aulas que o fundamentam.")
    linhas.append("")
    linhas.append("## Arquivos")
    linhas.append("")
    linhas.append("| # | Arquivo | Conteúdo |")
    linhas.append("|---|---|---|")
    linhas.append("| 00 | [00-INDICE.md](00-INDICE.md) | Este índice + resultado do inventário |")
    for _, arq, rotulo, n, m in DISCIPLINAS:
        linhas.append(f"| {arq[:2]} | [{arq}]({arq}) | {rotulo} ({n} de {m} aulas) |")
    linhas.append("| 07 | [07-tech-challenge-fase3.md](07-tech-challenge-fase3.md) | "
                  "Transcrição integral do enunciado do Tech Challenge |")
    linhas.append("| 99 | [99-MAPA-TECH-CHALLENGE.md](99-MAPA-TECH-CHALLENGE.md) | "
                  "Mapa requisito do TC → aulas + seções de lacunas |")
    linhas.append("")
    linhas.append("## Navegação por aula")
    linhas.append("")

    for chave, arq, rotulo, n, m in DISCIPLINAS:
        md_path = DOCS / arq
        heads = headings_de(md_path)
        linhas.append(f"### [{rotulo}]({arq}) — {n} de {m} aulas")
        for h in heads:
            linhas.append(f"- [{h}]({arq}#{anchor(h)})")
        linhas.append("")

    # ---- Inventário do Passo 0 ----
    linhas.append("---")
    linhas.append("")
    linhas.append("## Resultado do inventário (Passo 0)")
    linhas.append("")
    linhas.append("Percurso recursivo de `~/estudos/fiap/fase3` com `pathlib` (somente leitura). "
                  "Todos os 46 PDFs têm texto nativo extraível — **nenhum exigiu OCR**. "
                  "Extração com `pdfplumber 0.11.8` (fallback `pypdf 6.14.2`).")
    linhas.append("")
    linhas.append("| Disciplina (pasta no disco) | PDFs | Extraídos | OCR | Observação |")
    linhas.append("|---|---|---|---|---|")
    rotulos_inv = [
        ("Deploy em Nuvem", "Deploy em Nuvem", "6", "6", "—", "OK"),
        ("Integração com CI:CD (GitHub Actions)", "Integração com CI:CD (GitHub Actions)",
         "8", "8", "—", "`:` literal no nome da pasta no filesystem"),
        ("Latência e Performance em Modelos de Dados Não Estruturados",
         "Latência e Performance em Modelos de Dados Não Estruturados",
         "7", "7", "—", "**Aula 2 ausente** (não disponibilizada pela FIAP — lacuna conhecida)"),
        ("Monitoramento de Performance", "Monitoramento de Performance",
         "8", "8", "—", "Aula 1 = arquivo \"Material Complementar\" (capa: Latência vs. Throughput)"),
        ("Pipeline de Treino e Deploy Automático", "Pipeline de Treino e Deploy Automático",
         "8", "8", "—", "OK"),
        ("Servicos de Monitoracao", "Servicos de Monitoracao",
         "8", "8", "—", "pasta sem acento no disco"),
    ]
    for _, nome, pdfs, extr, ocr, obs in rotulos_inv:
        linhas.append(f"| {nome} | {pdfs} | {extr} | {ocr} | {obs} |")
    linhas.append("| **(raiz)** — `MLET - Tech Challenge Fase 3.pdf` | 1 | 1 | — | Enunciado do TC |")
    linhas.append("| **TOTAL** | **46** | **46** | **0** | 45 aulas + 1 enunciado |")
    linhas.append("")
    linhas.append("### Detalhamento por PDF")
    linhas.append("")
    for chave, arq, rotulo, n, m in DISCIPLINAS:
        # localizar itens do inventário para esta disciplina
        itens = None
        for k, v in inv_norm.items():
            if k == chave.strip():
                itens = v
                break
        if itens is None:
            continue
        linhas.append(f"**{rotulo}**")
        linhas.append("")
        linhas.append("| Aula | Arquivo no disco | Págs | Texto |")
        linhas.append("|---|---|---|---|")
        for it in itens:
            status = "nativo" if it["extraivel"] else "SCAN/OCR?"
            linhas.append(f"| {it['aula']} | `{it['nome_nfc']}` | {it['paginas']} | {status} |")
        if "Latência" in rotulo:
            linhas.append("| 2 | — (não disponibilizada pela FIAP) | — | **LACUNA** |")
        linhas.append("")

    linhas.append("### Lacuna conhecida")
    linhas.append("")
    linhas.append("- **Latência e Performance — Aula 2 \"Desafios de Performance em NLP e Áudio\"**: "
                  "não foi disponibilizada pela FIAP no portal (confirmado pelo aluno). Não é falha "
                  "de extração. Como o Tech Challenge é um classificador de texto (NLP), esta é a "
                  "lacuna de maior impacto — a cobertura compensatória está mapeada em "
                  "[99-MAPA-TECH-CHALLENGE.md](99-MAPA-TECH-CHALLENGE.md).")
    linhas.append("")

    out = DOCS / "00-INDICE.md"
    out.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    log.info("Índice escrito em %s (%d linhas)", out, len(linhas))


if __name__ == "__main__":
    main()
