"""Gera dumps de texto limpo de todos os PDFs da Fase 3, por disciplina.

Para cada PDF extrai o texto página a página com pdfplumber, remove artefatos
recorrentes (cabeçalho de paginação "Página N de M" — inclusive suas variantes
com letras duplicadas/espaçadas — e o watermark "PDF exclusivo para ... / email")
e escreve um .txt por PDF em scripts/_raw/<disciplina>/.

Somente leitura sobre ~/estudos. Saída em scripts/_raw/ (ignorada pelo git).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("dump_raw")

BASE = Path.home() / "estudos" / "fiap" / "fase3"
OUT = Path(__file__).parent / "_raw"

RE_PAGINA = re.compile(r"Página\s*\d+\s*de\s*\d+")
RE_WATERMARK = re.compile(r"PDF exclusivo para", re.IGNORECASE)
RE_EMAIL = re.compile(r"^\s*rafaneder@gmail\.com\s*$", re.IGNORECASE)


def limpa_linha(linha: str) -> str | None:
    """Retorna None se a linha for artefato (cabeçalho/rodapé); senão a linha."""
    if RE_PAGINA.search(linha):
        return None
    if RE_WATERMARK.search(linha):
        return None
    if RE_EMAIL.match(linha):
        return None
    return linha


def dump_pdf(pdf_path: Path, out_path: Path) -> int:
    import pdfplumber

    linhas_saida = []
    with pdfplumber.open(pdf_path) as pdf:
        n = len(pdf.pages)
        for i, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or "[SEM TEXTO EXTRAÍVEL NESTA PÁGINA]"
            linhas_saida.append(f"\n----- [pág. {i}/{n}] -----")
            for linha in txt.splitlines():
                mantida = limpa_linha(linha)
                if mantida is not None:
                    linhas_saida.append(mantida)
    conteudo = "\n".join(linhas_saida).strip() + "\n"
    out_path.write_text(conteudo, encoding="utf-8")
    return n


def main() -> None:
    OUT.mkdir(exist_ok=True)
    subpastas = sorted(
        [p for p in BASE.iterdir() if p.is_dir()], key=lambda p: p.name.lower()
    )
    total = 0
    for pasta in subpastas:
        # nome de saída seguro: normaliza NFC e troca ':' por '_' (pasta de CI/CD)
        nome_seguro = unicodedata.normalize("NFC", pasta.name).strip().replace(":", "_")
        dest = OUT / nome_seguro
        dest.mkdir(parents=True, exist_ok=True)
        pdfs = sorted(pasta.glob("*.pdf"), key=lambda p: p.name.lower())
        for pdf in pdfs:
            stem = unicodedata.normalize("NFC", pdf.stem).replace("/", "_")
            out_path = dest / f"{stem}.txt"
            paginas = dump_pdf(pdf, out_path)
            total += 1
            log.info(
                "%-40s -> %s (%d págs)",
                pasta.name[:40],
                out_path.name[:50],
                paginas,
            )

    # PDFs na raiz (enunciado do TC)
    raiz_dest = OUT / "_raiz"
    raiz_dest.mkdir(parents=True, exist_ok=True)
    for pdf in sorted(BASE.glob("*.pdf")):
        stem = unicodedata.normalize("NFC", pdf.stem)
        out_path = raiz_dest / f"{stem}.txt"
        paginas = dump_pdf(pdf, out_path)
        total += 1
        log.info("%-40s -> %s (%d págs)", "(raiz)", out_path.name[:50], paginas)

    log.info("TOTAL dumps gerados: %d", total)


if __name__ == "__main__":
    main()
