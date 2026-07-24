"""Passo 0 — Inventário dos PDFs da Fase 3 (FIAP MLET).

Percorre recursivamente ~/estudos/fiap/fase3 usando pathlib (sem montar caminhos
com strings acentuadas), coletando por PDF: nome exato no disco, nº de páginas e
se o texto é extraível (PDF nativo) ou provável scan/imagem (exigiria OCR).

Somente leitura. Não altera nada em ~/estudos.
"""

import json
import logging
import re
import unicodedata
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("inventario")

BASE = Path.home() / "estudos" / "fiap" / "fase3"

# Limiar de caracteres extraídos por página, abaixo do qual suspeitamos scan/imagem.
CHAR_THRESHOLD_PER_PAGE = 50


def num_aula(nome: str) -> str:
    """Extrai o número da aula do nome do arquivo, tolerando variações."""
    m = re.search(r"[Aa]ula\s*0*(\d{1,2})", nome)
    return m.group(1) if m else "?"


def analisa_pdf(caminho: Path) -> dict:
    import pdfplumber

    info = {
        "arquivo": caminho.name,
        "nome_nfc": unicodedata.normalize("NFC", caminho.name),
        "aula": num_aula(caminho.name),
        "paginas": None,
        "chars_total": 0,
        "extraivel": None,
        "erro": None,
    }
    try:
        with pdfplumber.open(caminho) as pdf:
            info["paginas"] = len(pdf.pages)
            total = 0
            # amostra até 5 páginas para decidir se é extraível
            for pg in pdf.pages[:5]:
                txt = pg.extract_text() or ""
                total += len(txt.strip())
            info["chars_total"] = total
            paginas_amostradas = min(5, info["paginas"]) or 1
            info["extraivel"] = (total / paginas_amostradas) >= CHAR_THRESHOLD_PER_PAGE
    except Exception as e:
        info["erro"] = f"{type(e).__name__}: {e}"
        log.error("Falha ao abrir %s: %s", caminho.name, e)
    return info


def main() -> None:
    if not BASE.is_dir():
        log.error("Diretório base não encontrado: %s", BASE)
        return

    resultado = {}
    total_pdfs = 0

    # subpastas de disciplina (ordenadas) + PDFs soltos na raiz
    subpastas = sorted(
        [p for p in BASE.iterdir() if p.is_dir()], key=lambda p: p.name.lower()
    )
    for pasta in subpastas:
        pdfs = sorted(pasta.glob("*.pdf"), key=lambda p: p.name.lower())
        itens = []
        for pdf in pdfs:
            itens.append(analisa_pdf(pdf))
            total_pdfs += 1
        resultado[pasta.name] = itens
        log.info("%-70s %d PDFs", pasta.name, len(pdfs))

    # PDFs na raiz
    raiz_pdfs = sorted(BASE.glob("*.pdf"), key=lambda p: p.name.lower())
    if raiz_pdfs:
        itens = [analisa_pdf(p) for p in raiz_pdfs]
        total_pdfs += len(itens)
        resultado["(raiz)"] = itens
        log.info("%-70s %d PDFs", "(raiz)", len(raiz_pdfs))

    log.info("TOTAL: %d PDFs", total_pdfs)

    out = Path(__file__).parent / "inventario.json"
    out.write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("Inventário salvo em %s", out)


if __name__ == "__main__":
    main()
