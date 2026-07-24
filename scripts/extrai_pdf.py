"""Extrai texto de um PDF por página usando pdfplumber (fallback pypdf).

Uso:
    python extrai_pdf.py "<glob-da-pasta>" "<glob-do-arquivo>" [pag_ini] [pag_fim]

Localiza o PDF via pathlib/glob (evita problemas de normalização Unicode NFD do
macOS) e imprime o texto delimitado por marcadores de página. Somente leitura.
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("extrai")

BASE = Path.home() / "estudos" / "fiap" / "fase3"


def extrai(pdf_path: Path, pag_ini: int, pag_fim: int) -> None:
    import pdfplumber

    log.info("Arquivo: %s", pdf_path.name)
    with pdfplumber.open(pdf_path) as pdf:
        n = len(pdf.pages)
        fim = min(pag_fim, n) if pag_fim else n
        for i in range(pag_ini - 1, fim):
            txt = pdf.pages[i].extract_text() or "[SEM TEXTO EXTRAÍVEL]"
            print(f"\n===== PÁGINA {i + 1}/{n} =====")
            print(txt)


def main() -> None:
    pasta_glob = sys.argv[1]
    arq_glob = sys.argv[2]
    pag_ini = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    pag_fim = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    pastas = sorted(BASE.glob(pasta_glob)) if pasta_glob != "." else [BASE]
    if not pastas:
        log.error("Nenhuma pasta casou com glob %r", pasta_glob)
        return
    pasta = pastas[0]
    arquivos = sorted(pasta.glob(arq_glob))
    if not arquivos:
        log.error("Nenhum PDF casou com %r em %s", arq_glob, pasta.name)
        return
    for pdf in arquivos:
        extrai(pdf, pag_ini, pag_fim)


if __name__ == "__main__":
    main()
