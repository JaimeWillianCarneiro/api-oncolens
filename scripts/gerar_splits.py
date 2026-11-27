"""Gera o CSV de splits fixos.

    python -m scripts.gerar_splits --zip mhjyrn35p4-2.zip --pasta-dados dados
    # ou, se o zip já foi extraído:
    python -m scripts.gerar_splits --raiz-imagens "dados/Oral Images Dataset/augmented_data"

Depois de gerado, splits/splits.csv deve ser commitado: ninguém mais roda este script.
"""
import argparse
from pathlib import Path

from src.data import (SEMENTE_SPLIT, gerar_splits, listar_imagens, preparar_dados,
                      salvar_splits, verificar_sem_vazamento)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=None, help="mhjyrn35p4-2.zip")
    ap.add_argument("--pasta-dados", default="dados")
    ap.add_argument("--raiz-imagens", default=None)
    ap.add_argument("--saida", default="splits/splits.csv")
    ap.add_argument("--semente", type=int, default=SEMENTE_SPLIT)
    ap.add_argument("--sobrescrever", action="store_true")
    a = ap.parse_args()

    if Path(a.saida).exists() and not a.sobrescrever:
        raise SystemExit(f"{a.saida} já existe. Splits são fixos; use --sobrescrever só se for intencional.")

    raiz = Path(a.raiz_imagens) if a.raiz_imagens else preparar_dados(a.zip, a.pasta_dados)
    df = listar_imagens(raiz)
    splits = gerar_splits(df, semente=a.semente)
    contagem = verificar_sem_vazamento(splits, raiz_imagens=raiz, checar_hash=True)
    salvar_splits(splits, a.saida)

    print(f"{len(splits)} imagens -> {a.saida}")
    print(contagem)
    print(splits.groupby(["split", "label"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
