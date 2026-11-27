"""Gera a tabela comparativa final entre baseline(s) e cnn_otimizada.ckpt.

    python -m scripts.comparar_modelos \
        --splits splits/splits_original.csv \
        --raiz-imagens data/original_data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.avaliacao import avaliar
from src.data import DataModule
from src.inference import carregar_modelo, resolver_device, tamanho_entrada


def _ler_tempo_treino(caminho_json: str | Path | None, chave: str | None = None) -> float | None:
    if not caminho_json or not Path(caminho_json).exists():
        return None
    dados = json.loads(Path(caminho_json).read_text(encoding="utf-8"))
    alvo = dados.get(chave, dados) if chave else dados
    return alvo.get("tempo_treino_s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--raiz-imagens", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--saida-csv", default="reports/tabela_comparativa.csv")
    ap.add_argument("--saida-md", default="reports/tabela_comparativa.md")
    a = ap.parse_args()

    device = resolver_device(a.device)
    candidatos = [
        ("cnn_geral (ref. historica)", "models/cnn_geral.ckpt", None, None),
        ("baseline_splits_fixos", "models/baseline_splits_fixos.ckpt", "reports/baseline_metrics.json", "retreinado_splits_fixos"),
        ("cnn_otimizada (Optuna)", "models/cnn_otimizada.ckpt", "reports/otimizada_metrics.json", None),
    ]

    linhas = []
    for nome, ckpt, json_tempo, chave_tempo in candidatos:
        if not Path(ckpt).exists():
            continue
        modelo = carregar_modelo(ckpt, device)
        dm = DataModule(a.splits, a.raiz_imagens, a.batch, a.workers, tamanho_entrada(modelo))
        dm.setup("test")
        m = avaliar(modelo, dm.test_dataloader(), device)
        tn, fp = m["matriz_confusao"][0]
        fn, tp = m["matriz_confusao"][1]
        tempo_treino = _ler_tempo_treino(json_tempo, chave_tempo)

        linhas.append({
            "modelo": nome,
            "acuracia": round(m["acuracia"], 4),
            "bce": round(m["bce"], 4),
            "f1": round(m["f1"], 4),
            "sensibilidade_maligno": round(tp / max(tp + fn, 1), 4),
            "matriz_confusao": str(m["matriz_confusao"]),
            "tempo_treino_s": round(tempo_treino, 2) if tempo_treino is not None else None,
            "tempo_inferencia_s": round(m["tempo_inferencia_s"], 3),
            "ms_por_imagem": round(m["tempo_por_imagem_ms"], 2),
        })

    df = pd.DataFrame(linhas)
    Path(a.saida_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.saida_csv, index=False)
    Path(a.saida_md).write_text(df.to_markdown(index=False), encoding="utf-8")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()