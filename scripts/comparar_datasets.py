"""Compara o desempenho do baseline entre augmented_data e original_data.

Cenários avaliados com mesma arquitetura, hiperparâmetros e seed:
  1. augmented_offline       -> splits/splits.csv em data/augmented_data
  2. original_puro           -> splits/splits_original.csv em data/original_data
  3. original_augment_online -> splits/splits_original.csv em data/original_data (aug só no treino)

Uso:
    python -m scripts.gerar_splits --raiz-imagens data/original_data --saida splits/splits_original.csv
    python -m scripts.comparar_datasets --raiz-augmented data/augmented_data --raiz-original data/original_data
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightning as L
import pandas as pd

from src.avaliacao import avaliar
from src.data import DataModule
from src.inference import carregar_modelo, resolver_device, tamanho_entrada
from src.modelo_treinado import CNN, CNNLightning

SEMENTE_TREINO = 1110


def _instanciar_modelo(referencia):
    if isinstance(referencia, CNNLightning):
        return CNNLightning(**dict(referencia.hparams))
    return CNN()


def _rodar_cenario(
    nome: str,
    referencia,
    tamanho: tuple[int, int],
    splits_csv: str | Path,
    raiz_imagens: str | Path,
    augment_treino: bool,
    epocas: int,
    batch_size: int,
    workers: int,
    device: str,
    pasta_ckpts: Path,
) -> dict:
    L.seed_everything(SEMENTE_TREINO, workers=True)
    dm = DataModule(
        splits_csv,
        raiz_imagens,
        batch_size=batch_size,
        num_workers=workers,
        tamanho=tamanho,
        augment_treino=augment_treino,
    )
    modelo = _instanciar_modelo(referencia)
    trainer = L.Trainer(
        max_epochs=epocas,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        accelerator="auto",
        devices=1,
    )

    t0 = time.perf_counter()
    trainer.fit(modelo, dm)
    tempo_treino = time.perf_counter() - t0

    ckpt_saida = pasta_ckpts / f"baseline_{nome}.ckpt"
    ckpt_saida.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(ckpt_saida)

    modelo_eval = carregar_modelo(ckpt_saida, device)
    dm.setup("fit")
    dm.setup("test")
    return {
        "cenario": nome,
        "splits": str(splits_csv),
        "raiz_imagens": str(raiz_imagens),
        "augment_treino": augment_treino,
        "epocas": epocas,
        "batch_size": batch_size,
        "tempo_treino_s": tempo_treino,
        "ckpt": str(ckpt_saida),
        "val": avaliar(modelo_eval, dm.val_dataloader(), device),
        "test": avaliar(modelo_eval, dm.test_dataloader(), device),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-ref", default="models/cnn_geral.ckpt")
    ap.add_argument("--splits-augmented", default="splits/splits.csv")
    ap.add_argument("--raiz-augmented", default="data/augmented_data")
    ap.add_argument("--splits-original", default="splits/splits_original.csv")
    ap.add_argument("--raiz-original", default="data/original_data")
    ap.add_argument("--epocas", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--pasta-ckpts", default="models/comparacao_datasets")
    ap.add_argument("--saida-json", default="reports/dataset_comparison.json")
    ap.add_argument("--saida-csv", default="reports/dataset_comparison.csv")
    a = ap.parse_args()

    device = resolver_device(a.device)
    ref = carregar_modelo(a.ckpt_ref, device)
    tamanho = tamanho_entrada(ref)

    cenarios = [
        ("augmented_offline", a.splits_augmented, a.raiz_augmented, False),
        ("original_puro", a.splits_original, a.raiz_original, False),
        ("original_augment_online", a.splits_original, a.raiz_original, True),
    ]

    resultados = {}
    linhas_csv = []
    for nome, splits_csv, raiz, aug in cenarios:
        res = _rodar_cenario(
            nome, ref, tamanho, splits_csv, raiz, aug,
            a.epocas, a.batch, a.workers, device, Path(a.pasta_ckpts)
        )
        resultados[nome] = res
        m = res["test"]
        tn, fp = m["matriz_confusao"][0]
        fn, tp = m["matriz_confusao"][1]
        sensibilidade = tp / max(tp + fn, 1)
        linhas_csv.append({
            "cenario": nome,
            "n_test": m["n_imagens"],
            "acuracia": round(m["acuracia"], 4),
            "bce": round(m["bce"], 4),
            "f1": round(m["f1"], 4),
            "sensibilidade_maligno": round(sensibilidade, 4),
            "matriz_confusao": str(m["matriz_confusao"]),
            "tempo_treino_s": round(res["tempo_treino_s"], 2),
            "tempo_por_imagem_ms": round(m["tempo_por_imagem_ms"], 2),
        })

    Path(a.saida_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.saida_json).write_text(json.dumps(resultados, indent=2, ensure_ascii=False))
    df = pd.DataFrame(linhas_csv)
    df.to_csv(a.saida_csv, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()