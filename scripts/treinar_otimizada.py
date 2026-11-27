"""Treinamento final da cnn_otimizada.ckpt a partir dos melhores hiperparâmetros do Optuna.

    python -m scripts.treinar_otimizada \
        --params reports/optuna_best_params.json \
        --splits splits/splits_original.csv \
        --raiz-imagens data/original_data \
        --augment-treino
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from src.avaliacao import avaliar
from src.data import TAMANHO_IMAGEM, DataModule
from src.inference import carregar_modelo, resolver_device
from src.modelo_treinado import CNNLightning

SEMENTE_TREINO = 1110


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="reports/optuna_best_params.json")
    ap.add_argument("--splits", default="splits/splits.csv")
    ap.add_argument("--raiz-imagens", required=True)
    ap.add_argument("--augment-treino", action="store_true")
    ap.add_argument("--epocas", type=int, default=25)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--saida-ckpt", default="models/cnn_otimizada.ckpt")
    ap.add_argument("--saida-json", default="reports/otimizada_metrics.json")
    a = ap.parse_args()

    device = resolver_device(a.device)
    L.seed_everything(SEMENTE_TREINO, workers=True)

    bruto = json.loads(Path(a.params).read_text(encoding="utf-8"))
    params = bruto.get("best_params", bruto)
    batch_size = int(params.get("batch_size", 32))

    modelo = CNNLightning(
        n_conv_layers=int(params["n_conv_layers"]),
        n_filters=int(params["n_filters"]),
        kernel_size=int(params["kernel_size"]),
        activation_name=str(params.get("activation_name", "relu")),
        dropout=float(params["dropout"]),
        learning_rate=float(params["learning_rate"]),
        input_channels=3,
        input_height=TAMANHO_IMAGEM[0],
        input_width=TAMANHO_IMAGEM[1],
    )

    dm = DataModule(
        a.splits,
        a.raiz_imagens,
        batch_size=batch_size,
        num_workers=a.workers,
        tamanho=TAMANHO_IMAGEM,
        augment_treino=a.augment_treino,
    )

    saida_ckpt = Path(a.saida_ckpt)
    saida_ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=saida_ckpt.parent,
        filename="_tmp_cnn_otimizada",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    es_cb = EarlyStopping(monitor="val_loss", mode="min", patience=a.patience)

    trainer = L.Trainer(
        max_epochs=a.epocas,
        callbacks=[ckpt_cb, es_cb],
        logger=False,
        accelerator="auto",
        devices=1,
    )

    t0 = time.perf_counter()
    trainer.fit(modelo, dm)
    tempo_treino = time.perf_counter() - t0

    shutil.move(ckpt_cb.best_model_path, saida_ckpt)
    melhor = carregar_modelo(saida_ckpt, device)
    dm.setup("fit")
    dm.setup("test")

    resultado = {
        "ckpt": str(saida_ckpt),
        "splits": a.splits,
        "raiz_imagens": a.raiz_imagens,
        "augment_treino": a.augment_treino,
        "hparams": params,
        "epocas_max": a.epocas,
        "epocas_executadas": trainer.current_epoch,
        "tempo_treino_s": tempo_treino,
        "val": avaliar(melhor, dm.val_dataloader(), device),
        "test": avaliar(melhor, dm.test_dataloader(), device),
    }

    Path(a.saida_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.saida_json).write_text(json.dumps(resultado, indent=2, ensure_ascii=False))
    print(json.dumps(resultado, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()