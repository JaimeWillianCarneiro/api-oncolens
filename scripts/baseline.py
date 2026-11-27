"""Reprodução do baseline e registro das métricas iniciais.

1) Avalia o checkpoint original (models/cnn_geral.ckpt) nos splits fixos.
   ⚠ Esse ckpt foi treinado com um split que não foi salvo; parte do novo
   conjunto de teste pode ter sido vista no treino. Trate como referência.
2) Com --retreinar, treina de novo a MESMA arquitetura (hparams lidos do ckpt)
   só com o treino fixo e avalia no teste fixo. Este é o baseline sem vazamento.

    python -m scripts.baseline --ckpt models/cnn_geral.ckpt \
        --raiz-imagens "dados/Oral Images Dataset/augmented_data" --retreinar
"""
import argparse
import json
import time
from pathlib import Path

import lightning as L

from src.avaliacao import avaliar
from src.data import DataModule
from src.inference import carregar_modelo, tamanho_entrada
from src.modelo_treinado import CNN, CNNLightning

SEMENTE_TREINO = 1110


def _avaliar_val_teste(modelo, dm, device):
    dm.setup("fit")
    dm.setup("test")
    return {"val": avaliar(modelo, dm.val_dataloader(), device),
            "test": avaliar(modelo, dm.test_dataloader(), device)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/cnn_geral.ckpt")
    ap.add_argument("--splits", default="splits/splits.csv")
    ap.add_argument("--raiz-imagens", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--retreinar", action="store_true")
    ap.add_argument("--epocas", type=int, default=10)
    ap.add_argument("--saida", default="reports/baseline_metrics.json")
    ap.add_argument("--saida-ckpt", default="models/baseline_splits_fixos.ckpt")
    a = ap.parse_args()

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"  # avaliação simples e reprodutível; o treino usa accelerator="auto"
    L.seed_everything(SEMENTE_TREINO, workers=True)

    original = carregar_modelo(a.ckpt, device)
    tamanho = tamanho_entrada(original)
    dm = DataModule(a.splits, a.raiz_imagens, a.batch, a.workers, tamanho)

    resultado = {
        "arquitetura": type(original).__name__,
        "hparams_do_ckpt": dict(getattr(original, "hparams", {}) or {}),
        "tamanho_entrada": list(tamanho),
        "ckpt_original (split de treino desconhecido)": _avaliar_val_teste(original, dm, device),
    }

    if a.retreinar:
        if isinstance(original, CNNLightning):
            novo = CNNLightning(**dict(original.hparams))
            lr_batch = {"learning_rate": novo.learning_rate}
        else:
            novo = CNN()
            lr_batch = {"learning_rate": 0.001}  # TAXA_DE_APRENDIZADO de modelo_treinado.py
        trainer = L.Trainer(max_epochs=a.epocas, enable_checkpointing=False, logger=False,
                            enable_progress_bar=False, accelerator="auto", devices=1)
        t0 = time.perf_counter()
        trainer.fit(novo, dm)
        tempo_treino = time.perf_counter() - t0
        Path(a.saida_ckpt).parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(a.saida_ckpt)
        novo = carregar_modelo(a.saida_ckpt, device)
        resultado["retreinado_splits_fixos"] = {
            "epocas": a.epocas, "batch_size": a.batch, **lr_batch,
            "tempo_treino_s": tempo_treino, "ckpt": a.saida_ckpt,
            **_avaliar_val_teste(novo, dm, device),
        }

    Path(a.saida).parent.mkdir(parents=True, exist_ok=True)
    Path(a.saida).write_text(json.dumps(resultado, indent=2, ensure_ascii=False, default=str))
    print(json.dumps({k: v for k, v in resultado.items() if k != "hparams_do_ckpt"},
                     indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
