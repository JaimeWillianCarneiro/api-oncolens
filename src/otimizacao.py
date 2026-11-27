"""Optuna: função objective sobre a CNNLightning (src/modelo_treinado.py).

    python -m src.otimizacao \
        --splits splits/splits_original.csv \
        --raiz-imagens data/original_data \
        --augment-treino --n-trials 12 --epocas 12
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import Callback, EarlyStopping
import optuna
import torch

from src.data import TAMANHO_IMAGEM, DataModule
from src.modelo_treinado import CNNLightning

SEMENTE_TREINO = 1110
SEMENTE_OPTUNA = 1110

# Hiperparâmetros fixos para manter o espaço enxuto (mesmos da CNN baseline)
FIXOS = {"kernel_size": 3, "activation_name": "relu"}


def sugerir_hiperparametros(trial: optuna.Trial, batch_sizes: list[int] | None = None) -> dict:
    opcoes_batch = batch_sizes or [16, 32, 64]
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", opcoes_batch),
        "n_filters": trial.suggest_int("n_filters", 16, 64, step=16),
        "n_conv_layers": trial.suggest_int("n_conv_layers", 2, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4, step=0.1),
    }

class OptunaPruningCallback(Callback):
    """Reporta val_acc ao Optuna a cada época e guarda o melhor ponto do trial."""

    def __init__(self, trial: optuna.Trial):
        super().__init__()
        self.trial = trial
        self.melhor_val_acc = -1.0
        self.val_loss_no_melhor = float("inf")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metricas = trainer.callback_metrics
        if "val_acc" not in metricas:
            return
        val_acc = float(metricas["val_acc"])
        val_loss = float(metricas.get("val_loss", float("inf")))
        if val_acc > self.melhor_val_acc:
            self.melhor_val_acc = val_acc
            self.val_loss_no_melhor = val_loss

        self.trial.report(val_acc, step=trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


def criar_objective(
    splits_csv,
    raiz_imagens,
    max_epochs: int = 10,
    num_workers: int = 2,
    tamanho=TAMANHO_IMAGEM,
    augment_treino: bool = False,
    patience: int = 3,
    batch_sizes: list[int] | None = None,
):
    def objective(trial: optuna.Trial) -> float:
        hp = sugerir_hiperparametros(trial, batch_sizes=batch_sizes)
        L.seed_everything(SEMENTE_TREINO, workers=True)

        dm = DataModule(
            splits_csv,
            raiz_imagens,
            hp["batch_size"],
            num_workers,
            tamanho,
            augment_treino=augment_treino,
        )
        dm.setup("fit")

        modelo = CNNLightning(
            n_conv_layers=hp["n_conv_layers"],
            n_filters=hp["n_filters"],
            dropout=hp["dropout"],
            learning_rate=hp["learning_rate"],
            input_channels=3,
            input_height=tamanho[0],
            input_width=tamanho[1],
            num_classes=1,
            **FIXOS,
        )

        pruner_cb = OptunaPruningCallback(trial)
        callbacks = [
            pruner_cb,
            EarlyStopping(monitor="val_loss", mode="min", patience=patience),
        ]

        trainer = L.Trainer(
            max_epochs=max_epochs,
            callbacks=callbacks,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="auto",
            devices=1,
        )

        t0 = time.perf_counter()
        try:
            trainer.fit(modelo, datamodule=dm)
            tempo_treino = time.perf_counter() - t0
            trial.set_user_attr("tempo_treino_s", tempo_treino)
            trial.set_user_attr("epocas_executadas", trainer.current_epoch + 1)
            trial.set_user_attr("val_loss", pruner_cb.val_loss_no_melhor)
            return pruner_cb.melhor_val_acc
        finally:
            del trainer, modelo, dm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()

    return objective


def criar_estudo(
    nome: str = "otimizacao_cnn",
    storage: str = "sqlite:///optuna_cnn.db",
    semente: int = SEMENTE_OPTUNA,
) -> optuna.Study:
    return optuna.create_study(
        direction="maximize",
        study_name=nome,
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=semente),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=3),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="splits/splits.csv")
    ap.add_argument("--raiz-imagens", required=True)
    ap.add_argument("--augment-treino", action="store_true")
    ap.add_argument("--n-trials", type=int, default=12)
    ap.add_argument("--epocas", type=int, default=12)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--nome", default="otimizacao_cnn")
    ap.add_argument("--storage", default="sqlite:///optuna_cnn.db")
    ap.add_argument("--saida-csv", default="reports/optuna_trials.csv")
    ap.add_argument("--saida-json", default="reports/optuna_best_params.json")
    a = ap.parse_args()

    estudo = criar_estudo(a.nome, a.storage)
    estudo.optimize(
        criar_objective(
            a.splits,
            a.raiz_imagens,
            max_epochs=a.epocas,
            num_workers=a.workers,
            augment_treino=a.augment_treino,
            patience=a.patience,
            batch_sizes=a.batch_sizes,
        ),
        n_trials=a.n_trials,
    )

    Path(a.saida_csv).parent.mkdir(parents=True, exist_ok=True)
    estudo.trials_dataframe().to_csv(a.saida_csv, index=False)

    melhor = estudo.best_trial
    resumo = {
        "study_name": a.nome,
        "splits": a.splits,
        "raiz_imagens": a.raiz_imagens,
        "augment_treino": a.augment_treino,
        "best_trial_number": melhor.number,
        "best_val_acc": float(melhor.value),
        "best_val_loss": float(melhor.user_attrs.get("val_loss", float("nan"))),
        "best_params": {**FIXOS, **melhor.params},
    }
    Path(a.saida_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.saida_json).write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Melhor val_acc:", resumo["best_val_acc"])
    print("Melhores parâmetros salvos em:", a.saida_json)
    print(json.dumps(resumo["best_params"], indent=2))


if __name__ == "__main__":
    main()