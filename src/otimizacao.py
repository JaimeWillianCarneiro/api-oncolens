"""Optuna: função objective sobre a CNNLightning (src/modelo_treinado.py).

    python -m src.otimizacao --raiz-imagens "dados/Oral Images Dataset/augmented_data" \
        --n-trials 10 --epocas 10

Regras:
  * o objective só usa treino e validação (DataModule.setup("fit")); o teste não é tocado;
  * métrica otimizada: val_acc (como no notebook), maximizar; val_loss (BCE) fica
    salvo em user_attrs para comparação;
  * a semente de treino é a mesma em todos os trials: a diferença entre trials
    vem só dos hiperparâmetros.
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import lightning as L
import optuna
import torch

from src.data import TAMANHO_IMAGEM, DataModule
from src.modelo_treinado import CNNLightning

SEMENTE_TREINO = 1110
SEMENTE_OPTUNA = 1110

# Hiperparâmetros NÃO buscados: ficam iguais à CNN do baseline (kernel 3, ReLU, sem dropout).
FIXOS = {"kernel_size": 3, "activation_name": "relu", "dropout": 0.0}


def sugerir_hiperparametros(trial: optuna.Trial) -> dict:
    """Espaço de busca enxuto: LR, batch e filtros (faixas de LR/filtros/camadas do notebook)."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "n_filters": trial.suggest_int("n_filters", 16, 64, step=16),
        "n_conv_layers": trial.suggest_int("n_conv_layers", 1, 3),
    }


def criar_objective(splits_csv, raiz_imagens, max_epochs: int = 10, num_workers: int = 2,
                    tamanho=TAMANHO_IMAGEM):
    def objective(trial: optuna.Trial) -> float:
        hp = sugerir_hiperparametros(trial)
        L.seed_everything(SEMENTE_TREINO, workers=True)

        dm = DataModule(splits_csv, raiz_imagens, hp["batch_size"], num_workers, tamanho)
        dm.setup("fit")

        modelo = CNNLightning(
            n_conv_layers=hp["n_conv_layers"], n_filters=hp["n_filters"],
            learning_rate=hp["learning_rate"], input_channels=3,
            input_height=tamanho[0], input_width=tamanho[1], num_classes=1, **FIXOS)

        trainer = L.Trainer(max_epochs=max_epochs, enable_checkpointing=False, logger=False,
                            enable_progress_bar=False, enable_model_summary=False,
                            accelerator="auto", devices=1)
        t0 = time.perf_counter()
        trainer.fit(modelo, datamodule=dm)
        trial.set_user_attr("tempo_treino_s", time.perf_counter() - t0)
        trial.set_user_attr("val_loss", float(trainer.callback_metrics["val_loss"]))
        val_acc = float(trainer.callback_metrics["val_acc"])

        del trainer, modelo, dm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return val_acc

    return objective


def criar_estudo(nome: str = "otimizacao_cnn", storage: str = "sqlite:///optuna_cnn.db",
                 semente: int = SEMENTE_OPTUNA) -> optuna.Study:
    return optuna.create_study(direction="maximize", study_name=nome, storage=storage,
                               load_if_exists=True,
                               sampler=optuna.samplers.TPESampler(seed=semente))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="splits/splits.csv")
    ap.add_argument("--raiz-imagens", required=True)
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--epocas", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--nome", default="otimizacao_cnn")
    ap.add_argument("--storage", default="sqlite:///optuna_cnn.db")
    ap.add_argument("--saida-csv", default="reports/optuna_trials.csv")
    a = ap.parse_args()

    estudo = criar_estudo(a.nome, a.storage)
    estudo.optimize(criar_objective(a.splits, a.raiz_imagens, a.epocas, a.workers),
                    n_trials=a.n_trials)
    Path(a.saida_csv).parent.mkdir(parents=True, exist_ok=True)
    estudo.trials_dataframe().to_csv(a.saida_csv, index=False)
    print("Melhor val_acc:", estudo.best_value)
    print("Melhores parâmetros:", estudo.best_trial.params)


if __name__ == "__main__":
    main()
