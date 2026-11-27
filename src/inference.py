"""Inferência limpa para o Jaime / app.py.

Uso:
    from src.inference import carregar_modelo, prever
    modelo = carregar_modelo("models/cnn_geral.ckpt")
    prever(modelo, "src/ehcancercaso1.png")
    # {'probabilidade': 0.83, 'classe': 'Maligno'}

Contrato (igual ao treino e ao explainability.py):
  * entrada: imagem RGB, redimensionada para o tamanho do modelo, pixels em [0,1],
    SEM normalização ImageNet;
  * saída do modelo: probabilidade de MALIGNO (label 1), sigmoid, shape [N,1].
    Não usar softmax.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image

from src.data import TAMANHO_IMAGEM, transform_padrao
from src.modelo_treinado import CNN, CNNLightning

# Faixas que estavam comentadas no final de modelo_treinado.py
LIMIAR_MALIGNO = 0.4
LIMIAR_BENIGNO = 0.2

Imagem = Union[str, Path, Image.Image, np.ndarray]

def resolver_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"

def _classe_do_checkpoint(ckpt: dict):
    chaves = list(ckpt["state_dict"].keys())
    if any(k.startswith("cnn_block.") for k in chaves):
        return CNN
    if any(k.startswith("conv.") for k in chaves):
        return CNNLightning
    raise ValueError(f"Checkpoint não reconhecido (chaves: {chaves[:4]}...)")


def carregar_modelo(caminho_ckpt: str | Path, device: str = "cpu") -> torch.nn.Module:
    """Carrega CNN ou CNNLightning detectando a classe pelas chaves do state_dict.

    Observação: a CNN "simples" não usa save_hyperparameters; ela é recriada com
    o padrão [16, 32, 64]. Se o ckpt tiver outra arquitetura, o load falha.
    """
    # weights_only=False: o ckpt é nosso (contém hparams do Lightning).
    ckpt = torch.load(caminho_ckpt, map_location="cpu", weights_only=False)
    classe = _classe_do_checkpoint(ckpt)
    modelo = classe.load_from_checkpoint(caminho_ckpt, map_location=device)
    modelo.to(device).eval()
    return modelo


def tamanho_entrada(modelo: torch.nn.Module) -> tuple[int, int]:
    """(altura, largura) usados no treino, quando o ckpt os registra; senão 224x224."""
    hp = getattr(modelo, "hparams", None)
    if hp is not None and "input_height" in hp and "input_width" in hp:
        return int(hp["input_height"]), int(hp["input_width"])
    return TAMANHO_IMAGEM


def preparar_imagem(imagem: Imagem, tamanho=TAMANHO_IMAGEM) -> torch.Tensor:
    """Qualquer entrada suportada -> tensor [1, 3, H, W] em [0,1]."""
    if isinstance(imagem, (str, Path)):
        imagem = Image.open(imagem)
    elif isinstance(imagem, np.ndarray):
        imagem = Image.fromarray(imagem)
    return transform_padrao(tamanho)(imagem.convert("RGB")).unsqueeze(0)


@torch.no_grad()
def prever_probabilidade(modelo: torch.nn.Module, imagem: Imagem) -> float:
    """P(maligno) em [0,1]."""
    device = next(modelo.parameters()).device
    x = preparar_imagem(imagem, tamanho_entrada(modelo)).to(device)
    modelo.eval()
    return float(modelo(x).reshape(-1)[0].item())


def rotular(prob: float) -> str:
    if prob > LIMIAR_MALIGNO:
        return "Maligno"
    if prob > LIMIAR_BENIGNO:
        return "Inconclusivo"
    return "Benigno"


def prever(modelo: torch.nn.Module, imagem: Imagem) -> dict:
    prob = prever_probabilidade(modelo, imagem)
    return {"probabilidade": prob, "classe": rotular(prob)}
