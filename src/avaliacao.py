"""Métricas do projeto: Acurácia, BCE, F1, matriz de confusão e tempo
(as mesmas da tabela comparativa prevista para 24/11)."""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score


@torch.no_grad()
def avaliar(modelo: torch.nn.Module, dataloader, device: str = "cpu", limiar: float = 0.5) -> dict:
    """Roda o modelo no dataloader e devolve as métricas (classe positiva = maligno = 1).

    matriz_confusao no formato sklearn: [[TN, FP], [FN, TP]].
    """
    modelo.to(device).eval()
    probs, alvos = [], []
    t0 = time.perf_counter()
    for x, y in dataloader:
        probs.append(modelo(x.to(device)).reshape(-1).cpu())
        alvos.append(y.reshape(-1).cpu())
    tempo = time.perf_counter() - t0

    p = torch.cat(probs).float()
    y = torch.cat(alvos)
    pred = (p > limiar).long()
    return {
        "n_imagens": int(len(y)),
        "acuracia": float((pred == y).float().mean()),
        "bce": float(F.binary_cross_entropy(p, y.float())),
        "f1": float(f1_score(y.numpy(), pred.numpy(), pos_label=1, zero_division=0)),
        "matriz_confusao": confusion_matrix(y.numpy(), pred.numpy(), labels=[0, 1]).tolist(),
        "limiar": limiar,
        "tempo_inferencia_s": tempo,
        "tempo_por_imagem_ms": 1000 * tempo / max(len(y), 1),
    }
