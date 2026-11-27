"""Gera a tabela comparativa final entre baseline(s) e cnn_otimizada.ckpt,
incluindo avaliação nos limiares operacionais e distribuição nas 3 faixas da API.

    python -m scripts.comparar_modelos \
        --splits splits/splits.csv \
        --raiz-imagens "/kaggle/input/datasets/luizbravin/oncolens1/Oral Images Dataset/augmented_data"
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score

from src.data import DataModule
from src.inference import (
    LIMIAR_BENIGNO,
    LIMIAR_MALIGNO,
    carregar_modelo,
    resolver_device,
    tamanho_entrada,
)


def _ler_tempo_treino(caminho_json: str | Path | None, chave: str | None = None) -> float | None:
    if not caminho_json or not Path(caminho_json).exists():
        return None
    dados = json.loads(Path(caminho_json).read_text(encoding="utf-8"))
    alvo = dados.get(chave, dados) if chave else dados
    return alvo.get("tempo_treino_s")


@torch.no_grad()
def _coletar_predicoes(modelo: torch.nn.Module, dataloader, device: str) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Executa uma única passada de inferência no dataloader e devolve (probs, alvos, tempo_s)."""
    modelo.to(device).eval()
    probs, alvos = [], []
    t0 = time.perf_counter()
    for x, y in dataloader:
        probs.append(modelo(x.to(device)).reshape(-1).cpu())
        alvos.append(y.reshape(-1).cpu())
    tempo = time.perf_counter() - t0
    return torch.cat(probs).float(), torch.cat(alvos).long(), tempo


def _metricas_no_limiar(p: torch.Tensor, y: torch.Tensor, limiar: float, tempo: float) -> dict:
    """Mesma definição de src/avaliacao.py, reutilizando as probabilidades já calculadas."""
    pred = (p > limiar).long()
    cm = confusion_matrix(y.numpy(), pred.numpy(), labels=[0, 1])
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])
    return {
        "n_imagens": int(len(y)),
        "acuracia": float((pred == y).float().mean()),
        "bce": float(F.binary_cross_entropy(p, y.float())),
        "f1": float(f1_score(y.numpy(), pred.numpy(), pos_label=1, zero_division=0)),
        "sensibilidade_maligno": tp / max(tp + fn, 1),
        "especificidade_benigno": tn / max(tn + fp, 1),
        "vpp_precisao_maligno": tp / max(tp + fp, 1),
        "vpn_confianca_benigno": tn / max(tn + fn, 1),
        "matriz_confusao": cm.tolist(),
        "tempo_inferencia_s": tempo,
        "tempo_por_imagem_ms": 1000 * tempo / max(len(y), 1),
    }


def _resumo_faixas_api(nome: str, p: torch.Tensor, y: torch.Tensor, lim_ben: float, lim_mal: float) -> list[dict]:
    """Distribuição nas 3 faixas operacionais da API: Benigno, Inconclusivo e Maligno."""
    total = max(len(y), 1)
    total_mal = max(int((y == 1).sum()), 1)
    faixas = [
        (f"Benigno (p <= {lim_ben:.3f})", p <= lim_ben),
        (f"Inconclusivo ({lim_ben:.3f} < p <= {lim_mal:.3f})", (p > lim_ben) & (p <= lim_mal)),
        (f"Maligno (p > {lim_mal:.3f})", p > lim_mal),
    ]
    linhas = []
    for rotulo, mask in faixas:
        n_ben = int(((y == 0) & mask).sum())
        n_mal = int(((y == 1) & mask).sum())
        n_tot = n_ben + n_mal
        linhas.append({
            "modelo": nome,
            "faixa": rotulo,
            "reais_benignos": n_ben,
            "reais_malignos": n_mal,
            "total": n_tot,
            "pct_dataset": round(100 * n_tot / total, 2),
            "pct_malignos_na_faixa": round(100 * n_mal / total_mal, 2),
        })
    return linhas


def _salvar_tabela(df: pd.DataFrame, caminho_csv: str | Path, caminho_md: str | Path) -> None:
    Path(caminho_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(caminho_md).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho_csv, index=False)
    Path(caminho_md).write_text(df.to_markdown(index=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--raiz-imagens", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limiar", type=float, default=0.5, help="Limiar principal da tabela comparativa")
    ap.add_argument("--limiar-benigno", type=float, default=None, help="Padrão: LIMIAR_BENIGNO de src.inference")
    ap.add_argument("--limiar-maligno", type=float, default=None, help="Padrão: LIMIAR_MALIGNO de src.inference")
    ap.add_argument("--limiares", type=float, nargs="+", default=None, help="Lista customizada de limiares para avaliar")
    ap.add_argument("--saida-csv", default="reports/tabela_comparativa.csv")
    ap.add_argument("--saida-md", default="reports/tabela_comparativa.md")
    ap.add_argument("--saida-limiares-csv", default="reports/tabela_limiares.csv")
    ap.add_argument("--saida-limiares-md", default="reports/tabela_limiares.md")
    ap.add_argument("--saida-faixas-csv", default="reports/tabela_faixas_api.csv")
    ap.add_argument("--saida-faixas-md", default="reports/tabela_faixas_api.md")
    a = ap.parse_args()

    lim_ben = LIMIAR_BENIGNO if a.limiar_benigno is None else a.limiar_benigno
    lim_mal = LIMIAR_MALIGNO if a.limiar_maligno is None else a.limiar_maligno
    if lim_ben >= lim_mal:
        raise ValueError(f"limiar_benigno ({lim_ben}) deve ser menor que limiar_maligno ({lim_mal}).")

    lista_limiares = a.limiares or list(dict.fromkeys([a.limiar, lim_mal, lim_ben]))
    device = resolver_device(a.device)

    candidatos = [
        ("cnn_geral (ref. historica)", "models/cnn_geral.ckpt", None, None),
        ("baseline_splits_fixos", "models/baseline_splits_fixos.ckpt", "reports/baseline_metrics.json", "retreinado_splits_fixos"),
        ("cnn_otimizada (Optuna)", "models/cnn_otimizada.ckpt", "reports/otimizada_metrics.json", None),
    ]

    linhas_principal = []
    linhas_limiares = []
    linhas_faixas = []

    for nome, ckpt, json_tempo, chave_tempo in candidatos:
        if not Path(ckpt).exists():
            continue
        modelo = carregar_modelo(ckpt, device)
        dm = DataModule(a.splits, a.raiz_imagens, a.batch, a.workers, tamanho_entrada(modelo))
        dm.setup("test")

        p, y, tempo_inf = _coletar_predicoes(modelo, dm.test_dataloader(), device)
        tempo_treino = _ler_tempo_treino(json_tempo, chave_tempo)

        # 1) Tabela principal no limiar padrão (--limiar, default 0.5)
        m_padrao = _metricas_no_limiar(p, y, a.limiar, tempo_inf)
        linhas_principal.append({
            "modelo": nome,
            "acuracia": round(m_padrao["acuracia"], 4),
            "bce": round(m_padrao["bce"], 4),
            "f1": round(m_padrao["f1"], 4),
            "sensibilidade_maligno": round(m_padrao["sensibilidade_maligno"], 4),
            "matriz_confusao": str(m_padrao["matriz_confusao"]),
            "tempo_treino_s": round(tempo_treino, 2) if tempo_treino is not None else None,
            "tempo_inferencia_s": round(m_padrao["tempo_inferencia_s"], 3),
            "ms_por_imagem": round(m_padrao["tempo_por_imagem_ms"], 2),
        })

        # 2) Tabela detalhada nos limiares (ex.: 0.5, 0.416, 0.16)
        for lim in lista_limiares:
            m = _metricas_no_limiar(p, y, lim, tempo_inf)
            linhas_limiares.append({
                "modelo": nome,
                "limiar": round(lim, 3),
                "acuracia": round(m["acuracia"], 4),
                "f1": round(m["f1"], 4),
                "sensibilidade_maligno": round(m["sensibilidade_maligno"], 4),
                "especificidade_benigno": round(m["especificidade_benigno"], 4),
                "vpp": round(m["vpp_precisao_maligno"], 4),
                "vpn": round(m["vpn_confianca_benigno"], 4),
                "matriz_confusao": str(m["matriz_confusao"]),
            })

        # 3) Quebra nas 3 faixas operacionais da API
        linhas_faixas.extend(_resumo_faixas_api(nome, p, y, lim_ben, lim_mal))

    df_principal = pd.DataFrame(linhas_principal)
    df_limiares = pd.DataFrame(linhas_limiares)
    df_faixas = pd.DataFrame(linhas_faixas)

    _salvar_tabela(df_principal, a.saida_csv, a.saida_md)
    _salvar_tabela(df_limiares, a.saida_limiares_csv, a.saida_limiares_md)
    _salvar_tabela(df_faixas, a.saida_faixas_csv, a.saida_faixas_md)

    print(f"\n=== Tabela Comparativa Principal (limiar={a.limiar}) ===")
    print(df_principal.to_string(index=False))
    print(f"\n=== Avaliação por Limiar ({lista_limiares}) ===")
    print(df_limiares.to_string(index=False))
    print(f"\n=== Distribuição nas 3 Faixas da API (Benigno <= {lim_ben} | Maligno > {lim_mal}) ===")
    print(df_faixas.to_string(index=False))


if __name__ == "__main__":
    main()