"""
    python -m pytest tests/test_modelo.py -s
    python -m tests.test_modelo
"""
from pathlib import Path

import pytest

from src.inference import carregar_modelo, prever

RAIZ = Path(__file__).resolve().parents[1]
CKPT = RAIZ / "models" / "cnn_geral.ckpt"
IMAGEM = RAIZ / "src" / "ehcancercaso1.png"


@pytest.mark.skipif(not (CKPT.exists() and IMAGEM.exists()), reason="ckpt ou imagem de exemplo ausentes")
def test_inferencia_retorna_probabilidade_valida():
    modelo = carregar_modelo(CKPT)
    saida = prever(modelo, IMAGEM)
    assert 0.0 <= saida["probabilidade"] <= 1.0
    assert saida["classe"] in {"Benigno", "Inconclusivo", "Maligno"}


@pytest.mark.skipif(not (CKPT.exists() and IMAGEM.exists()), reason="ckpt ou imagem de exemplo ausentes")
def test_inferencia_deterministica():
    modelo = carregar_modelo(CKPT)
    assert prever(modelo, IMAGEM) == prever(modelo, IMAGEM)
from src.inference import rotular


def test_faixas_de_rotulagem():
    assert rotular(0.19) == "Benigno"
    assert rotular(0.20) == "Benigno"
    assert rotular(0.21) == "Inconclusivo"
    assert rotular(0.40) == "Inconclusivo"
    assert rotular(0.41) == "Maligno"

if __name__ == "__main__":
    print(prever(carregar_modelo(CKPT), IMAGEM))