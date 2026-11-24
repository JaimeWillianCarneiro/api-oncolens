import pandas as pd
import pytest

from src.data import gerar_splits, verificar_sem_vazamento


def _df(n=400):
    return pd.DataFrame({"path": [f"img_{i}.jpg" for i in range(n)],
                         "label": [i % 2 for i in range(n)]})


def test_splits_disjuntos_e_estratificados():
    s = gerar_splits(_df())
    assert set(s["path"]) == set(_df()["path"])
    assert not s["path"].duplicated().any()
    for nome in ("train", "val", "test"):
        assert abs(s[s.split == nome]["label"].mean() - 0.5) < 0.05
    assert (s.split == "test").sum() == 40


def test_splits_deterministicos():
    pd.testing.assert_frame_equal(gerar_splits(_df(), semente=7), gerar_splits(_df(), semente=7))


def test_grupos_nao_cruzam_splits():
    # 4 versões aumentadas por imagem original, todas com o mesmo label
    df = pd.DataFrame({"path": [f"orig{i // 4}_aug{i % 4}.jpg" for i in range(800)],
                       "label": [(i // 4) % 2 for i in range(800)]})
    s = gerar_splits(df, funcao_grupo=lambda p: p.split("_")[0])
    assert (s.groupby("grupo")["split"].nunique() == 1).all()


def test_detecta_vazamento_de_grupo():
    s = gerar_splits(_df(40))
    s["grupo"] = "g"
    with pytest.raises(ValueError):
        verificar_sem_vazamento(s)
