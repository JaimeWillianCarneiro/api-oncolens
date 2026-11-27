import pandas as pd
import pytest
from PIL import Image
from src.data import DataModule, listar_imagens, salvar_splits, gerar_splits, verificar_sem_vazamento


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

def test_listar_imagens_detecta_original_data(tmp_path):
    for sub in ("benign_lesions", "malignant_lesions"):
        (tmp_path / sub).mkdir()
        Image.new("RGB", (32, 32)).save(tmp_path / sub / "amostra.png")
    df = listar_imagens(tmp_path)
    assert len(df) == 2
    assert set(df["label"]) == {0, 1}


def test_augment_apenas_no_treino(tmp_path):
    for sub in ("benign_lesions", "malignant_lesions"):
        (tmp_path / sub).mkdir()
        for i in range(15):
            Image.new("RGB", (32, 32)).save(tmp_path / sub / f"img_{i}.jpg")
    df = listar_imagens(tmp_path)
    splits = gerar_splits(df, semente=42)
    csv_path = tmp_path / "splits.csv"
    salvar_splits(splits, csv_path)

    dm = DataModule(csv_path, tmp_path, batch_size=4, num_workers=0, augment_treino=True)
    dm.setup()
    assert dm.train.transform is dm.transform_treino
    assert dm.val.transform is dm.transform
    assert dm.test.transform is dm.transform