"""Dados do OncoLens: splits fixos (sem vazamento), Dataset e DataModule.

Decisões principais
-------------------
* O split é gerado UMA vez, salvo em CSV (splits/splits.csv) e versionado.
  Todo o resto (baseline, Optuna, modelo final, avaliação) só LÊ esse CSV.
* A lista de imagens é ordenada (sorted) para não depender da ordem do
  sistema de arquivos (glob puro devolve ordem diferente em cada máquina).
* O Optuna só enxerga treino e validação; o teste só é aberto em setup("test").
* O pré-processamento (Resize + ToTensor, SEM Normalize) é o mesmo do treino
  do notebook e do explainability.py. É a fonte única: src/inference.py importa
  daqui.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Optional
from zipfile import ZipFile

import lightning as L
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# --- Constantes (valores vindos do notebook de treinamento / explainability.py)
TAMANHO_IMAGEM = (224, 224)   # explainability.py e notebook 04 usam 224x224
FRACAO_TESTE = 0.1            # notebook: test_size = 1/10
FRACAO_VALIDACAO = 0.1        # notebook: 1/10 do que sobrou após separar o teste
SEMENTE_SPLIT = 1110          # SEMENTE_ALEATORIA do notebook

SUBPASTA_IMAGENS = Path("Oral Images Dataset") / "augmented_data"
PASTA_BENIGNO = "augmented_benign"    # label 0
PASTA_MALIGNO = "augmented_malignant"  # label 1
COLUNAS_SPLIT = ["path", "label", "split"]


# ----------------------------------------------------------------------------
# Preparação / listagem dos dados
# ----------------------------------------------------------------------------
def preparar_dados(caminho_zip: str | Path, pasta_destino: str | Path) -> Path:
    """Extrai o zip do Mendeley uma única vez (o notebook extraía a cada DataModule)."""
    pasta_destino = Path(pasta_destino)
    raiz = pasta_destino / SUBPASTA_IMAGENS
    if not raiz.exists():
        with ZipFile(caminho_zip) as z:
            z.extractall(pasta_destino)
    return raiz


def listar_imagens(raiz_imagens: str | Path) -> pd.DataFrame:
    """DataFrame [path (relativo à raiz), label] em ordem determinística."""
    raiz = Path(raiz_imagens)
    linhas = []
    for pasta, label in ((PASTA_BENIGNO, 0), (PASTA_MALIGNO, 1)):
        for p in sorted((raiz / pasta).glob("*.jpg")):
            linhas.append({"path": p.relative_to(raiz).as_posix(), "label": label})
    if not linhas:
        raise FileNotFoundError(f"Nenhuma .jpg encontrada em {raiz}/{{{PASTA_BENIGNO},{PASTA_MALIGNO}}}")
    return pd.DataFrame(linhas)


# ----------------------------------------------------------------------------
# Splits fixos
# ----------------------------------------------------------------------------
def gerar_splits(
    df: pd.DataFrame,
    semente: int = SEMENTE_SPLIT,
    funcao_grupo: Optional[Callable[[str], str]] = None,
) -> pd.DataFrame:
    """Divide em train/val/test (estratificado por label).

    funcao_grupo: recebe o path relativo e devolve o id da imagem ORIGINAL de
    onde ela veio. Se informada, todas as variações da mesma imagem ficam no
    mesmo split. O dataset usado é o "augmented": sem isso, uma imagem e sua
    versão aumentada podem cair em splits diferentes (vazamento).
    Sem funcao_grupo o split é só estratificado — ver aviso em scripts/gerar_splits.py.
    """
    df = df.reset_index(drop=True).copy()

    if funcao_grupo is None:
        resto, teste = train_test_split(
            df, test_size=FRACAO_TESTE, stratify=df["label"], random_state=semente)
        treino, val = train_test_split(
            resto, test_size=FRACAO_VALIDACAO, stratify=resto["label"], random_state=semente)
    else:
        df["grupo"] = df["path"].map(funcao_grupo)
        n_teste = round(1 / FRACAO_TESTE)
        n_val = round(1 / FRACAO_VALIDACAO)
        # 1 fold de 10 (~10%) vira teste; depois 1 fold de 10 do resto (~10%) vira validação
        idx_resto, idx_teste = next(StratifiedGroupKFold(
            n_splits=n_teste, shuffle=True, random_state=semente
        ).split(df, df["label"], df["grupo"]))
        resto, teste = df.iloc[idx_resto], df.iloc[idx_teste]
        idx_tr, idx_val = next(StratifiedGroupKFold(
            n_splits=n_val, shuffle=True, random_state=semente
        ).split(resto, resto["label"], resto["grupo"]))
        treino, val = resto.iloc[idx_tr], resto.iloc[idx_val]

    partes = [treino.assign(split="train"), val.assign(split="val"), teste.assign(split="test")]
    saida = pd.concat(partes).sort_values("path").reset_index(drop=True)
    verificar_sem_vazamento(saida)
    return saida


def verificar_sem_vazamento(df: pd.DataFrame, raiz_imagens: str | Path | None = None,
                            checar_hash: bool = False) -> dict:
    """Levanta ValueError se houver vazamento detectável. Devolve contagens por split.

    Checa: (1) mesma imagem em dois splits; (2) mesmo grupo em dois splits (se há
    coluna 'grupo'); (3) opcional: arquivos idênticos (md5) em splits diferentes.
    NÃO detecta imagens aumentadas a partir da mesma original sem 'grupo'.
    """
    if df["path"].duplicated().any():
        raise ValueError("Há paths repetidos no CSV de splits.")
    if set(df["split"]) - {"train", "val", "test"}:
        raise ValueError("Valores inválidos na coluna 'split'.")
    if "grupo" in df.columns:
        n_splits_por_grupo = df.groupby("grupo")["split"].nunique()
        if (n_splits_por_grupo > 1).any():
            raise ValueError("Há grupos (imagem original) presentes em mais de um split.")
    if checar_hash:
        if raiz_imagens is None:
            raise ValueError("checar_hash exige raiz_imagens.")
        raiz = Path(raiz_imagens)
        h = df["path"].map(lambda p: hashlib.md5((raiz / p).read_bytes()).hexdigest())
        if (df.assign(h=h).groupby("h")["split"].nunique() > 1).any():
            raise ValueError("Arquivos idênticos (md5) em splits diferentes.")
    return df["split"].value_counts().to_dict()


def salvar_splits(df: pd.DataFrame, caminho: str | Path) -> None:
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho, index=False)


def carregar_splits(caminho: str | Path) -> pd.DataFrame:
    df = pd.read_csv(caminho)
    faltando = set(COLUNAS_SPLIT) - set(df.columns)
    if faltando:
        raise ValueError(f"{caminho} sem colunas {faltando}")
    verificar_sem_vazamento(df)
    return df


# ----------------------------------------------------------------------------
# Dataset / DataModule
# ----------------------------------------------------------------------------
def transform_padrao(tamanho=TAMANHO_IMAGEM) -> transforms.Compose:
    """Resize + ToTensor ([0,1]). Sem Normalize: o treino do notebook e o
    explainability.py trabalham com pixels em [0,1]."""
    return transforms.Compose([transforms.Resize(tamanho), transforms.ToTensor()])


class OralCancerDatasetDF(Dataset):
    def __init__(self, dataframe: pd.DataFrame, raiz_imagens: str | Path, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.raiz = Path(raiz_imagens)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        linha = self.df.iloc[idx]
        imagem = Image.open(self.raiz / linha["path"]).convert("RGB")
        if self.transform:
            imagem = self.transform(imagem)
        return imagem, int(linha["label"])


class DataModule(L.LightningDataModule):
    """Lê o CSV de splits fixos. batch_size agora é parâmetro (antes fixo em 64)."""

    def __init__(self, splits_csv: str | Path, raiz_imagens: str | Path,
                 batch_size: int = 64, num_workers: int = 2, tamanho=TAMANHO_IMAGEM):
        super().__init__()
        self.splits = carregar_splits(splits_csv)
        self.raiz = Path(raiz_imagens)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform_padrao(tamanho)

    def _ds(self, nome: str) -> OralCancerDatasetDF:
        return OralCancerDatasetDF(self.splits[self.splits["split"] == nome], self.raiz, self.transform)

    def setup(self, stage=None):
        if stage in ("fit", None):
            self.train, self.val = self._ds("train"), self._ds("val")
        if stage in ("validate", None):
            self.val = self._ds("val")
        if stage in ("test", None):
            self.test = self._ds("test")

    def _dl(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers,
                          persistent_workers=self.num_workers > 0)

    def train_dataloader(self):
        return self._dl(self.train, True)

    def val_dataloader(self):
        return self._dl(self.val, False)

    def test_dataloader(self):
        return self._dl(self.test, False)
