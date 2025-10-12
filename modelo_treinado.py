import lightning as L
from torch import nn
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer
from torchmetrics import Accuracy
from torchvision import transforms
# Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

TAXA_DE_APRENDIZADO = 0.001

checkpoint_callback = ModelCheckpoint(
    monitor="valid_loss",  # aqui deve bater com self.log
    dirpath="checkpoints",
    filename="melhor-modelo",
    save_top_k=1,
    mode="min"
)


class CNN(L.LightningModule):

    def __init__(
            self,
            cnn_out_channels=None, #filtros ou canais de saída
            n_lables: int = 2      # Modelo binário
    ):
        super().__init__() #inicializa com atributos já existentes no Lightning

        if cnn_out_channels is None:
            cnn_out_channels = [16, 32, 64] #determina o número padrão de filtros em cada camada
            
        #acurácia em cada etapa
        self.train_acc = Accuracy(task="binary")
        self.valid_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")

        in_channels = 3 #número de entradas por imagem - começa com ela inteira

        cnn_block = list() # lista para armazenar cada camada
        for out_channel in cnn_out_channels: #para cada camada
            cnn_block.append(
                nn.Conv2d( # camada convolucional
                    in_channels=in_channels, #entrada
                    out_channels=out_channel, # filtros aplicados
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cnn_block.append(nn.ReLU()) #função de ativação
            cnn_block.append(nn.MaxPool2d((2, 2))) # camada de pooling - reduz as dimensões
            in_channels = out_channel # para a próxima camada, a entrada será p número de filtros aplicados

        self.cnn_block = nn.Sequential(*cnn_block) # agrupa todas as camadas em sequência
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # reduz para 1x1 independentemente do tamanho da imagem
            nn.Flatten(),
            nn.Linear(cnn_out_channels[-1], 1),
            nn.Sigmoid()
        )
        
        #listas para salvar as labels na etapa de teste - serão usadas para visualização dos resultados
        self.y_true = []
        self.y_pred = []

    def forward(self, x): # passa os dados pela rede
        # é pré-determinado como __call__ do pytorch
        x = self.cnn_block(x)
        return self.head(x)

    def training_step(self, batch, batch_idx): #etapa de treino em lotes
        x, y = batch #separa atributo e target
        logits = self(x) #forward
        
        # CORREÇÃO: Ajustar shapes para serem compatíveis
        logits = logits.squeeze()  # Remove dimensão extra: [64, 1] -> [64]
        y = y.float()  # Garantir que y seja float
        
        loss_function = nn.BCELoss()
        loss = loss_function(logits, y)
        
        preds = (logits > 0.5).float() # previsão de classes: a que tem maior probabilidade
        
        self.train_acc.update(preds, y) #atualiza o valor da acurácia da classificação
        self.log("train_loss", loss, prog_bar=True) #mostra erro no terminal
        return loss #retorna o erro

    def on_training_epoch_end(self): #camada final de treinamento de uma época
        self.log("train_acc", self.train_acc.compute()) #cálcula a acurácia total da época
        self.train_acc.reset() #reseta a acurácia para a próxima época

    def validation_step(self, batch, batch_idx): #etapa de validação em lotes: avalia se o modelo está aprendendo ou não a cada época
        x, y = batch #separa atributo e target
        logits = self(x) #forward
        
        # CORREÇÃO: Ajustar shapes para serem compatíveis
        logits = logits.squeeze()  # Remove dimensão extra: [64, 1] -> [64]
        y = y.float()  # Garantir que y seja float
        
        loss_function = nn.BCELoss()
        loss = loss_function(logits, y)
        
        preds = (logits > 0.5).float() # previsão de classes: a que tem maior probabilidade
        self.valid_acc.update(preds, y) #atualiza o valor da acurácia da classificação
        self.log("valid_loss", loss, prog_bar=True) #mostra erro no terminal
        return loss #retorna o erro

    def on_validation_epoch_end(self): #camada final de validação de uma época
        self.log("valid_acc", self.valid_acc.compute(), prog_bar=True) #cálcula a acurácia total da época
        self.valid_acc.reset() #reseta a acurácia para a próxima época

    def test_step(self, batch, batch_idx): #etapa de treino em lotes:
        x, y = batch  #separa atributo e target
        #sava os dados caso se deseje visualizá-los depois
        self.x_test_data = x #atributos 
        self.y_test_data = y #target
        logits = self(x) #forward
        
        # CORREÇÃO: Ajustar shapes para serem compatíveis
        logits = logits.squeeze()  # Remove dimensão extra: [64, 1] -> [64]
        y = y.float()  # Garantir que y seja float
        
        loss_function = nn.BCELoss()
        loss = loss_function(logits, y)

        self.preds = (logits > 0.5).float()  # previsão de classes: a que tem maior probabilidade
        self.test_acc.update(self.preds, y) #atualiza o valor da acurácia da classificação
        self.log("test_loss", loss, prog_bar=True)  #mostra erro no terminal
        self.log("test_acc", self.test_acc.compute(), prog_bar=True)  #mostra acurácia no terminal
        
        self.y_true.extend(y.tolist()) #salva as labels reais
        self.y_pred.extend(self.preds.tolist()) #salva as labels previstas
        
        return loss #retorna o erro

    def configure_optimizers(self): #calcula e altera os gradientes para todos os parâmetros treináveis da rede, de acordo com a  taxa de aprendizado
        optimizer = torch.optim.Adam(self.parameters(), lr=TAXA_DE_APRENDIZADO)
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)


TAMANHO_VALIDACAO = 1/9
SEMENTE_ALEATORIA = 1110

class DataModule(L.LightningDataModule):
    def __init__(self, data_path: str = './'):
        super().__init__()
        self.data_path = data_path

        # transformação (tensorização)
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    ##########################################################
    #     Baixando os dados
    ##########################################################

        # descompacta os dados
        caminho_zip = 'mhjyrn35p4-2.zip'
        pasta_dados = "dados"
        try:
            with ZipFile(caminho_zip, 'r') as zip_obj:
                zip_obj.extractall(pasta_dados)
            print(f"Arquivos extraídos em '{pasta_dados}'.")
        except Exception as e:
            print(f"Erro ao descompactar: {e}")

        # monta DataFrame
        caminho_ben = os.path.join(pasta_dados, "Oral Images Dataset", "augmented_data", "augmented_benign", "*.jpg")
        caminho_mal = os.path.join(pasta_dados, "Oral Images Dataset", "augmented_data", "augmented_malignant", "*.jpg")

        imagens_ben = glob.glob(caminho_ben)
        imagens_mal = glob.glob(caminho_mal)

        df_ben = pd.DataFrame({"path": imagens_ben, "label": 0})
        df_mal = pd.DataFrame({"path": imagens_mal, "label": 1})
        self.df = pd.concat([df_ben, df_mal], ignore_index=True)


    def setup(self, stage=None):
        # Divide o dataset uma única vez
        df_train, df_test = train_test_split(
            self.df, test_size=TAMANHO_VALIDACAO, random_state=SEMENTE_ALEATORIA
        )
        df_train, df_val = train_test_split(
            df_train, test_size=TAMANHO_VALIDACAO, random_state=SEMENTE_ALEATORIA
        )
    
        # Cria os datasets conforme o estágio
        if stage == "fit" or stage is None:
            self.train = OralCancerDatasetDF(df_train, transform=self.transform)
            self.val = OralCancerDatasetDF(df_val, transform=self.transform)
    
        if stage == "test" or stage is None:
            self.test = OralCancerDatasetDF(df_test, transform=self.transform)


    def train_dataloader(self):
        return DataLoader(self.train, batch_size=64, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=64, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=64, num_workers=0)


modelo_geral = CNN.load_from_checkpoint("checkpoints/melhor-modelo-v2.ckpt") ##################################################

from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from PIL import Image

image_path = "teste_modelo.png"

# Define o mesmo pré-processamento usado no treino
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Carrega e transforma a imagem
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0)  # adiciona dimensão batch

# Coloca no TensorDataset
dataset = TensorDataset(image_tensor)
dataloader = DataLoader(dataset, batch_size=1)

trainer = Trainer(accelerator="auto", devices=1, logger=False)

# Função para predição corrigida
def predict_image(model, dataloader):
    predictions = []
    for batch in dataloader:
        x = batch[0]  # descompacta o tensor da tupla
        preds = model(x)  # passa pelo modelo
        predictions.append(preds)
    return predictions

# Faz a predição
predictions = predict_image(modelo_geral, dataloader)

# Supondo que predictions seja uma lista de tensores
pred_tensor = predictions[0]  # pega a predição do batch único
prob = pred_tensor.item()     # transforma em número

if prob > 0.6:
    classe = "Maligno"
elif prob > 0.4:
    classe = "Inconclusivo"
else:
    classe = "Benigno"

