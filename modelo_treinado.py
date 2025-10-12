# modelo_treinado.py
import lightning as L
from torch import nn
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer
from torchmetrics import Accuracy

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

# minha_cnn = CNN.load_from_checkpoint("melhor-modelo.ckpt")
minha_cnn = CNN.load_from_checkpoint("melhor-modelo.ckpt")
minha_cnn.eval()

minha_cnn.eval()

from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from PIL import Image

image_path = "teste_modelo.jpg"

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
predictions = predict_image(minha_cnn, dataloader)

# Supondo que predictions seja uma lista de tensores
pred_tensor = predictions[0]  # pega a predição do batch único
prob = pred_tensor.item()     # transforma em número

if prob > 0.6:
    classe = "Maligno"
elif prob > 0.4:
    classe = "Inconclusivo"
else:
    prob = "Benigno"

