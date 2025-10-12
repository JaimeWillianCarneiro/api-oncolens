import torch
from PIL import Image
from torchvision import transforms
from modelo_treinado import CancerClassifier

# carregar modelo
model = CancerClassifier.load_from_checkpoint("melhor-modelo.ckpt", map_location="cpu")
model.eval()

# transformações (mesmas do app.py)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# carregar imagem
img = Image.open("teste_boca.jpeg").convert("RGB")
x = transform(img).unsqueeze(0)

# inferência
with torch.no_grad():
    y = model(x)
    probs = torch.softmax(y, dim=1)[0]
    print("Saída:", probs)
    print("Classe predita:", torch.argmax(probs).item())
