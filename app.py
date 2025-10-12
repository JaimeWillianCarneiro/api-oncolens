from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import io
from modelo_treinado import CNN  # importa sua classe definida

app = Flask(__name__)
CORS(app)  # 🔥 permite o React acessar a API

# ==============================================================
# 1️⃣ CARREGAR O MODELO
# ==============================================================

# Cria instância do modelo exatamente como foi treinado
model = CNN.load_from_checkpoint("melhor-modelo.ckpt", map_location=torch.device("cpu"))
model.eval()

# ==============================================================
# 2️⃣ DEFINIR TRANSFORMAÇÕES IGUAIS ÀS DO TREINAMENTO
# ==============================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ==============================================================
# 3️⃣ ROTA DE ANÁLISE DE IMAGEM
# ==============================================================
@app.route("/analisar", methods=["POST"])
def analisar_imagem():
    try:
        if "imagem" not in request.files:
            return jsonify({"erro": "Nenhuma imagem enviada"}), 400

        arquivo = request.files["imagem"]
        imagem = Image.open(io.BytesIO(arquivo.read())).convert("RGB")

        # aplica transformações
        tensor = transform(imagem).unsqueeze(0)  # [1, C, H, W]

        # roda o modelo
        with torch.no_grad():
            saida = model(tensor)
            probabilidades = torch.softmax(saida, dim=1)[0]
            prob, classe_predita = torch.max(probabilidades, 0)

        # mapeia o índice para nome da classe
        # (ajuste conforme seu modelo)
        classes = ["Sem indícios de câncer", "Possível lesão detectada"]
        resultado = classes[classe_predita.item()]
        probabilidade = float(prob.item())

        return jsonify({
            "resultado": resultado,
            "probabilidade": probabilidade
        })

    except Exception as e:
        print("Erro:", e)
        return jsonify({"erro": str(e)}), 500

# ==============================================================
# 4️⃣ RODAR SERVIDOR
# ==============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
