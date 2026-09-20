# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import io
from src.modelo_treinado import CNN  

app = Flask(__name__)

CORS(app, resources={r"/analisar": {"origins": "https://jaimewilliancarneiro.github.io"}})

# ===============================
# 1️⃣ Carregar modelo treinado
# ===============================
model = CNN.load_from_checkpoint("cnn_geral.ckpt", map_location=torch.device("cpu"))
model.eval()

# ===============================
# 2️⃣ Transformações iguais às do treino
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ajusta ao tamanho esperado
    transforms.ToTensor(),
])

# ===============================
# 3️⃣ Rota para análise
# ===============================
@app.route('/analisar', methods=['POST'])
def analisar_imagem():
    if 'imagem' not in request.files:
        return jsonify({'erro': 'Nenhuma imagem enviada'}), 400

    try:
        arquivo = request.files['imagem']
        imagem = Image.open(io.BytesIO(arquivo.read())).convert("RGB")

        # aplica transformações
        tensor = transform(imagem).unsqueeze(0)  # [1, C, H, W]

        with torch.no_grad():
            prob = model(tensor).item()

        # interpreta o resultado
        if prob > 0.6:
            classe = "Maligno"
        elif prob > 0.4:
            classe = "Inconclusivo"
        else:
            classe = "Benigno"

        return jsonify({
            "resultado": classe,
            "probabilidade": float(prob)
        })

    except Exception as e:
        print("Erro na análise:", e)
        return jsonify({"erro": str(e)}), 500

# # ===============================
# # 4️⃣ Rodar servidor
# # ===============================
# if __name__ == "__main__":
#     import os
#     port = int(os.environ.get("PORT", 8080))  # pega a porta do Cloud Run ou usa 8080
#     app.run(host="0.0.0.0", port=port)
