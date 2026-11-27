import io
import os
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError

from src.explainability import SuperpixelGraphExplainer
from src.inference import carregar_modelo, prever

RAIZ = Path(__file__).resolve().parent
CKPT_PADRAO = RAIZ / "models" / "cnn_otimizada.ckpt"
CKPT_FALLBACK = RAIZ / "models" / "cnn_geral.ckpt"


def _resolver_ckpt(caminho_ckpt: str | Path | None = None) -> Path:
    if caminho_ckpt:
        return Path(caminho_ckpt)
    env_ckpt = os.environ.get("MODEL_PATH")
    if env_ckpt:
        return Path(env_ckpt)
    return CKPT_PADRAO if CKPT_PADRAO.exists() else CKPT_FALLBACK


def create_app(caminho_ckpt: str | Path | None = None, device: str = "cpu") -> Flask:
    app = Flask(__name__)
    CORS(
        app,
        resources={
            r"/analisar": {"origins": "https://jaimewilliancarneiro.github.io"},
            r"/health": {"origins": "*"},
        },
    )

    ckpt_path = _resolver_ckpt(caminho_ckpt)
    modelo = carregar_modelo(ckpt_path, device=device)
    explainer = SuperpixelGraphExplainer(modelo, device=device)

    app.config["MODEL_PATH"] = str(ckpt_path)
    app.config["MODEL_CLASS"] = type(modelo).__name__

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "modelo": app.config["MODEL_CLASS"],
            "checkpoint": Path(app.config["MODEL_PATH"]).name,
        }), 200

    @app.route("/analisar", methods=["POST"])
    def analisar_imagem():
        if "imagem" not in request.files:
            return jsonify({"erro": "Nenhuma imagem enviada"}), 400

        arquivo = request.files["imagem"]
        if not arquivo.filename:
            return jsonify({"erro": "Arquivo de imagem vazio"}), 400

        try:
            conteudo = arquivo.read()
            if not conteudo:
                return jsonify({"erro": "Arquivo de imagem vazio"}), 400
            imagem = Image.open(io.BytesIO(conteudo)).convert("RGB")
        except (UnidentifiedImageError, OSError):
            return jsonify({"erro": "Arquivo enviado não é uma imagem válida"}), 400

        try:
            saida = prever(modelo, imagem)
            resposta = {
                "resultado": saida["classe"],
                "probabilidade": float(saida["probabilidade"]),
            }

            if request.args.get("explicar", "").lower() in {"1", "true", "sim"}:
                exp = explainer.explain(imagem)
                resposta["explicabilidade"] = {
                    "num_superpixels": exp["num_superpixels"],
                    "num_arestas": exp["num_arestas"],
                    "fidelity_drop_top5": float(exp["fidelity_drop_top5"]),
                    "overlay_base64": exp["overlay_base64"],
                }

            return jsonify(resposta), 200
        except Exception as e:
            return jsonify({"erro": f"Erro interno na análise: {e}"}), 500

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)