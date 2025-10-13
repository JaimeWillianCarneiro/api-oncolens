# ===============================
# Etapa 1: Imagem base
# ===============================
FROM python:3.10-slim

# ===============================
# Etapa 2: Diretório de trabalho e variáveis de ambiente
# ===============================
WORKDIR /app
ENV PYTHONUNBUFFERED=1

# ===============================
# Etapa 3: Instalar dependências
# ===============================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# ===============================
# Etapa 4: Copiar TODOS os arquivos da sua aplicação
# ===============================
# Este comando copia app.py, modelo_treinado.py, cnn_geral.ckpt, etc.
COPY . .

# ===============================
# Etapa 5: Expor porta e usar Gunicorn para iniciar
# ===============================
EXPOSE 8080
# O Gunicorn é um servidor de produção. Esta é a forma correta de iniciar a API.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "0", "app:app"]