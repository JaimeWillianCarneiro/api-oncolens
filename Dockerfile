# Imagem base Python slim
FROM python:3.11-slim

# Diretório de trabalho
WORKDIR /app

# Copiar requirements
COPY requirements.txt requirements.txt

# Instalar dependências, incluindo PyTorch CPU
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar todo o código
COPY . .

# Expor a porta que o Cloud Run vai usar
EXPOSE 8080

# Variáveis de ambiente
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Comando para rodar o Flask com Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
