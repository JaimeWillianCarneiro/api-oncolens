# Api Oncolens


OncoLens 4 All is a platform focused on identifying oral lesions that may be potential indicators of head and neck cancer. The platform offers two distinct user experiences: for patients and for healthcare professionals. In both cases, the user uploads a photo of the lesion, and an AI model predicts the likelihood of it being cancerous. For healthcare professionals, the platform also provides a visualization highlighting the key regions that influenced the model's prediction.

Authors of the original project: 
- Glauber Nascimento de Oliveira
- Maria Emily Nayla Gomes
- Julia Guedes Almeida do Santos
- Walléria Simões Correia
- Jaime Willian Carneiro da Silva

Mudanças implementadas em novembro de 2026: Retreino com novos limiares, adicionamos explicabilidade com supernós
- Jaime Willian Carneiro da Silva
- Luiz Eduardo Bravin



### Execute o código à sua maneira

1. Geração dos splits fixos
```
# A partir do arquivo .zip original (extrai e gera splits/splits.csv)
python -m scripts.gerar_splits \
  --zip mhjyrn35p4-2.zip \
  --pasta-dados dados

# Ou a partir da pasta augmented_data já extraída
python -m scripts.gerar_splits \
  --raiz-imagens "dados/Oral Images Dataset/augmented_data" \
  --saida splits/splits.csv

# Split separado para o conjunto original (sem augmentation offline)
python -m scripts.gerar_splits \
  --raiz-imagens "dados/Oral Images Dataset/original_data" \
  --saida splits/splits_original.csv
```
2. Comparação de datasets (Original vs. Augment)
```
python -m scripts.comparar_datasets \
  --splits-augmented splits/splits.csv \
  --raiz-augmented "dados/Oral Images Dataset/augmented_data" \
  --splits-original splits/splits_original.csv \
  --raiz-original "dados/Oral Images Dataset/original_data" \
  --epocas 40 \
  --batch 16
```
3. Avaliação e retreino do baseline
```
python -m scripts.baseline \
  --ckpt models/cnn_geral.ckpt \
  --splits splits/splits.csv \
  --raiz-imagens "dados/Oral Images Dataset/augmented_data" \
  --retreinar \
  --epocas 10 \
  --batch 64
```
4. Otimização de hiperparâmetros com Optuna
```
# Opcional: limpa estudo anterior do SQLite
rm -f optuna_cnn.db

python -m src.otimizacao \
  --splits splits/splits.csv \
  --raiz-imagens "dados/Oral Images Dataset/augmented_data" \
  --batch-sizes 16 32 64 \
  --n-trials 12 \
  --epocas 15 \
  --patience 4 \
  --nome optuna_augmented
```
5. Treinamento final do modelo otimizado
```
python -m scripts.treinar_otimizada \
  --params reports/optuna_best_params.json \
  --splits splits/splits.csv \
  --raiz-imagens "dados/Oral Images Dataset/augmented_data" \
  --epocas 25 \
  --patience 5
```
6. Comparação final de modelos e limiares operacionais
```
# Usando os limiares padrão configurados em src/inference.py (0.16 e 0.461)
python -m scripts.comparar_modelos \
  --splits splits/splits.csv \
  --raiz-imagens "dados/Oral Images Dataset/augmented_data"

# Ou especificando os limiares explicitamente via CLI
python -m scripts.comparar_modelos \
  --splits splits/splits.csv \
  --raiz-imagens "dados/Oral Images Dataset/augmented_data" \
  --limiar-benigno 0.16 \
  --limiar-maligno 0.461
```
7. Execução dos testes automatizados
```
# Suíte completa
python -m pytest -v

# Execução isolada por arquivo
python -m pytest tests/test_modelo.py -v
python -m pytest tests/test_api.py -v
```
