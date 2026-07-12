# Classificador de Churn de Clientes

Projeto de Machine Learning para predição de cancelamento de serviços (churn). Desenvolvido com KNN e Árvore de Decisão, incluindo pipeline de treino modularizado e interface web interativa.

## Estrutura
- `data/`: Dataset original.
- `notebooks/Treino.ipynb`: Análise exploratória e experimentação inicial.
- `src/data_processing.py`: Limpeza e transformação de dados.
- `src/train_models.py`: Treinamento e exportação (joblib) dos modelos.
- `src/app.py`: Interface da aplicação rodando em Streamlit.

## Setup e Execução

**1. Ambiente Virtual e Dependências**
```bash
python -m venv venv

# Windows (CMD/PowerShell)
.\venv\Scripts\activate
# Windows (Git Bash)
source venv/Scripts/activate

pip install -r requirements.txt
```

**2. Pipeline de Treino**
Antes de testar a interface, é obrigatório treinar os modelos e gerar os binários na pasta `src/models/`.
```bash
python src/train_models.py
```

**3. Interface Web**
Suba o servidor do Streamlit.
```bash
# Windows (CMD/PowerShell)
.\venv\Scripts\streamlit run src\app.py
# Windows (Git Bash)
./venv/Scripts/streamlit run src/app.py
```
A aplicação abrirá automaticamente no navegador em `http://localhost:8501`.
