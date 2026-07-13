import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from data_processing import carregar_dados, preprocessar_dados

def train_and_export_models():
    print("Iniciando carregamento e processamento dos dados...")
    # Caminho relativo para o dataset a partir do script
    caminho_do_dataset = os.path.join(os.path.dirname(__file__), '..', 'data', 'Telco-Customer-Churn.csv')
    df = carregar_dados(caminho_do_dataset)
    df_processado = preprocessar_dados(df)
    
    # Separando os atributos preditivos e a resposta esperada
    atributos_clientes = df_processado.drop('Churn', axis=1)
    rotulos_churn = df_processado['Churn']
    
    # Salvar as colunas esperadas para a interface
    colunas_esperadas = list(atributos_clientes.columns)
    
    print("Dividindo o dataset em treino e teste...")
    atributos_treinamento, atributos_teste, rotulos_treinamento, rotulos_teste = train_test_split(atributos_clientes, rotulos_churn, test_size=0.2, random_state=42)
    
    print("Padronizando os dados (necessário para o KNN)...")
    padronizador_dados = StandardScaler()
    atributos_treinamento_padronizados = padronizador_dados.fit_transform(atributos_treinamento)
    
    # Treinando a Árvore de Decisão
    print("Treinando Árvore de Decisão (max_depth=5)...")
    modelo_arvore_decisao = DecisionTreeClassifier(max_depth=5, random_state=42)
    # A Árvore de Decisão não precisa de dados padronizados
    modelo_arvore_decisao.fit(atributos_treinamento, rotulos_treinamento)
    
    # Treinando o KNN
    print("Treinando KNN (n_neighbors=7, metric='euclidean')...")
    modelo_knn = KNeighborsClassifier(n_neighbors=7, metric='euclidean')
    # O KNN PRECISA dos dados padronizados
    modelo_knn.fit(atributos_treinamento_padronizados, rotulos_treinamento)
    
    # Criar pasta de destino se não existir
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    print("Exportando os modelos e artefatos...")
    joblib.dump(modelo_arvore_decisao, os.path.join(models_dir, 'decision_tree_model.pkl'))
    joblib.dump(modelo_knn, os.path.join(models_dir, 'knn_model.pkl'))
    joblib.dump(padronizador_dados, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(colunas_esperadas, os.path.join(models_dir, 'model_columns.pkl'))
    
    print("Exportação concluída com sucesso na pasta 'src/models/'!")

if __name__ == "__main__":
    train_and_export_models()
