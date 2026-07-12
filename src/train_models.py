import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from data_processing import load_and_clean_data, preprocess_data

def train_and_export_models():
    print("Iniciando carregamento e processamento dos dados...")
    # Caminho relativo para o dataset a partir do script
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Telco-Customer-Churn.csv')
    df = load_and_clean_data(data_path)
    df_processed = preprocess_data(df)
    
    # Separando os atributos preditivos e a resposta esperada
    X = df_processed.drop('Churn', axis=1)
    y = df_processed['Churn']
    
    # Salvar as colunas esperadas para a interface
    expected_columns = list(X.columns)
    
    print("Dividindo o dataset em treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Padronizando os dados (necessário para o KNN)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Treinando a Árvore de Decisão
    print("Treinando Árvore de Decisão (max_depth=5)...")
    dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    # A Árvore de Decisão não precisa de dados padronizados
    dt_model.fit(X_train, y_train)
    
    # Treinando o KNN
    print("Treinando KNN (n_neighbors=7, metric='euclidean')...")
    knn_model = KNeighborsClassifier(n_neighbors=7, metric='euclidean')
    # O KNN PRECISA dos dados padronizados
    knn_model.fit(X_train_scaled, y_train)
    
    # Criar pasta de destino se não existir
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    print("Exportando os modelos e artefatos...")
    joblib.dump(dt_model, os.path.join(models_dir, 'decision_tree_model.pkl'))
    joblib.dump(knn_model, os.path.join(models_dir, 'knn_model.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(expected_columns, os.path.join(models_dir, 'model_columns.pkl'))
    
    print("Exportação concluída com sucesso na pasta 'src/models/'!")

if __name__ == "__main__":
    train_and_export_models()
