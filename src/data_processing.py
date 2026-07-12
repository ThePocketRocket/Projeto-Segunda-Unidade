import pandas as pd

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Carrega o arquivo CSV, remove dados duplicados e trata valores nulos.
    """
    df = pd.read_csv(filepath)
    
    # Remove duplicatas
    df.drop_duplicates(inplace=True)
    
    # Converte TotalCharges para numérico (erros viram NaN)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Remove nulos
    df.dropna(inplace=True)
    
    # Remove a coluna inútil
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
        
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte as variáveis categóricas para numéricas (binárias e one-hot encoding).
    """
    df_processed = df.copy()
    
    # Dicionário para o mapeamento
    mapeamento_binario = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
    
    # Colunas binárias mapeadas de volta conforme o notebook original
    colunas_binarias = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    
    # Aplicando o mapeamento
    for coluna in colunas_binarias:
        if coluna in df_processed.columns:
            df_processed[coluna] = df_processed[coluna].map(mapeamento_binario)
            
    # Seleciona as colunas de texto restantes para One-Hot Encoding
    colunas_categoricas = df_processed.select_dtypes(include=['object', 'str']).columns.tolist()
    
    # Aplica o One-Hot Encoding
    df_processed = pd.get_dummies(df_processed, columns=colunas_categoricas, drop_first=True)
    
    # Convertendo os booleanos gerados para 1 e 0 (int)
    df_processed = df_processed.astype(int)
    
    return df_processed
