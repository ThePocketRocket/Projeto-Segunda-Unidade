import streamlit as st
import pandas as pd
import joblib
import os

# Configuração da página
st.set_page_config(page_title="Previsão de Churn", page_icon="📊", layout="wide")

# Carregamento dos artefatos
@st.cache_resource
def load_models():
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    modelo_knn = joblib.load(os.path.join(models_dir, 'knn_model.pkl'))
    padronizador_dados = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    colunas_modelo = joblib.load(os.path.join(models_dir, 'model_columns.pkl'))
    return modelo_knn, padronizador_dados, colunas_modelo

modelo_knn, padronizador_dados, colunas_modelo = load_models()

st.title("Sistema Inteligente de Classificação - Churn de Clientes")
st.markdown("Insira os dados do cliente abaixo para prever a probabilidade de cancelamento do serviço.")

# Layout do formulário com colunas
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Dados Demográficos")
    gender = st.selectbox("Gênero", ["Male", "Female"])
    senior_citizen = st.selectbox("Idoso (Senior Citizen)?", ["Não", "Sim"])
    partner = st.selectbox("Possui Parceiro(a)?", ["Yes", "No"])
    dependents = st.selectbox("Possui Dependentes?", ["Yes", "No"])

with col2:
    st.subheader("Serviços Contratados")
    phone_service = st.selectbox("Serviço de Telefone", ["Yes", "No"])
    multiple_lines = st.selectbox("Múltiplas Linhas", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Serviço de Internet", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Segurança Online", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Backup Online", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Proteção de Dispositivo", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Suporte Técnico", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming de TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming de Filmes", ["Yes", "No", "No internet service"])

with col3:
    st.subheader("Contrato e Faturamento")
    contract = st.selectbox("Tipo de Contrato", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Faturamento sem Papel", ["Yes", "No"])
    payment_method = st.selectbox("Método de Pagamento", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    tenure = st.number_input("Meses de Contrato (tenure)", min_value=0, max_value=100, value=1)
    monthly_charges = st.number_input("Cobrança Mensal", min_value=0.0, value=50.0)
    total_charges = st.number_input("Cobrança Total", min_value=0.0, value=50.0)

# Processamento e Predição
if st.button("Prever", use_container_width=True, type="primary"):
    # Montar o dataframe de input
    dados_entrada = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == "Sim" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }])

    # Transformações idênticas ao data_processing.py
    mapeamento_binario = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
    colunas_binarias = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    
    for coluna in colunas_binarias:
        dados_entrada[coluna] = dados_entrada[coluna].map(mapeamento_binario)
        
    # One-Hot Encoding
    colunas_categoricas = dados_entrada.select_dtypes(include=['object', 'str']).columns.tolist()
    dados_entrada = pd.get_dummies(dados_entrada, columns=colunas_categoricas, drop_first=True)
    
    # Garantir que todas as colunas do modelo estejam presentes
    for col in colunas_modelo:
        if col not in dados_entrada.columns:
            dados_entrada[col] = 0
            
    # Reordenar colunas exatamente como no treino (vital para o modelo)
    dados_entrada = dados_entrada[colunas_modelo]
    
    # Inferência
    dados_entrada_padronizados = padronizador_dados.transform(dados_entrada)
    previsao = modelo_knn.predict(dados_entrada_padronizados)[0]
        
    st.markdown("---")
    if previsao == 1:
        st.error("🚨 **Alerta!** De acordo com o modelo KNN, este cliente possui alto risco de **CHURN** (Cancelar o serviço).")
    else:
        st.success("✅ **Tudo Certo!** De acordo com o modelo KNN, este cliente **NÃO** deve cancelar o serviço no momento.")
        st.balloons()
