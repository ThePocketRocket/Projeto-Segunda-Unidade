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
    dt_model = joblib.load(os.path.join(models_dir, 'decision_tree_model.pkl'))
    knn_model = joblib.load(os.path.join(models_dir, 'knn_model.pkl'))
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    model_columns = joblib.load(os.path.join(models_dir, 'model_columns.pkl'))
    return dt_model, knn_model, scaler, model_columns

dt_model, knn_model, scaler, model_columns = load_models()

# Sidebar para opções
st.sidebar.title("Configurações")
modelo_escolhido = st.sidebar.radio("Escolha o Modelo de Predição", ("Árvore de Decisão", "KNN"))

st.sidebar.markdown("---")

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
    input_data = pd.DataFrame([{
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
        input_data[coluna] = input_data[coluna].map(mapeamento_binario)
        
    # One-Hot Encoding
    colunas_categoricas = input_data.select_dtypes(include=['object', 'str']).columns.tolist()
    input_data = pd.get_dummies(input_data, columns=colunas_categoricas, drop_first=True)
    
    # Garantir que todas as colunas do modelo estejam presentes
    for col in model_columns:
        if col not in input_data.columns:
            input_data[col] = 0
            
    # Reordenar colunas exatamente como no treino (vital para o modelo)
    input_data = input_data[model_columns]
    
    # Inferência
    if modelo_escolhido == "Árvore de Decisão":
        prediction = dt_model.predict(input_data)[0]
    else:
        # KNN precisa do scaler
        input_data_scaled = scaler.transform(input_data)
        prediction = knn_model.predict(input_data_scaled)[0]
        
    st.markdown("---")
    if prediction == 1:
        st.error(f"🚨 **Alerta!** De acordo com o modelo **{modelo_escolhido}**, este cliente possui alto risco de **CHURN** (Cancelar o serviço).")
    else:
        st.success(f"✅ **Tudo Certo!** De acordo com o modelo **{modelo_escolhido}**, este cliente **NÃO** deve cancelar o serviço no momento.")
        st.balloons()
