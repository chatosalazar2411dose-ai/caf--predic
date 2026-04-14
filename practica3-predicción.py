import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, auth

# Configuración de la página
st.set_page_config(
    page_title="☕ Predictor de Calidad de Café",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== INICIALIZAR SESSION STATE ==========
def init_session_state():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'user_email' not in st.session_state:
        st.session_state.user_email = None
    if 'user_name' not in st.session_state:
        st.session_state.user_name = None

init_session_state()

# ========== CONFIGURACIÓN DE FIREBASE ==========
FIREBASE_API_KEY = "AIzaSyCOv_kboRAeWJnymX4JYDqQAZu5kV8eYww"

def init_firebase_admin():
    """Inicializa Firebase Admin SDK"""
    try:
        if not firebase_admin._apps:
            # Intentar cargar desde secrets
            try:
                cred_dict = dict(st.secrets["firebase_auth_token"])
                cred = credentials.Certificate(cred_dict)
                firebase_admin.initialize_app(cred)
                return True
            except:
                # Modo demo si no hay secrets
                st.warning("Modo demostración - Firebase no configurado")
                return False
        return True
    except Exception as e:
        st.warning(f"Firebase no disponible: {e}")
        return False

def authenticate_user(email, password):
    """Autentica usuario usando REST API de Firebase"""
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            error = response.json().get("error", {}).get("message", "Error")
            st.error(f"Error: {error}")
            return None
    except Exception as e:
        st.error(f"Error de conexión: {e}")
        return None

def register_user(email, password, name):
    """Registra un nuevo usuario"""
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={FIREBASE_API_KEY}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            user_data = response.json()
            st.session_state.logged_in = True
            st.session_state.user_email = email
            st.session_state.user_name = name
            return True, "✅ Registro exitoso"
        else:
            error = response.json().get("error", {}).get("message", "Error")
            if error == "EMAIL_EXISTS":
                return False, "❌ El email ya está registrado"
            elif error == "WEAK_PASSWORD":
                return False, "❌ La contraseña debe tener al menos 6 caracteres"
            return False, f"❌ Error: {error}"
    except Exception as e:
        return False, f"❌ Error de conexión: {e}"

def logout_user():
    """Cierra sesión"""
    for key in ['logged_in', 'user_id', 'user_email', 'user_name']:
        if key in st.session_state:
            st.session_state[key] = None if key != 'logged_in' else False
    st.rerun()

# ========== INTERFAZ DE LOGIN ==========
def show_login_ui():
    st.markdown("""
    <h1 style='text-align: center; color: #6F4E37;'>
        ☕ Sistema Experto para Predicción de Calidad de Café
    </h1>
    <hr style='border: 2px solid #6F4E37;'>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔐 Iniciar Sesión")
        email = st.text_input("Email", placeholder="demo@cafe.com", key="login_email")
        password = st.text_input("Contraseña", type="password", placeholder="••••••••", key="login_pass")
        
        if st.button("Iniciar Sesión", use_container_width=True):
            if email and password:
                user_data = authenticate_user(email, password)
                if user_data:
                    st.session_state.logged_in = True
                    st.session_state.user_email = email
                    st.session_state.user_name = email.split('@')[0]
                    st.success("✅ Login exitoso")
                    st.rerun()
            else:
                st.warning("Ingresa email y contraseña")
        
        st.caption("🔧 Demo: demo@cafe.com / demo123 (debes crearlo primero)")
    
    with col2:
        st.subheader("📝 Registrarse")
        new_name = st.text_input("Nombre completo", placeholder="Tu nombre", key="reg_name")
        new_email = st.text_input("Email", placeholder="tu@email.com", key="reg_email")
        new_password = st.text_input("Contraseña", type="password", placeholder="Mínimo 6 caracteres", key="reg_pass")
        confirm_password = st.text_input("Confirmar contraseña", type="password", key="reg_confirm")
        
        if st.button("Registrarse", use_container_width=True):
            if not new_name:
                st.warning("Ingresa tu nombre")
            elif not new_email:
                st.warning("Ingresa tu email")
            elif not new_password:
                st.warning("Ingresa una contraseña")
            elif len(new_password) < 6:
                st.warning("La contraseña debe tener al menos 6 caracteres")
            elif new_password != confirm_password:
                st.warning("Las contraseñas no coinciden")
            else:
                success, message = register_user(new_email, new_password, new_name)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

# ========== APLICACIÓN PRINCIPAL ==========
def main_app():
    """Aplicación principal después del login"""
    
    with st.sidebar:
        st.image("https://em-content.zobj.net/thumbs/120/apple/354/hot-beverage_2615.png", width=100)
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 15px; border-radius: 10px; color: white; margin-bottom: 20px;'>
            <small>👤 <strong>{st.session_state.get('user_name', 'Usuario')}</strong></small><br>
            <small>📧 {st.session_state.get('user_email', '')}</small>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚪 Cerrar Sesión", use_container_width=True):
            logout_user()
        
        st.markdown("---")
        st.header("⚙️ Panel de Control")
        
        opcion_datos = st.radio(
            "📊 Fuente de datos:",
            ["📁 Datos de ejemplo", "📂 Cargar CSV"]
        )
        
        st.subheader("🔧 Parámetros del Modelo")
        test_size = st.slider("Tamaño del conjunto de prueba:", 0.1, 0.5, 0.3, 0.05)
        random_state = st.number_input("Semilla aleatoria:", 0, 100, 42)
    
    st.markdown("""
        <h1 style='text-align: center; color: #6F4E37;'>
            ☕ Sistema Experto para Predicción de Calidad de Café
        </h1>
        <hr style='border: 2px solid #6F4E37;'>
    """, unsafe_allow_html=True)
    
    st.success(f"✨ ¡Bienvenido/a {st.session_state.get('user_name', 'Usuario')}!")
    
    # Cargar datos
    if opcion_datos == "📂 Cargar CSV":
        archivo = st.file_uploader("Seleccionar archivo CSV", type=['csv'])
        if archivo:
            df = pd.read_csv(archivo)
        else:
            df = None
    else:
        data = {
            'altitud_msnm': [800, 950, 1100, 1250, 1400, 1550, 1700, 1850, 2000, 2150],
            'temp_promedio_c': [28.5, 27.2, 25.8, 24.3, 22.7, 21.2, 19.8, 18.3, 16.9, 15.4],
            'puntaje_calidad_1_10': [3.2, 4.8, 6.1, 7.2, 8.1, 8.7, 9.1, 9.4, 9.7, 9.9]
        }
        df = pd.DataFrame(data)
    
    if df is None:
        data = {
            'altitud_msnm': [800, 950, 1100, 1250, 1400, 1550, 1700, 1850, 2000, 2150],
            'temp_promedio_c': [28.5, 27.2, 25.8, 24.3, 22.7, 21.2, 19.8, 18.3, 16.9, 15.4],
            'puntaje_calidad_1_10': [3.2, 4.8, 6.1, 7.2, 8.1, 8.7, 9.1, 9.4, 9.7, 9.9]
        }
        df = pd.DataFrame(data)
    
    # Modelo
    X = df[['altitud_msnm', 'temp_promedio_c']]
    y = df['puntaje_calidad_1_10']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Datos", "🤖 Modelo", "📈 Visualizaciones", "🔮 Predicciones"])
    
    with tab1:
        st.dataframe(df)
        st.dataframe(df.describe())
    
    with tab2:
        r2_test = r2_score(y_test, model.predict(X_test))
        st.metric("R² Score", f"{r2_test:.3f}")
        st.info(f"Calidad = {model.intercept_:.2f} + ({model.coef_[0]:.2f}×Altitud) + ({model.coef_[1]:.2f}×Temperatura)")
    
    with tab3:
        fig = px.scatter_3d(df, x='altitud_msnm', y='temp_promedio_c', z='puntaje_calidad_1_10', color='puntaje_calidad_1_10')
        st.plotly_chart(fig)
    
    with tab4:
        st.subheader("🔮 Predecir Calidad")
        
        col1, col2 = st.columns(2)
        with col1:
            altitud = st.number_input("Altitud (msnm)", 0.0, 3000.0, 1650.0)
        with col2:
            temperatura = st.number_input("Temperatura (°C)", 10.0, 35.0, 20.0)
        
        if st.button("Predecir", type="primary"):
            prediccion_raw = model.predict([[altitud, temperatura]])[0]
            prediccion = max(0.0, min(10.0, prediccion_raw))
            
            st.metric("Calidad Predicha", f"{prediccion:.2f}/10")
            st.progress(prediccion/10)
            
            if prediccion >= 8:
                st.success("🌟 Café de Excelencia")
            elif prediccion >= 6:
                st.info("✅ Café de Buena Calidad")
            else:
                st.warning("⚠️ Requiere Mejoras")

# ========== PUNTO DE ENTRADA ==========
if not st.session_state.logged_in:
    show_login_ui()
else:
    main_app()
