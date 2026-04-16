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
    if 'show_history' not in st.session_state:
        st.session_state.show_history = False
    if 'show_recommendations' not in st.session_state:
        st.session_state.show_recommendations = False
    if 'show_register' not in st.session_state:
        st.session_state.show_register = False
    if 'modo' not in st.session_state:
        st.session_state.modo = "visualizacion"
    if 'df_editado' not in st.session_state:
        st.session_state.df_editado = None
    if 'tipo_grafica' not in st.session_state:
        st.session_state.tipo_grafica = "scatter_3d"
    if 'color_scheme' not in st.session_state:
        st.session_state.color_scheme = "Viridis"
    if 'tamaño_puntos' not in st.session_state:
        st.session_state.tamaño_puntos = 8
    if 'tipo_residuos' not in st.session_state:
        st.session_state.tipo_residuos = "Puntos"
    if 'mostrar_linea' not in st.session_state:
        st.session_state.mostrar_linea = True

init_session_state()

# ========== CONFIGURACIÓN DE FIREBASE ==========
FIREBASE_API_KEY = "AIzaSyCOv_kboRAeWJnymX4JYDqQAZu5kV8eYww"

def init_firebase_admin():
    """Inicializa Firebase Admin SDK"""
    try:
        if not firebase_admin._apps:
            try:
                cred_dict = dict(st.secrets["firebase_auth_token"])
                cred = credentials.Certificate(cred_dict)
                firebase_admin.initialize_app(cred)
                return True
            except:
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
    for key in ['logged_in', 'user_id', 'user_email', 'user_name', 'show_history', 'show_recommendations']:
        if key in st.session_state:
            st.session_state[key] = None if key not in ['logged_in', 'show_history', 'show_recommendations'] else False
    st.rerun()

# ========== INTERFAZ DE LOGIN MODIFICADA ==========
def show_login_ui():
    """Muestra la interfaz de login/registro con botones alternativos"""
    
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, black 0%, black 100%);
    }
    .login-header {
        text-align: center;
        padding: 20px;
        animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="login-header">
        <h1 style='color: #6F4E37; font-size: 3em;'>☕ Sistema Experto para Predicción de Calidad de Café</h1>
        <hr style='border: 2px solid #6F4E37; width: 50%; margin: auto;'>
        <p style='color: #6F4E37; font-size: 1.2em; margin-top: 20px;'>Sistema experto para evaluación de calidad de café</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.show_register:
        # ========== FORMULARIO DE LOGIN ==========
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div style='background: black; padding: 30px; border-radius: 15px; box-shadow: 0 5px 20px rgba(0,0,0,0.1); border: 1px solid #6F4E37;'>
            """, unsafe_allow_html=True)
            st.subheader("🔐 Iniciar Sesión")
            
            email = st.text_input("📧 Email", placeholder="tu@email.com", key="login_email")
            password = st.text_input("🔒 Contraseña", type="password", placeholder="••••••••", key="login_pass")
            
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("🚪 Iniciar Sesión", use_container_width=True):
                    if email and password:
                        user_data = authenticate_user(email, password)
                        if user_data:
                            st.session_state.logged_in = True
                            st.session_state.user_email = email
                            st.session_state.user_name = email.split('@')[0]
                            st.success("✅ Login exitoso")
                            st.rerun()
                    else:
                        st.warning("⚠️ Ingresa email y contraseña")
            
            with col_btn2:
                if st.button("📝 ¿No tienes cuenta? Regístrate", use_container_width=True):
                    st.session_state.show_register = True
                    st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        # ========== FORMULARIO DE REGISTRO ==========
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div style='background: black; padding: 30px; border-radius: 15px; box-shadow: 0 5px 20px rgba(0,0,0,0.1); border: 1px solid #6F4E37;'>
            """, unsafe_allow_html=True)
            st.subheader("📝 Crear una cuenta nueva")
            
            new_name = st.text_input("👤 Nombre completo", placeholder="Tu nombre", key="reg_name")
            new_email = st.text_input("📧 Email", placeholder="tu@email.com", key="reg_email")
            new_password = st.text_input("🔒 Contraseña", type="password", placeholder="Mínimo 6 caracteres", key="reg_pass")
            confirm_password = st.text_input("🔒 Confirmar contraseña", type="password", key="reg_confirm")
            
            col_btn_reg1, col_btn_reg2 = st.columns(2)
            
            with col_btn_reg1:
                if st.button("✅ Crear cuenta", use_container_width=True):
                    if not new_name:
                        st.warning("⚠️ Ingresa tu nombre")
                    elif not new_email:
                        st.warning("⚠️ Ingresa tu email")
                    elif not new_password:
                        st.warning("⚠️ Ingresa una contraseña")
                    elif len(new_password) < 6:
                        st.warning("⚠️ La contraseña debe tener al menos 6 caracteres")
                    elif new_password != confirm_password:
                        st.warning("⚠️ Las contraseñas no coinciden")
                    else:
                        with st.spinner("Creando usuario..."):
                            success, message = register_user(new_email, new_password, new_name)
                            if success:
                                st.success("✅ ¡Registro exitoso! Ahora inicia sesión.")
                                import time
                                time.sleep(2)
                                st.session_state.show_register = False
                                st.rerun()
                            else:
                                st.error(message)
            
            with col_btn_reg2:
                if st.button("🔙 Volver al inicio de sesión", use_container_width=True):
                    st.session_state.show_register = False
                    st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)

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
            ["📁 Datos de ejemplo", "📂 Cargar CSV"],
            help="Selecciona cómo quieres cargar los datos"
        )
        
        st.subheader("🔧 Parámetros del Modelo")
        test_size = st.slider("Tamaño del conjunto de prueba:", 0.1, 0.5, 0.3, 0.05, help="Proporción de datos para prueba")
        random_state = st.number_input("Semilla aleatoria:", 0, 100, 42, help="Para reproducibilidad de resultados")
        
        # Selector de modo de visualización
        st.markdown("---")
        st.subheader("🎨 Modo de Visualización")
        
        modo_visualizacion = st.radio(
            "Selecciona el modo:",
            ["📊 Modo Visualización", "✏️ Modo Edición"],
            help="Visualización: solo ver datos | Edición: modificar valores y personalizar gráficas"
        )
        
        st.session_state.modo = "edicion" if "Edición" in modo_visualizacion else "visualizacion"
        
        if st.session_state.modo == "edicion":
            st.markdown("---")
            st.subheader("🎨 Personalización de Gráficas")
            
            st.session_state.tipo_grafica = st.selectbox(
                "Tipo de gráfica 3D:",
                ["scatter_3d", "surface", "line_3d"]
            )
            
            st.session_state.color_scheme = st.selectbox(
                "Esquema de color:",
                ["Viridis", "Plasma", "Inferno", "Magma", "Cividis"]
            )
            
            st.session_state.tamaño_puntos = st.slider("Tamaño de puntos:", 3, 15, st.session_state.tamaño_puntos)
            st.session_state.tipo_residuos = st.radio(
                "Tipo de gráfico de residuos:",
                ["Puntos", "Barras", "Línea"],
                horizontal=True
            )
            st.session_state.mostrar_linea = st.checkbox("Mostrar línea de tendencia", value=st.session_state.mostrar_linea)
        
        st.markdown("---")
        st.subheader("📋 Opciones adicionales")
        
        if st.button("📜 Ver historial de predicciones", use_container_width=True):
            st.session_state.show_history = True
        if st.button("💡 Ver recomendaciones", use_container_width=True):
            st.session_state.show_recommendations = True
        
        st.markdown("---")
        st.markdown("""
        <div style='background: #f0f2f6; padding: 10px; border-radius: 5px; text-align: center;'>
            <small>ℹ️ Los resultados deben ser validados por un experto en café</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Título principal
    st.markdown("""
    <style>
    @keyframes slideIn {
        from { transform: translateX(-50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    .main-title {
        animation: slideIn 0.5s ease-out;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-title">
        <h1 style='color: #6F4E37;'>☕ Sistema Experto para Predicción de Calidad de Café</h1>
        <hr style='border: 2px solid #6F4E37;'>
    </div>
    """, unsafe_allow_html=True)
    
    st.success(f"✨ ¡Bienvenido/a {st.session_state.get('user_name', 'Usuario')}! ☕")
    
    # ========== CARGA DE DATOS ADAPTADA PARA TU CSV ==========
    if opcion_datos == "📂 Cargar CSV":
        archivo = st.file_uploader("Seleccionar archivo CSV", type=['csv'])
        if archivo:
            df = pd.read_csv(archivo)
            st.success(f"✅ {len(df)} registros cargados exitosamente")
            
            # Mostrar las columnas detectadas
            st.info(f"📋 Columnas detectadas: {', '.join(df.columns.tolist())}")
        else:
            st.warning("⚠️ Usando datos de ejemplo por defecto")
            df = None
    else:
        # Datos de ejemplo adaptados al formato de tu CSV
        data = {
            'temperatura': [20.6, 29.3, 25.9, 24.0, 17.3, 17.3, 15.9, 28.0, 24.0, 25.6],
            'altitud': [1060.5, 999.4, 764.2, 1410.9, 1214.9, 1798.6, 548.2, 1465.8, 1644.4, 1639.2],
            'puntaje': [10.0, 9.61, 10.0, 10.0, 10.0, 10.0, 8.50, 10.0, 10.0, 10.0]
        }
        df = pd.DataFrame(data)
        st.info(f"📊 Usando {len(df)} registros de ejemplo")
    
    if df is None:
        data = {
            'temperatura': [20.6, 29.3, 25.9, 24.0, 17.3, 17.3, 15.9, 28.0, 24.0, 25.6],
            'altitud': [1060.5, 999.4, 764.2, 1410.9, 1214.9, 1798.6, 548.2, 1465.8, 1644.4, 1639.2],
            'puntaje': [10.0, 9.61, 10.0, 10.0, 10.0, 10.0, 8.50, 10.0, 10.0, 10.0]
        }
        df = pd.DataFrame(data)
    
    # ========== ADAPTACIÓN DE COLUMNAS PARA TU CSV ==========
    # Renombrar columnas para que coincidan con el código existente
    if 'temperatura' in df.columns and 'altitud' in df.columns and 'puntaje' in df.columns:
        df = df.rename(columns={
            'temperatura': 'temp_promedio_c',
            'altitud': 'altitud_msnm',
            'puntaje': 'puntaje_calidad_1_10'
        })
        st.success("✅ Columnas adaptadas correctamente")
    elif 'temp_promedio_c' not in df.columns:
        # Si las columnas tienen otros nombres, intentar detectar
        posibles_temp = [col for col in df.columns if 'temp' in col.lower() or 'temperatura' in col.lower()]
        posibles_alt = [col for col in df.columns if 'alt' in col.lower() or 'altitud' in col.lower()]
        posibles_punt = [col for col in df.columns if 'punt' in col.lower() or 'calidad' in col.lower() or 'score' in col.lower()]
        
        if posibles_temp:
            df = df.rename(columns={posibles_temp[0]: 'temp_promedio_c'})
        if posibles_alt:
            df = df.rename(columns={posibles_alt[0]: 'altitud_msnm'})
        if posibles_punt:
            df = df.rename(columns={posibles_punt[0]: 'puntaje_calidad_1_10'})
    
    # Usar datos editados si existen
    if st.session_state.modo == "edicion" and st.session_state.df_editado is not None:
        df_actual = st.session_state.df_editado
        if len(df_actual) > 10000:
            st.warning("⚠️ El modo edición con más de 10,000 filas puede ser lento.")
    else:
        df_actual = df
    
    # Modelo
    X = df_actual[['altitud_msnm', 'temp_promedio_c']]
    y = df_actual['puntaje_calidad_1_10']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    # ========== DATOS DE PAÍSES PARA PREDICCIONES ==========
    paises_referencia = {
        'Brasil': {'altitud_base': 1200, 'temperatura_base': 22, 'factor_calidad': 0.85, 'variedades': ['Bourbon', 'Typica', 'Caturra', 'Mundo Novo', 'Catuaí']},
        'Vietnam': {'altitud_base': 1000, 'temperatura_base': 24, 'factor_calidad': 0.75, 'variedades': ['Robusta', 'Catimor', 'Arabica']},
        'Colombia': {'altitud_base': 1600, 'temperatura_base': 20, 'factor_calidad': 0.95, 'variedades': ['Caturra', 'Colombia', 'Castillo', 'Tabi', 'Geisha']},
        'Indonesia': {'altitud_base': 1400, 'temperatura_base': 23, 'factor_calidad': 0.80, 'variedades': ['Java', 'Kintamani', 'Toraja', 'Mandheling', 'Robusta']},
        'Etiopía': {'altitud_base': 1800, 'temperatura_base': 18, 'factor_calidad': 0.98, 'variedades': ['Heirloom', 'Sidamo', 'Yirgacheffe', 'Harar', 'Limu']},
        'Honduras': {'altitud_base': 1300, 'temperatura_base': 21, 'factor_calidad': 0.88, 'variedades': ['Bourbon', 'Catuai', 'IHCAFE 90', 'Lempira']},
        'India': {'altitud_base': 1100, 'temperatura_base': 23, 'factor_calidad': 0.78, 'variedades': ['Kent', 'S795', 'Selection 9', 'Robusta']},
        'Uganda': {'altitud_base': 1400, 'temperatura_base': 22, 'factor_calidad': 0.76, 'variedades': ['Bugisu', 'Drugar', 'Robusta']},
        'México': {'altitud_base': 1300, 'temperatura_base': 21, 'factor_calidad': 0.86, 'variedades': ['Bourbon', 'Typica', 'Caturra', 'Garnica', 'Costa Rica 95']},
        'Guatemala': {'altitud_base': 1500, 'temperatura_base': 20, 'factor_calidad': 0.92, 'variedades': ['Bourbon', 'Catuai', 'Caturra', 'Pache', 'Maragogype', 'Geisha']},
        'Perú': {'altitud_base': 1550, 'temperatura_base': 19, 'factor_calidad': 0.90, 'variedades': ['Typica', 'Caturra', 'Bourbon', 'Pache', 'Catimor', 'Geisha']},
        'Nicaragua': {'altitud_base': 1300, 'temperatura_base': 21, 'factor_calidad': 0.87, 'variedades': ['Bourbon', 'Catuai', 'Caturra', 'Maragogype', 'Java']},
        'Costa Rica': {'altitud_base': 1500, 'temperatura_base': 20, 'factor_calidad': 0.94, 'variedades': ['Caturra', 'Catuai', 'Villa Sarchi', 'Geisha', 'Venecia']},
        'El Salvador': {'altitud_base': 1400, 'temperatura_base': 21, 'factor_calidad': 0.89, 'variedades': ['Bourbon', 'Pacas', 'Pacamara', 'Sarchimor']},
        'Kenia': {'altitud_base': 1700, 'temperatura_base': 19, 'factor_calidad': 0.96, 'variedades': ['SL28', 'SL34', 'Ruiru 11', 'Batian', 'K7']},
        'Tanzania': {'altitud_base': 1500, 'temperatura_base': 20, 'factor_calidad': 0.88, 'variedades': ['Bourbon', 'Kent', 'Typica', 'N39']},
        'Ecuador': {'altitud_base': 1300, 'temperatura_base': 22, 'factor_calidad': 0.85, 'variedades': ['Typica', 'Bourbon', 'Caturra', 'Catimor']},
        'Jamaica': {'altitud_base': 1200, 'temperatura_base': 22, 'factor_calidad': 0.93, 'variedades': ['Typica', 'Bourbon', 'Geisha', 'Blue Mountain']}
    }
    
    # ========== DATOS DE PAÍSES PRODUCTORES PARA TAB5 ==========
    paises_cafe = pd.DataFrame({
        'País': ['Brasil', 'Vietnam', 'Colombia', 'Indonesia', 'Etiopía', 'Honduras', 'India', 'Uganda', 'México', 'Guatemala',
                 'Perú', 'Nicaragua', 'Costa Rica', 'El Salvador', 'Kenia', 'Tanzania', 'Ecuador', 'Camerún', 'Costa de Marfil', 'Jamaica'],
        'Continente': ['América del Sur', 'Asia', 'América del Sur', 'Asia', 'África', 'América Central', 'Asia', 'África', 
                       'América del Norte', 'América Central', 'América del Sur', 'América Central', 'América Central', 
                       'América Central', 'África', 'África', 'América del Sur', 'África', 'África', 'América Central'],
        'Producción (miles de sacos)': [58000, 32000, 14000, 12000, 8000, 6500, 5800, 5000, 4200, 3800,
                                        3600, 2800, 1800, 1200, 1500, 1400, 1100, 900, 800, 500],
        'Altitud óptima (msnm)': ['800-2000', '800-1500', '1200-2000', '1000-1800', '1500-2200', '1000-1600', '800-1500', '1200-1900', 
                                  '800-1500', '1200-1800', '1200-1900', '1000-1600', '1200-1800', '1000-1600', '1400-2200', '1200-2000',
                                  '800-1800', '1000-1600', '800-1500', '800-1500'],
        'Tipos de Café': [
            'Arábica (Bourbon, Typica, Caturra, Mundo Novo), Robusta',
            'Robusta (principalmente), Arábica (Catimor)',
            'Arábica (Caturra, Colombia, Castillo, Tabi, Geisha)',
            'Robusta (90%), Arábica (Java, Kintamani, Toraja, Mandheling)',
            'Arábica (Heirloom, Sidamo, Yirgacheffe, Harar, Limu)',
            'Arábica (Bourbon, Catuai, IHCAFE 90, Lempira)',
            'Robusta (70%), Arábica (Kent, S795, Selection 9)',
            'Robusta (80%), Arábica (Bugisu, Drugar)',
            'Arábica (Bourbon, Typica, Caturra, Garnica, Costa Rica 95)',
            'Arábica (Bourbon, Catuai, Caturra, Pache, Maragogype, Geisha)',
            'Arábica (Typica, Caturra, Bourbon, Pache, Catimor, Geisha)',
            'Arábica (Bourbon, Catuai, Caturra, Maragogype, Java)',
            'Arábica (Caturra, Catuai, Villa Sarchi, Geisha, Venecia)',
            'Arábica (Bourbon, Pacas, Pacamara, Sarchimor)',
            'Arábica (SL28, SL34, Ruiru 11, Batian, K7)',
            'Arábica (Bourbon, Kent, Typica, N39), Robusta',
            'Arábica (Typica, Bourbon, Caturra, Catimor), Robusta',
            'Robusta (principalmente), Arábica (Java)',
            'Robusta (95%), Arábica (5%)',
            'Arábica (Typica, Bourbon, Geisha, Blue Mountain)'
        ],
        'Perfil de sabor': [
            'Dulce, chocolate, frutos secos, caramelo',
            'Terroso, fuerte, amaderado, cuerpo alto',
            'Suave, afrutado, caramelo, notas cítricas',
            'Terroso, herbal, especiado, chocolate oscuro',
            'Floral, cítrico, vinoso, frutos rojos, té',
            'Dulce, chocolate, frutal, notas de caramelo',
            'Especiado, terroso, chocolate, cardamomo',
            'Terroso, chocolate, cuerpo pesado',
            'Chocolate, nuez, caramelo, cítricos suaves',
            'Chocolate, frutal, floral, notas de especias',
            'Ácido, floral, chocolate, cítrico brillante',
            'Frutal, chocolate, dulce, notas de nuez',
            'Cítrico, floral, dulce, cuerpo ligero',
            'Dulce, chocolate, notas de frutas tropicales',
            'Vinoso, frutal, complejo, grosella negra',
            'Ácido, frutal, notas de ciruela, chocolate',
            'Ácido, floral, chocolate, notas cítricas',
            'Terroso, fuerte, amaderado, nuez',
            'Terroso, amaderado, chocolate',
            'Suave, floral, cítrico, chocolate, nuez'
        ],
        'Tostado recomendado': [
            'Medio a Medio-Oscuro', 'Oscuro', 'Claro a Medio', 'Medio-Oscuro', 'Claro', 'Medio', 
            'Medio a Oscuro', 'Oscuro', 'Medio', 'Claro a Medio', 'Claro', 'Medio', 'Claro', 
            'Medio', 'Claro', 'Medio', 'Claro a Medio', 'Oscuro', 'Oscuro', 'Claro'
        ]
    })
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Datos", "🤖 Modelo", "📈 Visualizaciones", "🔮 Predicciones", "🌍 Países Productores"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("📋 Dataset de Entrenamiento")
            
            if st.session_state.modo == "edicion" and len(df_actual) <= 10000:
                st.info("✏️ Modo Edición activado - Puedes modificar los valores")
                edited_df = st.data_editor(
                    df_actual,
                    use_container_width=True,
                    height=300,
                    num_rows="dynamic" if len(df_actual) < 5000 else "fixed",
                    column_config={
                        "altitud_msnm": st.column_config.NumberColumn("Altitud (msnm)", min_value=0, max_value=4000),
                        "temp_promedio_c": st.column_config.NumberColumn("Temperatura (°C)", min_value=0, max_value=40),
                        "puntaje_calidad_1_10": st.column_config.NumberColumn("Puntaje Calidad", min_value=0, max_value=10, step=0.1)
                    }
                )
                
                col_save1, col_save2 = st.columns(2)
                with col_save1:
                    if st.button("💾 Guardar cambios", use_container_width=True):
                        st.session_state.df_editado = edited_df
                        st.success("✅ Cambios guardados")
                        st.rerun()
                with col_save2:
                    if st.button("🔄 Restaurar original", use_container_width=True):
                        st.session_state.df_editado = None
                        st.rerun()
                
                csv = df_actual.to_csv(index=False)
                st.download_button(label="📥 Descargar CSV", data=csv, file_name="datos_cafe.csv", mime="text/csv")
            else:
                if st.session_state.modo == "edicion" and len(df_actual) > 10000:
                    st.warning("⚠️ Modo edición desactivado para datasets grandes")
                st.dataframe(df_actual, use_container_width=True, height=300)
        
        with col2:
            st.subheader("📊 Estadísticas Descriptivas")
            st.dataframe(df_actual.describe().round(2), use_container_width=True)
        
        st.subheader("📈 Matriz de Correlación")
        fig_corr = px.imshow(df_actual.corr(), text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab2:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📈 R² Entrenamiento", f"{r2_train:.3f}")
        with col2:
            st.metric("📉 R² Prueba", f"{r2_test:.3f}")
        with col3:
            st.metric("📊 MSE Entrenamiento", f"{mse_train:.3f}")
        with col4:
            st.metric("📊 MSE Prueba", f"{mse_test:.3f}")
        
        st.subheader("🧠 Ecuación del Modelo")
        ecuacion = f"""
        **Calidad = {model.intercept_:.2f}** + **({model.coef_[0]:.2f} × Altitud)** + **({model.coef_[1]:.2f} × Temperatura)**
        """
        st.info(ecuacion)
        
        st.subheader("⚖️ Importancia de Variables")
        importancia = pd.DataFrame({
            'Variable': ['Altitud', 'Temperatura'],
            'Coeficiente': model.coef_,
            'Importancia Absoluta': np.abs(model.coef_)
        }).sort_values('Importancia Absoluta', ascending=True)
        
        fig_imp = px.bar(importancia, x='Importancia Absoluta', y='Variable', orientation='h', 
                         color='Variable', title="Impacto en la Calidad del Café")
        st.plotly_chart(fig_imp, use_container_width=True)
    
    with tab3:
        # Obtener muestra para gráficas si el dataset es grande
        if len(df_actual) > 5000:
            df_grafica = df_actual.sample(5000, random_state=42)
            st.info(f"📊 Mostrando muestra de 5000 puntos para gráficas (total: {len(df_actual)} filas)")
        else:
            df_grafica = df_actual
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Relación 3D")
            
            tipo_grafica = st.session_state.tipo_grafica
            color_scheme = st.session_state.color_scheme
            tamaño_puntos = st.session_state.tamaño_puntos
            
            if tipo_grafica == "scatter_3d":
                fig_3d = px.scatter_3d(
                    df_grafica, 
                    x='altitud_msnm', 
                    y='temp_promedio_c', 
                    z='puntaje_calidad_1_10',
                    color='puntaje_calidad_1_10',
                    color_continuous_scale=color_scheme.lower(),
                    title="Relación Altitud vs Temperatura vs Calidad"
                )
                fig_3d.update_traces(marker=dict(size=tamaño_puntos))
            elif tipo_grafica == "surface":
                from scipy.interpolate import griddata
                altitud_vals = np.linspace(df_grafica['altitud_msnm'].min(), df_grafica['altitud_msnm'].max(), 50)
                temp_vals = np.linspace(df_grafica['temp_promedio_c'].min(), df_grafica['temp_promedio_c'].max(), 50)
                altitud_grid, temp_grid = np.meshgrid(altitud_vals, temp_vals)
                puntos = df_grafica[['altitud_msnm', 'temp_promedio_c']].values
                valores = df_grafica['puntaje_calidad_1_10'].values
                z_grid = griddata(puntos, valores, (altitud_grid, temp_grid), method='cubic')
                fig_3d = go.Figure(data=[go.Surface(z=z_grid, x=altitud_vals, y=temp_vals, colorscale=color_scheme.lower())])
                fig_3d.update_layout(title="Superficie de Calidad")
            else:
                fig_3d = px.line_3d(
                    df_grafica.sort_values('altitud_msnm'),
                    x='altitud_msnm', 
                    y='temp_promedio_c', 
                    z='puntaje_calidad_1_10',
                    title="Tendencia 3D"
                )
            
            st.plotly_chart(fig_3d, use_container_width=True)
        
        with col2:
            st.subheader("📊 Predicciones vs Reales")
            comparacion = pd.DataFrame({'Real': y_test, 'Predicho': y_pred_test, 'Error': y_test - y_pred_test})
            fig_comp = px.scatter(comparacion, x='Real', y='Predicho', title="Predicciones vs Valores Reales")
            min_val = min(y_test.min(), y_pred_test.min())
            max_val = max(y_test.max(), y_pred_test.max())
            fig_comp.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                         mode='lines', name='Predicción Perfecta', line=dict(dash='dash', color='red')))
            st.plotly_chart(fig_comp, use_container_width=True)
        
        st.subheader("📉 Análisis de Residuos")
        residuos = y_test - y_pred_test
        
        if st.session_state.tipo_residuos == "Puntos":
            fig_res = px.scatter(x=y_pred_test, y=residuos, title="Residuos vs Predicciones",
                                labels={'x': 'Valores Predichos', 'y': 'Residuos'},
                                trendline="ols" if st.session_state.mostrar_linea else None)
        elif st.session_state.tipo_residuos == "Barras":
            fig_res = px.bar(x=range(len(residuos)), y=residuos, title="Residuos por observación",
                            labels={'x': 'Observación', 'y': 'Residuo'})
        else:
            fig_res = px.line(x=range(len(residuos)), y=residuos, title="Tendencia de Residuos",
                             labels={'x': 'Observación', 'y': 'Residuo'})
        
        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_res, use_container_width=True)
    
    with tab4:
        st.subheader("🔮 Predecir Calidad de Café según País de Origen")
        
        # Selector de país
        col_pais, col_info = st.columns([1, 1])
        
        with col_pais:
            st.markdown("##### 🌍 Selecciona el país de origen")
            pais_origen = st.selectbox(
                "País productor:",
                list(paises_referencia.keys()),
                key="pais_prediccion",
                help="Selecciona el país para ver sus características de cultivo"
            )
            
            # Obtener datos del país seleccionado
            datos_pais = paises_referencia[pais_origen]
            
            st.markdown("##### 📊 Características del país")
            st.markdown(f"""
            <div style='background: #1e1e2e; padding: 15px; border-radius: 10px; margin-top: 10px;'>
                <small>🏔️ <strong>Altitud típica:</strong> {datos_pais['altitud_base']} msnm</small><br>
                <small>🌡️ <strong>Temperatura media:</strong> {datos_pais['temperatura_base']}°C</small><br>
                <small>⭐ <strong>Factor de calidad:</strong> {datos_pais['factor_calidad'] * 100:.0f}%</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col_info:
            st.markdown("##### 🌱 Variedades cultivadas")
            variedades_html = '<div style="background: #1e1e2e; padding: 15px; border-radius: 10px;"><div style="display: flex; flex-wrap: wrap; gap: 8px;">'
            for var in datos_pais['variedades']:
                variedades_html += f'<span style="background: #6F4E37; color: white; padding: 5px 12px; border-radius: 20px; font-size: 12px;">🌱 {var}</span>'
            variedades_html += '</div></div>'
            st.markdown(variedades_html, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Entradas de usuario con valores sugeridos según el país
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📍 Parámetros del Cultivo")
            
            altitud_sugerida = datos_pais['altitud_base']
            
            altitud_input = st.number_input(
                "Altitud (msnm)",
                min_value=0.0,
                max_value=3000.0,
                value=float(altitud_sugerida),
                step=50.0,
                help="Altitud sobre el nivel del mar",
                key="altitud_pred"
            )
            
            # Mostrar si está dentro del rango óptimo del país
            altitud_min = altitud_sugerida - 400
            altitud_max = altitud_sugerida + 400
            if altitud_min <= altitud_input <= altitud_max:
                st.caption(f"✅ Dentro del rango óptimo para {pais_origen} ({altitud_min}-{altitud_max} msnm)")
            else:
                st.warning(f"⚠️ Fuera del rango óptimo para {pais_origen} (recomendado: {altitud_min}-{altitud_max} msnm)")
            
            # Selector de variedad específica del país
            variedad_input = st.selectbox(
                "🌱 Variedad de Café",
                datos_pais['variedades'],
                key="variedad_pred",
                help=f"Variedades típicas de {pais_origen}"
            )
        
        with col2:
            st.markdown("##### 🌡️ Condiciones Climáticas")
            
            temp_sugerida = datos_pais['temperatura_base']
            
            temperatura_input = st.number_input(
                "Temperatura promedio (°C)",
                min_value=10.0,
                max_value=35.0,
                value=float(temp_sugerida),
                step=0.5,
                help="Temperatura promedio de la zona",
                key="temp_pred"
            )
            
            # Mostrar si está dentro del rango óptimo
            temp_min = temp_sugerida - 3
            temp_max = temp_sugerida + 3
            if temp_min <= temperatura_input <= temp_max:
                st.caption(f"✅ Temperatura óptima para {pais_origen}")
            else:
                st.warning(f"⚠️ Temperatura fuera del rango óptimo (recomendado: {temp_min}-{temp_max}°C)")
            
            humedad_input = st.slider(
                "💧 Humedad relativa (%)", 
                60, 90, 75, 
                help="Humedad ambiental",
                key="humedad_pred"
            )
        
        st.markdown("---")
        
        if st.button("🎯 Predecir Calidad", type="primary", use_container_width=True):
            # Crear DataFrame con los nuevos datos
            nuevo_cafe = pd.DataFrame({
                'altitud_msnm': [altitud_input],
                'temp_promedio_c': [temperatura_input]
            })
            
            # Hacer predicción
            puntaje_predicho_raw = model.predict(nuevo_cafe)[0]
            
            # Ajustar puntaje según el factor de calidad del país
            factor_pais = datos_pais['factor_calidad']
            puntaje_ajustado_por_pais = puntaje_predicho_raw * factor_pais
            
            # Limitar entre 0 y 10
            puntaje_predicho = max(0.0, min(10.0, puntaje_ajustado_por_pais))
            puntaje_sin_ajuste = max(0.0, min(10.0, puntaje_predicho_raw))
            
            st.markdown("---")
            st.subheader("📋 Resultado de la Evaluación")
            
            # Mostrar información del país
            st.info(f"🌍 **País de origen:** {pais_origen} | 🌱 **Variedad:** {variedad_input}")
            
            # Métricas en columnas
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("📍 Altitud", f"{altitud_input:.0f} msnm")
            with col_b:
                st.metric("🌡️ Temperatura", f"{temperatura_input:.1f}°C")
            with col_c:
                st.metric("🌍 Factor País", f"{factor_pais * 100:.0f}%")
            with col_d:
                st.metric("🎯 Calidad Final", f"{puntaje_predicho:.2f}/10")
            
            # Barra de progreso
            st.progress(puntaje_predicho / 10.0)
            
            # Mostrar comparación con y sin ajuste por país
            if abs(puntaje_sin_ajuste - puntaje_predicho) > 0.1:
                st.caption(f"📊 Predicción base: {puntaje_sin_ajuste:.2f} | Ajustada por país: {puntaje_predicho:.2f}")
            
            # Información adicional sobre el país
            with st.expander(f"ℹ️ Información sobre el café de {pais_origen}"):
                st.markdown(f"""
                **🌱 {pais_origen} - Datos de cultivo típicos:**
                - Altitud óptima: {altitud_sugerida - 400} - {altitud_sugerida + 400} msnm
                - Temperatura óptima: {temp_sugerida - 3} - {temp_sugerida + 3}°C
                - Variedades principales: {', '.join(datos_pais['variedades'])}
                
                **💡 Recomendación para {variedad_input}:**
                - Tostado recomendado: {'Claro a Medio' if factor_pais > 0.9 else 'Medio a Oscuro'}
                - Perfil esperado: {'Afruitado y floral' if factor_pais > 0.9 else 'Cuerpo medio y chocolate'}
                """)
            
            # Clasificación
            st.subheader("🏷️ Clasificación")
            
            if puntaje_predicho >= 9:
                st.success("### 🌟 EXCELENCIA - Café de Especialidad Premium")
                st.markdown(f"""
                - **Origen:** {pais_origen} - {variedad_input}
                - **Perfil**: Acidez brillante, aroma floral, notas frutales
                - **Precio estimado**: $50-80/kg
                - **Recomendación**: Exportación a mercados especializados
                """)
            elif puntaje_predicho >= 8:
                st.success("### 👍 MUY BUENO - Café de Especialidad")
                st.markdown(f"""
                - **Origen:** {pais_origen} - {variedad_input}
                - **Perfil**: Balanceado, notas frutales, cuerpo medio
                - **Precio estimado**: $30-50/kg
                - **Recomendación**: Cafeterías de especialidad
                """)
            elif puntaje_predicho >= 7:
                st.info("### ✅ BUENO - Café Comercial de Alta Calidad")
                st.markdown(f"""
                - **Origen:** {pais_origen}
                - **Perfil**: Cuerpo medio, sabor limpio, acidez moderada
                - **Precio estimado**: $15-30/kg
                - **Recomendación**: Mercado comercial premium
                """)
            elif puntaje_predicho >= 6:
                st.warning("### ⚠️ REGULAR - Café Comercial Estándar")
                st.markdown(f"""
                - **Perfil**: Sabor simple, acidez baja
                - **Precio estimado**: $8-15/kg
                - **Recomendación**: Mercado local
                """)
            else:
                st.error("### 📉 BAJO - Café de Calidad Inferior")
                st.markdown("""
                - **Perfil**: Defectos en taza, amargor
                - **Precio estimado**: < $8/kg
                - **Recomendación**: Mejorar prácticas de cultivo
                """)
            
            # Recomendaciones personalizadas según el país
            st.subheader("💡 Recomendaciones Personalizadas")
            
            col_rec1, col_rec2 = st.columns(2)
            
            with col_rec1:
                if altitud_input < altitud_sugerida - 400:
                    st.info(f"🌱 **Altitud baja para {pais_origen}:** Considerar variedades más resistentes o cultivo en zonas más altas")
                elif altitud_input > altitud_sugerida + 400:
                    st.info(f"⛰️ **Altitud alta para {pais_origen}:** Proteger cultivos de temperaturas extremas")
                else:
                    st.success(f"✅ Altitud óptima para café de {pais_origen}")
                
                if temperatura_input > temp_sugerida + 3:
                    st.info("☀️ **Temperatura alta:** Implementar sombra con árboles de leguminosas")
                elif temperatura_input < temp_sugerida - 3:
                    st.info("❄️ **Temperatura baja:** Proteger los cultivos de heladas")
                else:
                    st.success(f"✅ Temperatura óptima para {pais_origen}")
            
            with col_rec2:
                if puntaje_predicho < 7:
                    st.info("📊 **Revisar prácticas:** Cosecha selectiva y fermentación controlada")
                
                # Recomendación según el país
                if pais_origen == 'Colombia':
                    st.info("🇨🇴 **Recomendación Colombia:** Cosecha manual selectiva para preservar calidad")
                elif pais_origen == 'Etiopía':
                    st.info("🇪🇹 **Recomendación Etiopía:** Procesamiento natural para realzar notas frutales")
                elif pais_origen == 'Brasil':
                    st.info("🇧🇷 **Recomendación Brasil:** Control estricto de fermentación")
                elif pais_origen == 'Costa Rica':
                    st.info("🇨🇷 **Recomendación Costa Rica:** Beneficio húmedo de alta calidad")
                else:
                    st.info(f"🌍 **Recomendación {pais_origen}:** Mantener trazabilidad completa del lote")
    
    with tab5:
        st.subheader("🌍 Países Productores de Café")
        st.markdown("---")
        
        # Mostrar tabla resumen
        st.dataframe(paises_cafe[['País', 'Continente', 'Producción (miles de sacos)', 'Tipos de Café']], 
                     use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Selector de país con imagen
        col_sel1, col_sel2 = st.columns([1, 2])
        
        with col_sel1:
            st.subheader("🇺🇳 Selecciona un país")
            pais_seleccionado = st.selectbox("", paises_cafe['País'].tolist(), label_visibility="collapsed", key="pais_tab5")
        
        if pais_seleccionado:
            pais_data = paises_cafe[paises_cafe['País'] == pais_seleccionado].iloc[0]
            
            # Mostrar información detallada del país seleccionado
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                        padding: 25px; border-radius: 15px; margin: 10px 0; border: 1px solid #6F4E37;'>
                <h2 style='color: #6F4E37; text-align: center;'>☕ {pais_seleccionado}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Métricas principales
            col_met1, col_met2, col_met3, col_met4 = st.columns(4)
            with col_met1:
                st.metric("🌍 Continente", pais_data['Continente'])
            with col_met2:
                st.metric("📦 Producción", f"{pais_data['Producción (miles de sacos)']:,.0f} miles de sacos")
            with col_met3:
                st.metric("⛰️ Altitud óptima", pais_data['Altitud óptima (msnm)'])
            with col_met4:
                st.metric("🔥 Tostado recomendado", pais_data['Tostado recomendado'])
            
            # Tipos de Café (sección destacada)
            st.markdown("---")
            st.markdown("## 🌱 **Tipos de Café que cultiva**")
            
            # Mostrar tipos de café en formato de tarjetas
            tipos_cafe = pais_data['Tipos de Café'].split(',')
            
            cols_tipos = st.columns(min(len(tipos_cafe), 4))
            for i, tipo in enumerate(tipos_cafe):
                with cols_tipos[i % 4]:
                    st.markdown(f"""
                    <div style='background: #2d2d44; padding: 12px; border-radius: 10px; text-align: center; margin: 5px; border-left: 4px solid #6F4E37;'>
                        <strong style='color: #D2691E;'>🌱 {tipo.strip()}</strong>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Perfil de sabor y características
            col_sabor1, col_sabor2 = st.columns(2)
            
            with col_sabor1:
                st.markdown(f"""
                <div style='background: #1e1e2e; padding: 20px; border-radius: 12px; height: 100%;'>
                    <h3 style='color: #6F4E37;'>👃 Perfil de Sabor</h3>
                    <p style='font-size: 16px;'>{pais_data['Perfil de sabor']}</p>
                    <hr>
                    <h4>🎯 Notas predominantes:</h4>
                    <ul>
                """, unsafe_allow_html=True)
                
                # Extraer notas de sabor
                notas = [n.strip() for n in pais_data['Perfil de sabor'].split(',')]
                for nota in notas[:5]:
                    st.markdown(f"<li>✨ {nota}</li>", unsafe_allow_html=True)
                
                st.markdown("</ul></div>", unsafe_allow_html=True)
            
            with col_sabor2:
                st.markdown(f"""
                <div style='background: #1e1e2e; padding: 20px; border-radius: 12px; height: 100%;'>
                    <h3 style='color: #6F4E37;'>🍵 Características del Café</h3>
                    <p><strong>🏆 Variedades principales:</strong><br>{pais_data['Tipos de Café']}</p>
                    <hr>
                    <p><strong>🔥 Tostado ideal:</strong> {pais_data['Tostado recomendado']}</p>
                    <p><strong>⛰️ Altitud de cultivo:</strong> {pais_data['Altitud óptima (msnm)']} msnm</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Gráfico de producción por país
        st.markdown("---")
        st.subheader("📊 Producción de Café por País")
        
        top_n = st.slider("Mostrar top N países:", 5, 20, 10, key="top_paises")
        paises_top = paises_cafe.nlargest(top_n, 'Producción (miles de sacos)')
        
        fig_produccion = px.bar(paises_top, 
                                x='País', 
                                y='Producción (miles de sacos)',
                                color='Continente',
                                title=f"Top {top_n} Países Productores de Café",
                                text='Producción (miles de sacos)',
                                color_discrete_sequence=px.colors.qualitative.Set2)
        fig_produccion.update_traces(textposition='outside')
        fig_produccion.update_layout(height=500)
        st.plotly_chart(fig_produccion, use_container_width=True)
        
        # Gráfico por continente
        col_graf1, col_graf2 = st.columns(2)
        
        with col_graf1:
            st.subheader("🌎 Producción por Continente")
            produccion_continente = paises_cafe.groupby('Continente')['Producción (miles de sacos)'].sum().reset_index()
            fig_continente = px.pie(produccion_continente, 
                                    values='Producción (miles de sacos)', 
                                    names='Continente',
                                    title="Distribución por Continente",
                                    hole=0.3,
                                    color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig_continente, use_container_width=True)
        
        with col_graf2:
            st.subheader("🏆 Principales Productores")
            top_5 = paises_cafe.nlargest(5, 'Producción (miles de sacos)')
            fig_top5 = px.bar(top_5, x='País', y='Producción (miles de sacos)', 
                              color='País', title="Top 5 Productores Mundiales",
                              text='Producción (miles de sacos)')
            fig_top5.update_traces(textposition='outside')
            fig_top5.update_layout(showlegend=False)
            st.plotly_chart(fig_top5, use_container_width=True)
    
    # Mostrar historial
    if st.session_state.get('show_history', False):
        with st.expander("📜 Mi Historial de Predicciones", expanded=True):
            st.info("📊 Aquí se mostrarán tus predicciones guardadas")
            if st.button("Cerrar historial"):
                st.session_state.show_history = False
                st.rerun()
    
    # Mostrar recomendaciones
    if st.session_state.get('show_recommendations', False):
        with st.expander("💡 Recomendaciones para Mejorar la Calidad", expanded=True):
            st.markdown("""
            ### 📋 Guía de Buenas Prácticas para Caficultores
            
            #### 🌱 **Cultivo**
            - ✅ Mantener altitud entre 1200-2000 msnm
            - ✅ Temperatura óptima: 18-24°C
            - ✅ Implementar sombra regulada (20-40%)
            
            #### 🍒 **Cosecha**
            - ✅ Cosecha selectiva (solo cerezas rojas)
            - ✅ Beneficio dentro de 6-8 horas post-cosecha
            
            #### 🏭 **Beneficio**
            - ✅ Fermentación controlada (12-24 horas)
            - ✅ Secado uniforme (humedad 10-12%)
            """)
            if st.button("Cerrar recomendaciones"):
                st.session_state.show_recommendations = False
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>☕ Desarrollado con ❤️ para caficultores</p>
        <p>📧 Usuario: {st.session_state.get('user_email', '')}</p>
    </div>
    """, unsafe_allow_html=True)

# ========== PUNTO DE ENTRADA ==========
if not st.session_state.logged_in:
    show_login_ui()
else:
    main_app()
