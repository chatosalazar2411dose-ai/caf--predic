import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
import pyrebase
import json
import requests
from datetime import datetime

# Configuración de la página DEBE ser el primer comando Streamlit
st.set_page_config(
    page_title="☕ Predictor de Calidad de Café",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CONFIGURACIÓN DE FIREBASE ==========
# Cargar configuración desde secrets.toml
firebase_config = {
    "apiKey": st.secrets["FIREBASE_API_KEY"],
    "authDomain": st.secrets["FIREBASE_AUTH_DOMAIN"],
    "databaseURL": st.secrets["FIREBASE_DATABASE_URL"],
    "projectId": st.secrets["FIREBASE_PROJECT_ID"],
    "storageBucket": st.secrets["FIREBASE_STORAGE_BUCKET"],
    "messagingSenderId": st.secrets["FIREBASE_MESSAGING_SENDER_ID"],
    "appId": st.secrets["FIREBASE_APP_ID"],
    "measurementId": st.secrets["FIREBASE_MEASUREMENT_ID"]
}

# Inicializar Firebase
firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()
db = firebase.database()

# ========== FUNCIONES DE AUTENTICACIÓN ==========
def init_session_state():
    """Inicializa las variables de sesión"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'user_email' not in st.session_state:
        st.session_state.user_email = None
    if 'user_name' not in st.session_state:
        st.session_state.user_name = None

def login_user(email, password):
    """Autentica al usuario con Firebase"""
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        st.session_state.logged_in = True
        st.session_state.user_id = user['localId']
        st.session_state.user_email = email
        
        # Obtener nombre de usuario desde la base de datos
        try:
            user_data = db.child("users").child(user['localId']).child("profile").get().val()
            if user_data and 'name' in user_data:
                st.session_state.user_name = user_data['name']
            else:
                st.session_state.user_name = email.split('@')[0]
        except:
            st.session_state.user_name = email.split('@')[0]
        
        return True, "Login exitoso"
    except requests.exceptions.HTTPError as e:
        error_msg = json.loads(e.args[1])["error"]["message"]
        if error_msg == "EMAIL_NOT_FOUND":
            return False, "Email no registrado"
        elif error_msg == "INVALID_PASSWORD":
            return False, "Contraseña incorrecta"
        elif error_msg == "USER_DISABLED":
            return False, "Usuario deshabilitado"
        else:
            return False, f"Error: {error_msg}"
    except Exception as e:
        return False, f"Error de conexión: {str(e)}"

def register_user(email, password, name):
    """Registra un nuevo usuario en Firebase"""
    try:
        # Crear usuario en Authentication
        user = auth.create_user_with_email_and_password(email, password)
        
        # Guardar datos adicionales en Realtime Database
        user_data = {
            "profile": {
                "name": name,
                "email": email,
                "created_at": datetime.now().isoformat(),
                "role": "user"
            },
            "predictions": []
        }
        db.child("users").child(user['localId']).set(user_data)
        
        # Iniciar sesión automáticamente
        st.session_state.logged_in = True
        st.session_state.user_id = user['localId']
        st.session_state.user_email = email
        st.session_state.user_name = name
        
        return True, "Registro exitoso"
    except requests.exceptions.HTTPError as e:
        error_msg = json.loads(e.args[1])["error"]["message"]
        if error_msg == "EMAIL_EXISTS":
            return False, "El email ya está registrado"
        elif error_msg == "WEAK_PASSWORD":
            return False, "La contraseña es muy débil (mínimo 6 caracteres)"
        else:
            return False, f"Error: {error_msg}"
    except Exception as e:
        return False, f"Error de registro: {str(e)}"

def logout_user():
    """Cierra la sesión del usuario"""
    for key in ['logged_in', 'user_id', 'user_email', 'user_name']:
        if key in st.session_state:
            st.session_state[key] = None if key != 'logged_in' else False
    st.rerun()

def save_prediction_to_firebase(prediction_data):
    """Guarda una predicción en Firebase"""
    if st.session_state.logged_in and st.session_state.user_id:
        try:
            predictions_ref = db.child("users").child(st.session_state.user_id).child("predictions")
            new_pred = predictions_ref.push(prediction_data)
            return True
        except Exception as e:
            st.error(f"Error al guardar predicción: {e}")
            return False
    return False

def get_user_predictions():
    """Obtiene el historial de predicciones del usuario"""
    if st.session_state.logged_in and st.session_state.user_id:
        try:
            predictions = db.child("users").child(st.session_state.user_id).child("predictions").get()
            if predictions.val():
                return predictions.val()
        except Exception as e:
            st.error(f"Error al obtener historial: {e}")
    return {}

# ========== INTERFAZ DE LOGIN ==========
def show_login_ui():
    """Muestra la interfaz de login/registro"""
    st.markdown("""
    <style>
  
    .auth-title {
        text-align: center;
        margin-bottom: 30px;
    }
    .auth-title h1 {
        font-size: 2.5em;
        margin-bottom: 10px;
    }
    .auth-title p {
        opacity: 0.9;
    }
    .stButton > button {
        background-color: #ff6b6b;
        color: white;
        border: none;
        padding: 10px 25px;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #ff5252;
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.container():
            st.markdown('<div class="auth-container">', unsafe_allow_html=True)
            st.markdown("""
            <div class="auth-title">
                <h1>☕ Café Quality Predictor</h1>
                <p>Sistema experto para evaluación de calidad de café</p>
            </div>
            """, unsafe_allow_html=True)
            
            tab1, tab2 = st.tabs(["🔐 Iniciar Sesión", "📝 Registrarse"])
            
            with tab1:
                with st.form("login_form"):
                    email = st.text_input("📧 Email", placeholder="tu@email.com")
                    password = st.text_input("🔒 Contraseña", type="password", placeholder="••••••••")
                    submit = st.form_submit_button("🚪 Iniciar Sesión", use_container_width=True)
                    
                    if submit:
                        if email and password:
                            success, message = login_user(email, password)
                            if success:
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)
                        else:
                            st.warning("Por favor ingresa email y contraseña")
            
            with tab2:
                with st.form("register_form"):
                    name = st.text_input("👤 Nombre completo", placeholder="Tu nombre")
                    email = st.text_input("📧 Email", placeholder="tu@email.com")
                    password = st.text_input("🔒 Contraseña", type="password", placeholder="Mínimo 6 caracteres")
                    confirm_password = st.text_input("🔒 Confirmar contraseña", type="password")
                    submit = st.form_submit_button("📝 Registrarse", use_container_width=True)
                    
                    if submit:
                        if not name:
                            st.warning("Por favor ingresa tu nombre")
                        elif not email:
                            st.warning("Por favor ingresa tu email")
                        elif not password:
                            st.warning("Por favor ingresa una contraseña")
                        elif len(password) < 6:
                            st.warning("La contraseña debe tener al menos 6 caracteres")
                        elif password != confirm_password:
                            st.warning("Las contraseñas no coinciden")
                        else:
                            success, message = register_user(email, password, name)
                            if success:
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)
            
            st.markdown("---")
            st.caption("🔒 Tus datos están seguros con Firebase Authentication")
            st.markdown('</div>', unsafe_allow_html=True)

# ========== APLICACIÓN PRINCIPAL ==========
def main_app():
    """Aplicación principal (solo accesible después del login)"""
    
    # Sidebar con información del usuario
    with st.sidebar:
        st.image("https://em-content.zobj.net/thumbs/120/apple/354/hot-beverage_2615.png", width=100)
        
        # Mostrar información del usuario
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 15px; border-radius: 10px; color: white; margin-bottom: 20px;'>
            <small>👤 <strong>{st.session_state.user_name}</strong></small><br>
            <small>📧 {st.session_state.user_email}</small><br>
            <small>🆔 {st.session_state.user_id[:8]}...</small>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚪 Cerrar Sesión", use_container_width=True):
            logout_user()
        
        st.markdown("---")
        st.header("⚙️ Panel de Control")
        
        # Dataset selector
        opcion_datos = st.radio(
            "📊 Fuente de datos:",
            ["📁 Datos de ejemplo", "📂 Cargar CSV"],
            help="Selecciona cómo quieres cargar los datos"
        )
        
        # Parámetros del modelo
        st.subheader("🔧 Parámetros del Modelo")
        test_size = st.slider("Tamaño del conjunto de prueba:", 0.1, 0.5, 0.3, 0.05)
        random_state = st.number_input("Semilla aleatoria:", 0, 100, 42)
        
        # Ver historial de predicciones
        if st.button("📜 Ver mi historial", use_container_width=True):
            st.session_state.show_history = True
    
    # Cargar datos
    if opcion_datos == "📂 Cargar CSV":
        archivo = st.file_uploader("Seleccionar archivo CSV", type=['csv'])
        if archivo:
            df = pd.read_csv(archivo)
            st.success(f"✅ {len(df)} registros cargados")
        else:
            st.warning("⚠️ Usando datos de ejemplo por defecto")
            df = None
    else:
        data = {
            'altitud_msnm': [800, 950, 1100, 1250, 1400, 1550, 1700, 1850, 2000, 2150],
            'temp_promedio_c': [28.5, 27.2, 25.8, 24.3, 22.7, 21.2, 19.8, 18.3, 16.9, 15.4],
            'puntaje_calidad_1_10': [3.2, 4.8, 6.1, 7.2, 8.1, 8.7, 9.1, 9.4, 9.7, 9.9]
        }
        df = pd.DataFrame(data)
        st.info(f"📊 Usando {len(df)} registros de ejemplo")
    
    if df is None:
        data = {
            'altitud_msnm': [800, 950, 1100, 1250, 1400, 1550, 1700, 1850, 2000, 2150],
            'temp_promedio_c': [28.5, 27.2, 25.8, 24.3, 22.7, 21.2, 19.8, 18.3, 16.9, 15.4],
            'puntaje_calidad_1_10': [3.2, 4.8, 6.1, 7.2, 8.1, 8.7, 9.1, 9.4, 9.7, 9.9]
        }
        df = pd.DataFrame(data)
    
    # Preparar y entrenar modelo
    X = df[['altitud_msnm', 'temp_promedio_c']]
    y = df['puntaje_calidad_1_10']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    # Título principal
    st.markdown("""
        <h1 style='text-align: center; color: #6F4E37;'>
            ☕ Sistema Experto para Predicción de Calidad de Café
        </h1>
        <hr style='border: 2px solid #6F4E37;'>
    """, unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Datos", "🤖 Modelo", "📈 Visualizaciones", "🔮 Predicciones"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Dataset de Entrenamiento")
            st.dataframe(df, use_container_width=True, height=300)
        with col2:
            st.subheader("Estadísticas Descriptivas")
            st.dataframe(df.describe().round(2), use_container_width=True)
        
        fig_corr = px.imshow(df.corr(), text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab2:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² (Entrenamiento)", f"{r2_train:.3f}")
        with col2:
            st.metric("R² (Prueba)", f"{r2_test:.3f}")
        with col3:
            st.metric("MSE (Entrenamiento)", f"{mse_train:.3f}")
        with col4:
            st.metric("MSE (Prueba)", f"{mse_test:.3f}")
        
        ecuacion = f"""
        **Calidad = {model.intercept_:.2f}** + **({model.coef_[0]:.2f} × Altitud)** + **({model.coef_[1]:.2f} × Temperatura)**
        """
        st.info(ecuacion)
        
        importancia = pd.DataFrame({
            'Variable': ['Altitud', 'Temperatura'],
            'Coeficiente': model.coef_,
            'Importancia Absoluta': np.abs(model.coef_)
        }).sort_values('Importancia Absoluta', ascending=True)
        
        fig_imp = px.bar(importancia, x='Importancia Absoluta', y='Variable', orientation='h', color='Variable')
        st.plotly_chart(fig_imp, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig_3d = px.scatter_3d(df, x='altitud_msnm', y='temp_promedio_c', z='puntaje_calidad_1_10',
                                   color='puntaje_calidad_1_10', title="Relación 3D")
            st.plotly_chart(fig_3d, use_container_width=True)
        with col2:
            comparacion = pd.DataFrame({'Real': y_test, 'Predicho': y_pred_test, 'Error': y_test - y_pred_test})
            fig_comp = px.scatter(comparacion, x='Real', y='Predicho', title="Predicciones vs Reales")
            min_val = min(y_test.min(), y_pred_test.min())
            max_val = max(y_test.max(), y_pred_test.max())
            fig_comp.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                         mode='lines', name='Perfecta', line=dict(dash='dash', color='red')))
            st.plotly_chart(fig_comp, use_container_width=True)
        
        residuos = y_test - y_pred_test
        fig_res = px.scatter(x=y_pred_test, y=residuos, title="Residuos vs Predicciones")
        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_res, use_container_width=True)
    
    with tab4:
        st.subheader("🔮 Predecir Calidad de Nuevo Lote")
        
        col1, col2 = st.columns(2)
        with col1:
            altitud_input = st.number_input("Altitud (msnm)", min_value=0.0, max_value=3000.0, value=1650.0, step=50.0)
            variedad = st.selectbox("Variedad de Café", ["Borbón", "Caturra", "Typica", "Catuaí", "Geisha"])
        with col2:
            temperatura_input = st.number_input("Temperatura promedio (°C)", min_value=10.0, max_value=35.0, value=20.0, step=0.5)
            humedad = st.slider("Humedad relativa (%)", 60, 90, 75)
        
        if st.button("🎯 Predecir Calidad", type="primary", use_container_width=True):
            nuevo_cafe = pd.DataFrame({'altitud_msnm': [altitud_input], 'temp_promedio_c': [temperatura_input]})
            puntaje_predicho_raw = model.predict(nuevo_cafe)[0]
            puntaje_predicho = max(0.0, min(10.0, puntaje_predicho_raw))
            
            # Guardar predicción en Firebase
            prediction_record = {
                "timestamp": datetime.now().isoformat(),
                "altitud": altitud_input,
                "temperatura": temperatura_input,
                "variedad": variedad,
                "humedad": humedad,
                "puntaje_raw": float(puntaje_predicho_raw),
                "puntaje_ajustado": float(puntaje_predicho)
            }
            save_prediction_to_firebase(prediction_record)
            
            st.markdown("---")
            st.subheader("📋 Resultado de la Evaluación")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Altitud", f"{altitud_input:.0f} msnm")
            with col2:
                st.metric("Temperatura", f"{temperatura_input:.1f}°C")
            with col3:
                st.metric("Puntaje Predicho", f"{puntaje_predicho:.2f}/10")
            
            st.progress(puntaje_predicho / 10.0)
            
            if puntaje_predicho_raw != puntaje_predicho:
                st.warning(f"⚠️ Predicción ajustada de {puntaje_predicho_raw:.2f} a {puntaje_predicho:.2f}")
            
            st.subheader("🏷️ Clasificación")
            if puntaje_predicho >= 9:
                st.success("### 🌟 EXCELENCIA - Café de Especialidad Premium")
            elif puntaje_predicho >= 8:
                st.success("### 👍 MUY BUENO - Café de Especialidad")
            elif puntaje_predicho >= 7:
                st.info("### ✅ BUENO - Café Comercial de Alta Calidad")
            elif puntaje_predicho >= 6:
                st.warning("### ⚠️ REGULAR - Café Comercial Estándar")
            else:
                st.error("### 📉 BAJO - Café de Calidad Inferior")
    
    # Mostrar historial si se solicitó
    if st.session_state.get('show_history', False):
        with st.expander("📜 Mi Historial de Predicciones", expanded=True):
            predictions = get_user_predictions()
            if predictions:
                pred_list = []
                for key, pred in predictions.items():
                    pred_list.append(pred)
                hist_df = pd.DataFrame(pred_list)
                st.dataframe(hist_df.sort_values('timestamp', ascending=False), use_container_width=True)
            else:
                st.info("Aún no tienes predicciones guardadas")
            if st.button("Cerrar historial"):
                st.session_state.show_history = False
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: gray;'>
        Desarrollado con ❤️ para caficultores | Firebase Authentication<br>
        Usuario: {st.session_state.user_name} | {st.session_state.user_email}
    </div>
    """, unsafe_allow_html=True)

# ========== PUNTO DE ENTRADA PRINCIPAL ==========
def run():
    init_session_state()
    
    if not st.session_state.logged_in:
        show_login_ui()
    else:
        main_app()

if __name__ == "__main__":
    run()