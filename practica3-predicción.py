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
import hashlib

# Configuración de la página
st.set_page_config(
    page_title="☕ Predictor de Calidad de Café",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== OPTIMIZACIÓN PARA DATASETS GRANDES ==========
@st.cache_data
def cargar_csv_con_optimizacion(archivo, max_filas=None):
    """Carga CSV con optimización para archivos grandes"""
    try:
        if max_filas:
            # Cargar solo una muestra si es muy grande
            df = pd.read_csv(archivo, nrows=max_filas)
            st.info(f"📊 Cargadas {len(df)} filas (muestra de un archivo más grande)")
        else:
            df = pd.read_csv(archivo)
            if len(df) > 50000:
                st.warning(f"⚠️ El archivo tiene {len(df)} filas. Se usará una muestra de 50000 para rendimiento óptimo.")
                df = df.sample(50000, random_state=42)
        return df
    except Exception as e:
        st.error(f"Error al cargar archivo: {e}")
        return None

@st.cache_data
def calcular_correlacion_optimizada(df, max_filas=10000):
    """Calcula correlación con muestreo para datasets grandes"""
    if len(df) > max_filas:
        df_sample = df.sample(max_filas, random_state=42)
        st.caption(f"📊 Correlación calculada con muestra de {max_filas} filas")
        return df_sample.corr()
    return df.corr()

@st.cache_data
def entrenar_modelo_optimizado(X, y):
    """Entrena el modelo con caché"""
    model = LinearRegression()
    model.fit(X, y)
    return model

def mostrar_dataframe_optimizado(df, max_filas=1000):
    """Muestra dataframe con límite de filas"""
    if len(df) > max_filas:
        st.dataframe(df.head(max_filas), use_container_width=True, height=300)
        st.caption(f"📊 Mostrando {max_filas} de {len(df)} filas totales")
    else:
        st.dataframe(df, use_container_width=True, height=300)

def obtener_muestra_para_grafica(df, max_puntos=5000):
    """Obtiene una muestra para gráficas"""
    if len(df) > max_puntos:
        return df.sample(max_puntos, random_state=42)
    return df

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
    if 'df_original' not in st.session_state:
        st.session_state.df_original = None
    if 'dataset_size' not in st.session_state:
        st.session_state.dataset_size = "pequeno"

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

# ========== INTERFAZ DE LOGIN ==========
def show_login_ui():
    """Muestra la interfaz de login/registro"""
    
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
        
        # Opción para limitar filas en datasets grandes
        if opcion_datos == "📂 Cargar CSV":
            limit_filas = st.number_input(
                "🔢 Límite de filas a cargar (0 = todas):",
                min_value=0,
                max_value=500000,
                value=50000,
                step=5000,
                help="Para archivos muy grandes, puedes cargar solo una muestra"
            )
        else:
            limit_filas = 0
        
        st.subheader("🔧 Parámetros del Modelo")
        test_size = st.slider("Tamaño del conjunto de prueba:", 0.1, 0.5, 0.3, 0.05)
        random_state = st.number_input("Semilla aleatoria:", 0, 100, 42)
        
        st.markdown("---")
        st.subheader("🎨 Modo de Visualización")
        
        modo_visualizacion = st.radio(
            "Selecciona el modo:",
            ["📊 Modo Visualización", "✏️ Modo Edición"],
            help="Visualización: solo ver datos | Edición: modificar valores"
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
            st.session_state.tamaño_puntos = st.slider("Tamaño de puntos:", 3, 15, 8)
            st.session_state.tipo_residuos = st.radio(
                "Tipo de gráfico de residuos:",
                ["Puntos", "Barras", "Línea"],
                horizontal=True
            )
            st.session_state.mostrar_linea = st.checkbox("Mostrar línea de tendencia", value=True)
        
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
    
    # Cargar datos con optimización
    if opcion_datos == "📂 Cargar CSV":
        archivo = st.file_uploader("Seleccionar archivo CSV", type=['csv'])
        if archivo:
            max_filas = limit_filas if limit_filas > 0 else None
            df = cargar_csv_con_optimizacion(archivo, max_filas)
            if df is not None:
                st.session_state.df_original = df
                st.success(f"✅ {len(df)} registros cargados exitosamente")
                if len(df) >= 50000:
                    st.info("💡 Consejo: Para mejor rendimiento, usa el modo Visualización y evita editar datasets grandes.")
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
    
    # Usar datos editados si existen
    if st.session_state.modo == "edicion" and st.session_state.df_editado is not None:
        df_actual = st.session_state.df_editado
        if len(df_actual) > 10000:
            st.warning("⚠️ El modo edición con más de 10,000 filas puede ser lento. Considera usar el modo visualización.")
    else:
        df_actual = df
    
    # Obtener muestra para gráficas (si es necesario)
    df_grafica = obtener_muestra_para_grafica(df_actual, 5000)
    if len(df_actual) > 5000:
        st.info(f"📊 Mostrando muestra de 5000 puntos para gráficas (total: {len(df_actual)} filas)")
    
    # Modelo con caché
    X = df_actual[['altitud_msnm', 'temp_promedio_c']]
    y = df_actual['puntaje_calidad_1_10']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = entrenar_modelo_optimizado(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Datos", "🤖 Modelo", "📈 Visualizaciones", "🔮 Predicciones"])
    
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
                    st.warning("⚠️ Modo edición desactivado para datasets grandes. Cambia al modo visualización.")
                mostrar_dataframe_optimizado(df_actual, 1000)
        
        with col2:
            st.subheader("📊 Estadísticas Descriptivas")
            st.dataframe(df_actual.describe().round(2), use_container_width=True)
        
        st.subheader("📈 Matriz de Correlación")
        fig_corr = px.imshow(calcular_correlacion_optimizada(df_actual), text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
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
        st.subheader("🔮 Predecir Calidad de Nuevo Lote")
        
        col1, col2 = st.columns(2)
        with col1:
            altitud = st.number_input("Altitud (msnm)", 0.0, 3000.0, 1650.0, step=50.0)
            variedad = st.selectbox("🌱 Variedad de Café", ["Borbón", "Caturra", "Typica", "Catuaí", "Geisha"])
        with col2:
            temperatura = st.number_input("Temperatura promedio (°C)", 10.0, 35.0, 20.0, step=0.5)
            humedad = st.slider("💧 Humedad relativa (%)", 60, 90, 75)
        
        if st.button("🎯 Predecir Calidad", type="primary", use_container_width=True):
            prediccion_raw = model.predict([[altitud, temperatura]])[0]
            prediccion = max(0.0, min(10.0, prediccion_raw))
            
            st.markdown("---")
            st.subheader("📋 Resultado de la Evaluación")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("📍 Altitud", f"{altitud:.0f} msnm")
            with col_b:
                st.metric("🌡️ Temperatura", f"{temperatura:.1f}°C")
            with col_c:
                st.metric("🎯 Calidad Predicha", f"{prediccion:.2f}/10")
            
            st.progress(prediccion/10)
            
            if prediccion_raw != prediccion:
                st.warning(f"⚠️ Predicción ajustada de {prediccion_raw:.2f} a {prediccion:.2f}")
            
            st.subheader("🏷️ Clasificación")
            if prediccion >= 9:
                st.success("### 🌟 EXCELENCIA - Café de Especialidad Premium")
            elif prediccion >= 8:
                st.success("### 👍 MUY BUENO - Café de Especialidad")
            elif prediccion >= 7:
                st.info("### ✅ BUENO - Café Comercial de Alta Calidad")
            elif prediccion >= 6:
                st.warning("### ⚠️ REGULAR - Café Comercial Estándar")
            else:
                st.error("### 📉 BAJO - Café de Calidad Inferior")
            
            st.subheader("💡 Recomendaciones")
            if altitud < 1200:
                st.info("🌱 Considerar variedades resistentes a bajas altitudes")
            if temperatura > 25:
                st.info("☀️ Implementar sombra para reducir temperatura")
            if prediccion < 7:
                st.info("📊 Revisar prácticas de cosecha y fermentación")
    
    if st.session_state.get('show_history', False):
        with st.expander("📜 Mi Historial de Predicciones", expanded=True):
            st.info("📊 Aquí se mostrarán tus predicciones guardadas")
            if st.button("Cerrar historial"):
                st.session_state.show_history = False
                st.rerun()
    
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
