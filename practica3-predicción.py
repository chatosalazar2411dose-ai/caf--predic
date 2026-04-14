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
    for key in ['logged_in', 'user_id', 'user_email', 'user_name', 'show_history', 'show_recommendations']:
        if key in st.session_state:
            st.session_state[key] = None if key not in ['logged_in', 'show_history', 'show_recommendations'] else False
    st.rerun()

# ========== INTERFAZ DE LOGIN ==========
def show_login_ui():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background: white; padding: 30px; border-radius: 15px; box-shadow: 0 5px 20px rgba(0,0,0,0.1);'>
        """, unsafe_allow_html=True)
        st.subheader("🔐 Iniciar Sesión")
        email = st.text_input("📧 Email", placeholder="demo@cafe.com", key="login_email")
        password = st.text_input("🔒 Contraseña", type="password", placeholder="••••••••", key="login_pass")
        
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
        
        st.caption("🔧 Credenciales de prueba: demo@cafe.com / demo123 (debes crearlo primero)")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: white; padding: 30px; border-radius: 15px; box-shadow: 0 5px 20px rgba(0,0,0,0.1);'>
        """, unsafe_allow_html=True)
        st.subheader("📝 Registrarse")
        new_name = st.text_input("👤 Nombre completo", placeholder="Tu nombre", key="reg_name")
        new_email = st.text_input("📧 Email", placeholder="tu@email.com", key="reg_email")
        new_password = st.text_input("🔒 Contraseña", type="password", placeholder="Mínimo 6 caracteres", key="reg_pass")
        confirm_password = st.text_input("🔒 Confirmar contraseña", type="password", key="reg_confirm")
        
        if st.button("📝 Registrarse", use_container_width=True):
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
                success, message = register_user(new_email, new_password, new_name)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        st.markdown("</div>", unsafe_allow_html=True)

# ========== APLICACIÓN PRINCIPAL ==========
def main_app():
    """Aplicación principal después del login"""
    
    # Sidebar con información del usuario y controles
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
    
    # Título principal con animación
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
    
    # Mensaje de bienvenida
    st.balloons()
    st.success(f"✨ ¡Bienvenido/a {st.session_state.get('user_name', 'Usuario')}! ☕")
    
    # Cargar datos
    if opcion_datos == "📂 Cargar CSV":
        archivo = st.file_uploader("Seleccionar archivo CSV", type=['csv'])
        if archivo:
            df = pd.read_csv(archivo)
            st.success(f"✅ {len(df)} registros cargados exitosamente")
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
    
    # Modelo
    X = df[['altitud_msnm', 'temp_promedio_c']]
    y = df['puntaje_calidad_1_10']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Métricas
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
            st.dataframe(df, use_container_width=True, height=300)
        with col2:
            st.subheader("📊 Estadísticas Descriptivas")
            st.dataframe(df.describe().round(2), use_container_width=True)
        
        st.subheader("📈 Matriz de Correlación")
        fig_corr = px.imshow(df.corr(), text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
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
            fig_3d = px.scatter_3d(df, x='altitud_msnm', y='temp_promedio_c', z='puntaje_calidad_1_10',
                                   color='puntaje_calidad_1_10', title="Relación Altitud vs Temperatura vs Calidad")
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
        fig_res = px.scatter(x=y_pred_test, y=residuos, title="Residuos vs Predicciones",
                            labels={'x': 'Valores Predichos', 'y': 'Residuos'})
        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_res, use_container_width=True)
    
    with tab4:
        st.subheader("🔮 Predecir Calidad de Nuevo Lote")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 📍 Ubicación del Cultivo")
            altitud = st.number_input("Altitud (msnm)", 0.0, 3000.0, 1650.0, step=50.0, 
                                      help="Altitud sobre el nivel del mar")
            variedad = st.selectbox("🌱 Variedad de Café", ["Borbón", "Caturra", "Typica", "Catuaí", "Geisha"])
        with col2:
            st.markdown("##### 🌡️ Condiciones Climáticas")
            temperatura = st.number_input("Temperatura promedio (°C)", 10.0, 35.0, 20.0, step=0.5,
                                         help="Temperatura promedio de la zona")
            humedad = st.slider("💧 Humedad relativa (%)", 60, 90, 75, help="Humedad ambiental")
        
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
                st.markdown("""
                - **Perfil**: Acidez brillante, aroma floral
                - **Precio estimado**: $50-80/kg
                - **Recomendación**: Exportación a mercados especializados
                """)
            elif prediccion >= 8:
                st.success("### 👍 MUY BUENO - Café de Especialidad")
                st.markdown("""
                - **Perfil**: Balanceado, notas frutales
                - **Precio estimado**: $30-50/kg
                - **Recomendación**: Cafeterías de especialidad
                """)
            elif prediccion >= 7:
                st.info("### ✅ BUENO - Café Comercial de Alta Calidad")
                st.markdown("""
                - **Perfil**: Cuerpo medio, sabor limpio
                - **Precio estimado**: $15-30/kg
                - **Recomendación**: Mercado comercial premium
                """)
            elif prediccion >= 6:
                st.warning("### ⚠️ REGULAR - Café Comercial Estándar")
                st.markdown("""
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
            
            # Recomendaciones personalizadas
            st.subheader("💡 Recomendaciones")
            if altitud < 1200:
                st.info("🌱 Considerar variedades resistentes a bajas altitudes")
            if temperatura > 25:
                st.info("☀️ Implementar sombra para reducir temperatura")
            if prediccion < 7:
                st.info("📊 Revisar prácticas de cosecha y fermentación")
    
    # Mostrar historial si se solicitó
    if st.session_state.get('show_history', False):
        with st.expander("📜 Mi Historial de Predicciones", expanded=True):
            st.info("📊 Aquí se mostrarán tus predicciones guardadas")
            st.caption("💡 Próximamente: Guardado automático de predicciones")
            if st.button("Cerrar historial"):
                st.session_state.show_history = False
                st.rerun()
    
    # Mostrar recomendaciones si se solicitó
    if st.session_state.get('show_recommendations', False):
        with st.expander("💡 Recomendaciones para Mejorar la Calidad", expanded=True):
            st.markdown("""
            ### 📋 Guía de Buenas Prácticas para Caficultores
            
            #### 🌱 **Cultivo**
            - ✅ Mantener altitud entre 1200-2000 msnm para mejor calidad
            - ✅ Temperatura óptima: 18-24°C
            - ✅ Implementar sombra regulada (20-40%)
            - ✅ Realizar análisis de suelo anual
            
            #### 🍒 **Cosecha**
            - ✅ Cosecha selectiva (solo cerezas rojas)
            - ✅ Frecuencia: cada 8-10 días
            - ✅ Beneficio dentro de 6-8 horas post-cosecha
            
            #### 🏭 **Beneficio**
            - ✅ Fermentación controlada (12-24 horas)
            - ✅ Secado uniforme (humedad 10-12%)
            - ✅ Almacenamiento en pergamino
            
            #### 📈 **Certificaciones**
            - 🌍 Certificación Orgánica
            - 🤝 Comercio Justo
            - 🌱 Rainforest Alliance
            - ☕ Specialty Coffee Association (SCA)
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
        <p><small>ℹ️ Herramienta de apoyo diagnóstico. Validar resultados con especialistas en café.</small></p>
    </div>
    """, unsafe_allow_html=True)

# ========== PUNTO DE ENTRADA ==========
if not st.session_state.logged_in:
    show_login_ui()
else:
    main_app()
