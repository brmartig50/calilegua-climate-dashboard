import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------
# 1. Configuración de la Página
# ---------------------------------------------------------
st.set_page_config(
    page_title="P.N. Calilegua | Eco-Data Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Personalizado para mejorar el aspecto visual
st.markdown("""
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1B5E20;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #424242;
        margin-bottom: 20px;
    }
    .context-box {
        background-color: #F1F8E9;
        border-left: 5px solid #558B2F;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Hero Section & Contexto Ecológico
# ---------------------------------------------------------
st.markdown('<p class="main-title">🌿 Parque Nacional Calilegua: Diagnóstico Eco-Hidrológico</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Análisis de series temporales multivariables para la conservación del bioma de las Yungas (Jujuy, Argentina)</p>', unsafe_allow_html=True)

with st.container():
    st.markdown("""
    <div class="context-box">
    <b>Contexto Ecológico:</b> El Parque Nacional Calilegua resguarda una de las muestras más representativas de las <i>Yungas</i> (selvas de montaña). 
    Este ecosistema depende de un delicado equilibrio hídrico: las precipitaciones estacionales y la condensación de humedad en el dosel forestal 
    sostienen la biodiversidad durante los meses secos. Este dashboard analiza la interacción entre temperatura, humedad del suelo y balance hídrico.
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. Sidebar (Controles y Parámetros)
# ---------------------------------------------------------
st.sidebar.header("⚙️ Configuración del Análisis")
year_selected = st.sidebar.selectbox("Selecciona el Año de Estudio:", [2023, 2022], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("👨‍🔬 Autor")
st.sidebar.markdown("""
**Bruno Martín González**  
*Físico & Data Scientist*  
[LinkedIn](https://www.linkedin.com/in/bruno-mart%C3%ADn-gonz%C3%A1lez-96349a245/) | [GitHub](https://github.com/)
""")

# ---------------------------------------------------------
# 4. Ingesta de Datos Multi-Variable (API Open-Meteo)
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_eco_data(year):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": -23.6333,
        "longitude": -64.8500,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "soil_moisture_0_to_7cm",
            "et0_fao_evapotranspiration",
            "shortwave_radiation"
        ],
        "timezone": "America/Argentina/Jujuy"
    }
    res = requests.get(url, params=params)
    data = res.json()
    
    df = pd.DataFrame(data['hourly'])
    df.rename(columns={
        'time': 'fecha_hora',
        'temperature_2m': 'temp_c',
        'relative_humidity_2m': 'humedad_relativa',
        'precipitation': 'precipitacion_mm',
        'soil_moisture_0_to_7cm': 'humedad_suelo',
        'et0_fao_evapotranspiration': 'evapotranspiracion_mm',
        'shortwave_radiation': 'radiacion_solar'
    }, inplace=True)
    
    df['fecha_hora'] = pd.to_datetime(df['fecha_hora'])
    df.set_index('fecha_hora', inplace=True)
    
    # Feature Engineering
    df['temp_media_movil_7d'] = df['temp_c'].rolling(168, center=True).mean()
    return df

with st.spinner("Descargando parámetros climáticos y edáficos desde la API..."):
    df = fetch_eco_data(year_selected)

# ---------------------------------------------------------
# 5. Tarjetas de Indicadores Clave (KPIs Eco-Climáticos)
# ---------------------------------------------------------
total_rain = df['precipitacion_mm'].sum()
total_et0 = df['evapotranspiracion_mm'].sum()
balance_hidrico = total_rain - total_et0
humedad_suelo_prom = df['humedad_suelo'].mean()
temp_prom = df['temp_c'].mean()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("🌡️ Temp. Media", f"{temp_prom:.1f} °C")
col2.metric("🌧️ Lluvia Acumulada", f"{total_rain:.1f} mm")
col3.metric("💧 Evapotranspiración", f"{total_et0:.1f} mm")
col4.metric("⚖️ Balance Hídrico", f"{balance_hidrico:.1f} mm", 
            delta="Superávit" if balance_hidrico > 0 else "Déficit",
            delta_color="normal" if balance_hidrico > 0 else "inverse")
col5.metric("🌱 Humedad Suelo (0-7cm)", f"{humedad_suelo_prom:.3f} m³/m³")

st.markdown("---")

# ---------------------------------------------------------
# 6. Pestañas de Análisis Detallado
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Balance Hídrico & Suelo", 
    "🌡️ Microclima & Atmósfera", 
    "🔥 Matriz Térmica Mensual", 
    "📋 Datos Crudos & Exportación"
])

# --- TAB 1: BALANCE HÍDRICO Y SUELO ---
with tab1:
    st.subheader("Relación entre Precipitación, Evapotranspiración y Humedad del Suelo")
    st.markdown("""
    En las Yungas, la pérdida de agua por evapotranspiración en época seca suele superar la precipitación directa. 
    Observa cómo la **humedad del suelo (línea verde)** reacciona a los eventos intensos de lluvia y cae durante el invierno.
    """)
    
    # Resample diario
    df_daily = df.resample('D').agg({
        'precipitacion_mm': 'sum',
        'evapotranspiracion_mm': 'sum',
        'humedad_suelo': 'mean'
    }).reset_index()

    fig_hidro = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig_hidro.add_trace(
        go.Bar(x=df_daily['fecha_hora'], y=df_daily['precipitacion_mm'], name="Precipitación (mm)", marker_color='#29B6F6', opacity=0.7),
        secondary_y=False
    )
    fig_hidro.add_trace(
        go.Scatter(x=df_daily['fecha_hora'], y=df_daily['evapotranspiracion_mm'], name="Evapotranspiración (mm)", line=dict(color='#FFA726', width=2)),
        secondary_y=False
    )
    fig_hidro.add_trace(
        go.Scatter(x=df_daily['fecha_hora'], y=df_daily['humedad_suelo'], name="Humedad Suelo (m³/m³)", line=dict(color='#2E7D32', width=2.5)),
        secondary_y=True
    )

    fig_hidro.update_layout(title_text="Dinámica Eco-Hidrológica Diaria", template="plotly_white", height=450)
    fig_hidro.update_yaxes(title_text="Agua (mm)", secondary_y=False)
    fig_hidro.update_yaxes(title_text="Humedad del Suelo (m³/m³)", secondary_y=True)
    
    st.plotly_chart(fig_hidro, use_container_width=True)

# --- TAB 2: MICROCLIMA Y ATMÓSFERA ---
with tab2:
    st.subheader("Evolución Térmica y Humedad Relativa")
    st.markdown("La **humedad relativa** elevada es la responsable de la formación de nieblas montanas que caracterizan al sotobosque de Calilegua.")
    
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(x=df.index, y=df['temp_c'], mode='lines', name='Temp Horaria', line=dict(color='lightgray', width=1)))
    fig_temp.add_trace(go.Scatter(x=df.index, y=df['temp_media_movil_7d'], mode='lines', name='Media Móvil (7d)', line=dict(color='#D32F2F', width=2.5)))
    fig_temp.update_layout(title="Serie Temporal de Temperatura (°C)", xaxis_title="Fecha", yaxis_title="°C", template="plotly_white", height=400)
    
    st.plotly_chart(fig_temp, use_container_width=True)

# --- TAB 3: MATRIZ TÉRMICA ---
with tab3:
    st.subheader("Matriz de Temperatura Promedio: Hora vs Mes")
    st.markdown("Este mapa de calor identifica las ventanas horarias de mayor estrés térmico a lo largo del año.")
    
    df['mes'] = df.index.strftime('%b')
    df['hora'] = df.index.hour
    
    pivot = df.pivot_table(index='mes', columns='hora', values='temp_c', aggfunc='mean')
    meses_orden = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot = pivot.reindex([m for m in meses_orden if m in pivot.index])
    
    fig_heatmap = px.imshow(
        pivot,
        labels=dict(x="Hora del Día", y="Mes", color="Temp (°C)"),
        x=pivot.columns,
        y=pivot.index,
        color_continuous_scale="YlOrRd"
    )
    fig_heatmap.update_layout(template="plotly_white", height=450)
    st.plotly_chart(fig_heatmap, use_container_width=True)

# --- TAB 4: DATOS CRUDOS ---
with tab4:
    st.subheader("Exploración de Datos Limpios")
    st.dataframe(df, use_container_width=True)
    
    # Botón de descarga de datos
    csv = df.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Descargar Dataset Limpio en CSV",
        data=csv,
        file_name=f"calilegua_eco_data_{year_selected}.csv",
        mime="text/csv"
    )
