import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. Configuración de la página (Título, Icono, Layout)
# ---------------------------------------------------------
st.set_page_config(
    page_title="P.N. Calilegua | Climate Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo personalizado básico
st.markdown("""
    <style>
    .main-header {
        font-size:2.3rem;
        font-weight:700;
        color: #2E7D32;
    }
    .sub-header {
        font-size:1.1rem;
        color: #555555;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Encabezado y Descripción
# ---------------------------------------------------------
st.markdown('<p class="main-header">🌿 Parque Nacional Calilegua — Análisis Climático</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Dashboard interactivo de monitoreo de temperatura y precipitación en el bioma de las Yungas (Jujuy, Argentina).</p>', unsafe_allow_html=True)
st.divider()

# ---------------------------------------------------------
# 3. Sidebar (Controles e Información)
# ---------------------------------------------------------
st.sidebar.header("⚙️ Parámetros")
year_selected = st.sidebar.selectbox("Selecciona el año de análisis:", [2023, 2022], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("📌 Sobre el Proyecto")
st.sidebar.info(
    """
    **Autor:** Bruno Martín González  
    **Perfil:** Físico & Científico de Datos (Sistemas Complejos)  
    **Fuente de Datos:** API Histórica de Open-Meteo  
    **Coordenadas:** -23.6333, -64.8500 (P.N. Calilegua)
    """
)

# ---------------------------------------------------------
# 4. Extracción de Datos (con Caching de Streamlit)
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_weather_data(year):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": -23.6333,
        "longitude": -64.8500,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": ["temperature_2m", "precipitation"],
        "timezone": "America/Argentina/Jujuy"
    }
    res = requests.get(url, params=params)
    data = res.json()
    
    df = pd.DataFrame(data['hourly'])
    df.rename(columns={
        'time': 'fecha_hora',
        'temperature_2m': 'temperatura_c',
        'precipitation': 'precipitacion_mm'
    }, inplace=True)
    
    df['fecha_hora'] = pd.to_datetime(df['fecha_hora'])
    df.set_index('fecha_hora', inplace=True)
    df['temp_media_movil_7d'] = df['temperatura_c'].rolling(168, center=True).mean()
    return df

with st.spinner("Cargando datos meteorológicos desde la API..."):
    df = fetch_weather_data(year_selected)

# ---------------------------------------------------------
# 5. Tarjetas de Métricas Principales (KPIs)
# ---------------------------------------------------------
temp_avg = df['temperatura_c'].mean()
temp_max = df['temperatura_c'].max()
temp_min = df['temperatura_c'].min()
total_rain = df['precipitacion_mm'].sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("🌡️ Temp. Promedio", f"{temp_avg:.1f} °C")
col2.metric("🔥 Temp. Máxima", f"{temp_max:.1f} °C")
col3.metric("❄️ Temp. Mínima", f"{temp_min:.1f} °C")
col4.metric("🌧️ Lluvia Total", f"{total_rain:.1f} mm")

st.markdown("---")

# ---------------------------------------------------------
# 6. Pestañas de Análisis Visual
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Series Temporales", "🔥 Matriz Térmica", "🌿 Estacionalidad y Extremos"])

# --- TAB 1: Series Temporales ---
with tab1:
    st.subheader("Evolución de Temperatura y Precipitación")
    
    # Gráfico interactivo de Temperatura
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(x=df.index, y=df['temperatura_c'], mode='lines', name='Temp Horaria', line=dict(color='lightgray', width=1)))
    fig_temp.add_trace(go.Scatter(x=df.index, y=df['temp_media_movil_7d'], mode='lines', name='Media Móvil 7d', line=dict(color='#D32F2F', width=2.5)))
    fig_temp.update_layout(title="Temperatura Horaria vs Media Móvil (7 días)", xaxis_title="Fecha", yaxis_title="°C", template="plotly_white")
    st.plotly_chart(fig_temp, use_container_width=True)
    
    # Gráfico de Precipitación Diaria
    df_daily_rain = df['precipitacion_mm'].resample('D').sum().reset_index()
    fig_rain = px.bar(df_daily_rain, x='fecha_hora', y='precipitacion_mm', title="Precipitación Diaria Acumulada (mm)", labels={'fecha_hora': 'Fecha', 'precipitacion_mm': 'Lluvia (mm)'}, color_discrete_sequence=['#1976D2'])
    fig_rain.update_layout(template="plotly_white")
    st.plotly_chart(fig_rain, use_container_width=True)

# --- TAB 2: Heatmap Térmico ---
with tab2:
    st.subheader("Matriz de Temperatura: Hora del Día vs Mes")
    st.write("Visualización del comportamiento térmico para identificar picos de calor diarios según la época del año.")
    
    df['mes'] = df.index.strftime('%b')
    df['hora'] = df.index.hour
    
    pivot = df.pivot_table(index='mes', columns='hora', values='temperatura_c', aggfunc='mean')
    meses_orden = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot = pivot.reindex([m for m in meses_orden if m in pivot.index])
    
    fig_heatmap = px.imshow(
        pivot,
        labels=dict(x="Hora del Día (h)", y="Mes", color="Temp (°C)"),
        x=pivot.columns,
        y=pivot.index,
        color_continuous_scale="Viridis"
    )
    fig_heatmap.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig_heatmap, use_container_width=True)

# --- TAB 3: Estacionalidad y Extremos ---
with tab3:
    st.subheader("Comportamiento Estacional (Yungas)")
    
    def clasificar_estacion(m):
        return 'Estación Lluviosa (Nov-Abr)' if m in [11, 12, 1, 2, 3, 4] else 'Estación Seca (May-Oct)'
    
    df['estacion'] = df.index.month.map(clasificar_estacion)
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("### Resumen Estadístico por Estación")
        resumen = df.groupby('estacion').agg(
            Temp_Media=('temperatura_c', 'mean'),
            Lluvia_Total_mm=('precipitacion_mm', 'sum')
        ).reset_index()
        st.dataframe(resumen, use_container_width=True)
        
    with col_b:
        st.markdown("### Días Récord del Año")
        df_daily = df.resample('D').agg({'temperatura_c': ['min', 'max'], 'precipitacion_mm': 'sum'})
        df_daily.columns = ['t_min', 't_max', 'lluvia']
        
        max_t_day = df_daily['t_max'].idxmax()
        max_rain_day = df_daily['lluvia'].idxmax()
        
        st.write(f"🔥 **Día más caluroso:** {max_t_day.strftime('%d %B %Y')} ({df_daily.loc[max_t_day, 't_max']:.1f} °C)")
        st.write(f"🌧️ **Día más lluvioso:** {max_rain_day.strftime('%d %B %Y')} ({df_daily.loc[max_rain_day, 'lluvia']:.1f} mm)")