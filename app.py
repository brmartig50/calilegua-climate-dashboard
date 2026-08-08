import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import solve_ivp
from datetime import timedelta

# ---------------------------------------------------------
# 1. Configuración de la Página
# ---------------------------------------------------------
st.set_page_config(
    page_title="P.N. Calilegua | Eco-Physics Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS personalizado
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
        margin-bottom: 5px;
    }
    .author-title {
        font-size: 1rem;
        color: #1565C0;
        font-weight: 600;
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
# 2. Contexto (Siempre visible)
# ---------------------------------------------------------
st.markdown('<p class="main-title">🌿 Parque Nacional Calilegua: Diagnóstico Eco-Físico</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Series temporales multivariables y modelado dinámico de sistemas complejos en el bioma de las Yungas (Jujuy, Argentina)</p>', unsafe_allow_html=True)
st.markdown('<p class="author-title">👨‍🔬 Autor: Bruno Martín González | Física, Clima & Data Science</p>', unsafe_allow_html=True)

with st.container():
    st.markdown("""
    <div class="context-box">
    <b>Contexto Científico:</b> El Parque Nacional Calilegua resguarda una muestra clave de las <i>Yungas</i> (selva de montaña). 
    Este dashboard combina la ingesta de datos meteorológicos y edáficos por API con <b>modelado físico basado en ecuaciones diferenciales (EDO)</b> 
    y análisis topológico en el <b>espacio de estados</b> para evaluar el balance hídrico y la inercia térmica del ecosistema.
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. Sidebar (Controles y Navegación)
# ---------------------------------------------------------
st.sidebar.header("⚙️ Panel de Control")
year_selected = st.sidebar.selectbox("📅 Selecciona el Año de Estudio:", [2024, 2023, 2022], index=0)

st.sidebar.divider()

selected_view = st.sidebar.radio(
    "📊 Selecciona la Vista de Análisis:",
    options=[
        "1. Balance Hídrico & Suelo", 
        "2. Microclima & Atmósfera", 
        "3. Matriz Térmica Mensual", 
        "4. Modelado Físico (EDO) & Espacio de Estados",
        "5. Datos Crudos & Exportación"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("[💻 Mi GitHub](https://github.com/brmartig50)")
st.sidebar.markdown("[🔗 Mi LinkedIn](https://www.linkedin.com/in/bruno-mart%C3%ADn-gonz%C3%A1lez-96349a245/)")

# ---------------------------------------------------------
# 4. Ingesta de Datos Multi-Variable Segura (Manejo de Errores)
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
    
    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status() # Lanza un error si la API rechaza la llamada
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
    except Exception as e:
        return pd.DataFrame() # Retorna dataframe vacío en caso de error

with st.spinner("Descargando parámetros climáticos y edáficos desde la API..."):
    df = fetch_eco_data(year_selected)

if df.empty:
    st.error("❌ Error conectando con la API climática (Open-Meteo). Por favor, compruebe su conexión o inténtelo de nuevo más tarde.")
    st.stop()

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

# Diccionario auxiliar para traducir meses en los textos dinámicos
meses_es = {1:'Enero', 2:'Febrero', 3:'Marzo', 4:'Abril', 5:'Mayo', 6:'Junio', 
            7:'Julio', 8:'Agosto', 9:'Septiembre', 10:'Octubre', 11:'Noviembre', 12:'Diciembre'}

# ---------------------------------------------------------
# 6. Vistas Dinámicas Controladas por el Menú Lateral
# ---------------------------------------------------------

if selected_view == "1. Balance Hídrico & Suelo":
    st.subheader(f"Relación Hidrológica en {year_selected}")
    
    df_daily = df.resample('D').agg({
        'precipitacion_mm': 'sum',
        'evapotranspiracion_mm': 'sum',
        'humedad_suelo': 'mean'
    }).reset_index()

    fig_hidro = make_subplots(specs=[[{"secondary_y": True}]])
    fig_hidro.add_trace(go.Bar(x=df_daily['fecha_hora'], y=df_daily['precipitacion_mm'], name="Precipitación (mm)", marker_color='#29B6F6', opacity=0.7), secondary_y=False)
    fig_hidro.add_trace(go.Scatter(x=df_daily['fecha_hora'], y=df_daily['evapotranspiracion_mm'], name="Evapotranspiración (mm)", line=dict(color='#FFA726', width=2)), secondary_y=False)
    fig_hidro.add_trace(go.Scatter(x=df_daily['fecha_hora'], y=df_daily['humedad_suelo'], name="Humedad Suelo (m³/m³)", line=dict(color='#2E7D32', width=2.5)), secondary_y=True)

    fig_hidro.update_layout(title_text="Dinámica Eco-Hidrológica Diaria", template="plotly_white", height=450)
    fig_hidro.update_yaxes(title_text="Agua (mm)", secondary_y=False)
    fig_hidro.update_yaxes(title_text="Humedad del Suelo (m³/m³)", secondary_y=True)
    
    st.plotly_chart(fig_hidro, use_container_width=True)

    # Cálculo dinámico para conclusiones
    mes_min_humedad = meses_es[df_daily.loc[df_daily['humedad_suelo'].idxmin(), 'fecha_hora'].month]
    max_lluvia_dia = df_daily['precipitacion_mm'].max()

    st.info(f"""
    💡 **Conclusiones del Análisis Hidrológico ({year_selected}):**
    * **Punto Máximo de Estrés Hídrico:** El vaciado crítico de la reserva de agua edáfica ocurrió en **{mes_min_humedad}**, evidenciando la etapa más dura para la vegetación de las Yungas en este periodo.
    * **Respuesta Rápida Superficial:** Las tormentas intensas (como el máximo registrado de {max_lluvia_dia:.1f} mm/día) generan picos inmediatos de absorción, pero la humedad edáfica superficial drena rápidamente debido al consumo por el denso sotobosque.
    """)

elif selected_view == "2. Microclima & Atmósfera":
    st.subheader(f"Evolución Térmica y Humedad Relativa ({year_selected})")
    
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(x=df.index, y=df['temp_c'], mode='lines', name='Temp Horaria', line=dict(color='lightgray', width=1)))
    fig_temp.add_trace(go.Scatter(x=df.index, y=df['temp_media_movil_7d'], mode='lines', name='Media Móvil (7d)', line=dict(color='#D32F2F', width=2.5)))
    fig_temp.update_layout(title="Serie Temporal de Temperatura (°C)", xaxis_title="Fecha", yaxis_title="°C", template="plotly_white", height=400)
    
    st.plotly_chart(fig_temp, use_container_width=True)

    # Cálculo dinámico para conclusiones
    temp_max_yr = df['temp_c'].max()
    temp_min_yr = df['temp_c'].min()

    st.success(f"""
    💡 **Conclusiones Microclimáticas:**
    * **Amplitud Térmica Controlada:** Durante {year_selected}, las temperaturas del ecosistema oscilaron entre **{temp_min_yr:.1f} °C y {temp_max_yr:.1f} °C**. Esta contención de extremos térmicos corrobora el importante papel regulador de la biomasa forestal.
    * **Inercia Estacional:** La media móvil de 7 días revela transiciones térmicas muy atenuadas entre estaciones, evitando choques ambientales letales para la flora endémica.
    """)

elif selected_view == "3. Matriz Térmica Mensual":
    st.subheader(f"Matriz de Carga Térmica ({year_selected})")
    
    df_temp = df.copy()
    df_temp['mes_num'] = df_temp.index.month
    df_temp['mes_nombre'] = df_temp['mes_num'].map(meses_es)
    df_temp['hora'] = df_temp.index.hour
    
    pivot = df_temp.pivot_table(index='mes_nombre', columns='hora', values='temp_c', aggfunc='mean')
    orden_meses = [meses_es[i] for i in range(1, 13)]
    pivot = pivot.reindex([m for m in orden_meses if m in pivot.index])
    
    fig_heatmap = px.imshow(
        pivot,
        labels=dict(x="Hora del Día", y="Mes", color="Temp Media (°C)"),
        x=pivot.columns,
        y=pivot.index,
        color_continuous_scale="YlOrRd"
    )
    fig_heatmap.update_layout(template="plotly_white", height=450)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    hora_max = pivot.mean().idxmax()
    st.info(f"""
    💡 **Conclusiones de la Matriz Térmica:**
    * **Pico Térmico Diario:** En {year_selected}, la mayor carga térmica del sistema converge consistentemente alrededor de las **{hora_max:02d}:00 h**. Esta franja temporal impulsa la tasa máxima de evapotranspiración vegetal.
    * **Condensación Nocturna:** Sin importar la dureza del verano, las madrugadas logran descender significativamente de temperatura, habilitando la condensación oclusa (lluvia horizontal) en el follaje.
    """)

elif selected_view == "4. Modelado Físico (EDO) & Espacio de Estados":
    st.subheader("🔬 Dinámica de Inercia Térmica (Ecuaciones Diferenciales)")
    st.markdown("""
    Modelo físico de **Balance de Energía basado en Relajación de Newton**. 
    Modelamos la variación de la temperatura $T(t)$ forzada por la radiación solar incidente $R(t)$ y disipada por la transferencia convectiva ambiental:
    
    $$ \\frac{dT}{dt} = -\\frac{1}{\\tau} (T - T_{\\text{base}}) + \\alpha R(t) $$
    """)
    
    # Controles Dinámicos para el Modelo (Sin Hardcode)
    col_date, col_tau, col_alpha, col_offset = st.columns(4)
    min_date = df.index.min().date()
    max_date = df.index.max().date() - timedelta(days=7) # Ventana límite de 7 días
    
    with col_date:
        start_date = st.date_input("Inicio Simulación (7 días):", value=min_date + timedelta(days=15), min_value=min_date, max_value=max_date)
    with col_tau:
        tau_val = st.slider("Inercia (τ horas):", min_value=1.0, max_value=24.0, value=6.0, step=0.5)
    with col_alpha:
        alpha_val = st.slider("Absorción Radiativa (α):", min_value=0.001, max_value=0.05, value=0.015, step=0.001, format="%.3f")
    with col_offset:
        offset_val = st.slider("Offset T.Base (°C):", min_value=-15.0, max_value=5.0, value=-5.0, step=0.5)

    # Simulación acotada por las fechas del usuario
    end_date = start_date + timedelta(days=7)
    mask = (df.index.date >= start_date) & (df.index.date < end_date)
    sub_df = df.loc[mask].copy()

    if not sub_df.empty:
        time_hours = np.arange(len(sub_df))
        rad_data = sub_df['radiacion_solar'].values
        T_real = sub_df['temp_c'].values
        T0 = T_real[0]
        
        # Parámetro termodinámico justificado
        T_base = np.mean(T_real) + offset_val 

        def thermal_ode(t, T):
            idx = int(np.clip(t, 0, len(rad_data)-1))
            R_t = rad_data[idx]
            dTdt = -(1.0 / tau_val) * (T[0] - T_base) + alpha_val * R_t
            return [dTdt]

        sol = solve_ivp(thermal_ode, [0, len(time_hours)-1], [T0], t_eval=time_hours, method='RK45')

        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(x=sub_df.index, y=T_real, mode='lines', name='Medición Real (API)', line=dict(color='#2E7D32', width=2)))
        fig_sim.add_trace(go.Scatter(x=sub_df.index, y=sol.y[0], mode='lines', name=f'Simulación EDO', line=dict(color='#D32F2F', width=2.5, dash='dash')))
        
        fig_sim.update_layout(
            title=f"Validación Numérica: Semana del {start_date.strftime('%d/%m/%Y')}",
            xaxis_title="Fecha / Hora", yaxis_title="Temperatura (°C)",
            template="plotly_white", height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_sim, use_container_width=True)

    st.divider()

    # --- ESPACIO DE ESTADOS ---
    st.subheader("🌀 Espacio de Estados: Densidad Topológica")
    st.markdown("Proyección de la estabilidad del ecosistema en el plano de estados $[T(t) \\text{ vs. } \\text{Humedad del Suelo}(t)]$. Se utiliza un modelo de contorno de densidad 2D para evaluar la cuenca de atracción.")

    fig_phase = go.Figure(go.Histogram2dContour(
        x=df['temp_c'], 
        y=df['humedad_suelo'],
        colorscale='Turbo',
        contours=dict(showlines=False)
    ))

    fig_phase.update_layout(
        title=f"Densidad de Estados del Ecosistema ({year_selected})",
        xaxis_title="Temperatura (°C)",
        yaxis_title="Humedad del Suelo (m³/m³)",
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig_phase, use_container_width=True)

    st.info("""
    💡 **Interpretación Físico-Matemática:**
    * **Cuenca de Atracción:** Las áreas más cálidas (rojo/amarillo) en el mapa de densidad representan los regímenes más estables del ecosistema a lo largo del año (zonas de alta probabilidad donde el ecosistema pasa la mayor parte del tiempo).
    * **Transiciones No Estables:** Las densidades bajas (azul) demuestran que el sistema no se detiene en estados de transición (enfriamientos bruscos o desecaciones repentinas), lo que denota una alta resiliencia topológica.
    """)

elif selected_view == "5. Datos Crudos & Exportación":
    st.subheader("Exploración del Dataframe Base")
    st.dataframe(df, use_container_width=True)
    
    csv = df.to_csv().encode('utf-8')
    st.download_button(
        label=f"📥 Descargar CSV Analítico ({year_selected})",
        data=csv,
        file_name=f"calilegua_eco_physics_{year_selected}.csv",
        mime="text/csv"
    )
