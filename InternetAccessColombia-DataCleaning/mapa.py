import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap

# Configuración para usar el ancho completo de la página
st.set_page_config(layout="wide")

# Agregar CSS personalizado para ajustar el layout y hacerlo responsive
st.markdown("""
    <style>
    .streamlit-expanderHeader, .streamlit-expanderBody {
        padding-left: 10px;
        padding-right: 10px;
    }
    iframe {
        width: 100% !important;
        height: 700px !important;
    }
    [data-testid="stElementToolbar"] {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)

# Función para cargar los datos desde S3
@st.cache_data
def cargar_datos():
    return pd.read_csv('https://datainternetaccess-cleaning.s3.us-west-1.amazonaws.com/df_unificado_limpio_actualizado.csv')

df_unido_limpio = cargar_datos()

# Obtener lista de departamentos únicos directamente del dataset cargado
todos_departamentos = ['Todos'] + sorted(df_unido_limpio['DEPARTAMENTO'].unique())

# Sidebar para seleccionar departamento y municipio
st.sidebar.title("Filtrar datos del mapa")
departamento_seleccionado = st.sidebar.selectbox('Selecciona el departamento', todos_departamentos)

# Filtrar municipios según el departamento seleccionado
if departamento_seleccionado == 'Todos':
    df_filtrado_departamento = df_unido_limpio
else:
    df_filtrado_departamento = df_unido_limpio[df_unido_limpio['DEPARTAMENTO'] == departamento_seleccionado]

municipios = ['Todos'] + sorted(df_filtrado_departamento['MUNICIPIO'].unique())
municipio_seleccionado = st.sidebar.selectbox('Selecciona el municipio', municipios)

# Selección del límite de puntos a mostrar
limite_puntos = st.sidebar.slider('Número máximo de datos a mostrar', min_value=1, max_value=1000, value=100)

# Función para generar el mapa
def generar_mapa(df, municipio, limite):
    if municipio != 'Todos':
        df = df[df['MUNICIPIO'] == municipio]

    limite = min(limite, len(df))
    df = df.head(limite)

    if df.empty:
        st.warning("No hay datos para mostrar con los filtros seleccionados.")
        return folium.Map(location=[4.570868, -74.297333], zoom_start=6)

    centro = [df['Latitud'].mean(), df['Longitud'].mean()]
    mapa = folium.Map(location=centro, zoom_start=7)
    heat_data = [[row['Latitud'], row['Longitud']] for _, row in df.iterrows()]
    HeatMap(heat_data).add_to(mapa)

    return mapa

# Generar el mapa con los filtros aplicados
mapa_generado = generar_mapa(df_filtrado_departamento, municipio_seleccionado, limite_puntos)
folium_static(mapa_generado, width=1300, height=700)

# Mostrar detalles adicionales del filtrado
st.write(f"Departamento seleccionado: {departamento_seleccionado}")
st.write(f"Municipio seleccionado: {municipio_seleccionado}")
st.write(f"Número máximo de puntos mostrados: {limite_puntos}")

st.subheader("Detalles de los puntos seleccionados:")
st.dataframe(df_filtrado_departamento[['PROVEEDOR', 'MUNICIPIO', 'SEGMENTO', 'TECNOLOGÍA', 'VELOCIDAD BAJADA', 'VELOCIDAD SUBIDA', 'No. ACCESOS FIJOS A INTERNET']].reset_index(drop=True))
