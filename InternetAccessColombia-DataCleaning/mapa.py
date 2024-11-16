import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap
import os

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

# Función para cargar los datos con base en la selección
@st.cache_data
def cargar_datos(departamento):
    if departamento == 'Todos':
        # Cargar el dataset completo si se selecciona 'Todos'
        print("Cargando el dataset completo: df_unificado_limpio.csv")
        return pd.read_csv('https://datainternetaccess-cleaning.s3.us-west-1.amazonaws.com/df_unificado_limpio.csv')
    else:
        # Cargar el dataset específico del departamento
        file_path = f'Limpieza/data/departamentos/{departamento}.csv'
        if os.path.exists(file_path):
            print(f"Cargando el dataset para el departamento: {departamento}")
            return pd.read_csv(file_path)
        else:
            st.warning(f"No se encontró el archivo para el departamento: {departamento}.")
            print(f"Archivo no encontrado para el departamento: {departamento}")
            return pd.DataFrame()  # Retornar DataFrame vacío si no existe el archivo
# Sidebar para seleccionar departamento y municipio
st.sidebar.title("Filtrar datos del mapa")

# Obtener lista de departamentos únicos
todos_departamentos = ['Todos'] + sorted([file.replace('.csv', '') for file in os.listdir('Limpieza/data/departamentos') if file.endswith('.csv')])
departamento_seleccionado = st.sidebar.selectbox('Selecciona el departamento', todos_departamentos)

# Cargar los datos del departamento seleccionado
df_unido_limpio = cargar_datos(departamento_seleccionado)

# Validar si hay datos antes de continuar
if df_unido_limpio.empty:
    st.warning("No hay datos disponibles para los filtros seleccionados.")
else:
    municipios = ['Todos'] + sorted(df_unido_limpio['MUNICIPIO'].unique())

    municipio_seleccionado = st.sidebar.selectbox('Selecciona el municipio', municipios)

    # Selección del límite de puntos a mostrar
    limite_puntos = st.sidebar.slider('Número máximo de datos a mostrar', min_value=1, max_value=1000, value=100)

    # Función para generar el mapa
    def generar_mapa(municipio, limite):
        df_filtrado = df_unido_limpio.copy()

        if municipio != 'Todos':
            df_filtrado = df_filtrado[df_filtrado['MUNICIPIO'] == municipio]

        limite = min(limite, len(df_filtrado))
        df_filtrado = df_filtrado.head(limite)

        # Verificar si hay datos después del filtrado
        if df_filtrado.empty:
            st.warning("No hay datos para mostrar con los filtros seleccionados.")
            return folium.Map(location=[4.570868, -74.297333], zoom_start=6), df_filtrado

        centro = [df_filtrado['Latitud'].mean(), df_filtrado['Longitud'].mean()]
        mapa = folium.Map(location=centro, zoom_start=7)

        # Generar HeatMap
        heat_data = [[row['Latitud'], row['Longitud']] for _, row in df_filtrado.iterrows()]
        HeatMap(heat_data).add_to(mapa)

        return mapa, df_filtrado

    # Generar el mapa con los filtros aplicados
    mapa_generado, df_filtrado = generar_mapa(municipio_seleccionado, limite_puntos)

    # Mostrar el mapa en Streamlit
    folium_static(mapa_generado, width=1300, height=700)

    # Mostrar detalles adicionales del filtrado
    st.write(f"Departamento seleccionado: {departamento_seleccionado}")
    st.write(f"Municipio seleccionado: {municipio_seleccionado}")
    st.write(f"Número máximo de puntos mostrados: {limite_puntos}")

    # Mostrar los puntos seleccionados
    st.subheader("Detalles de los puntos seleccionados:")
    st.dataframe(df_filtrado[['PROVEEDOR', 'MUNICIPIO', 'SEGMENTO', 'TECNOLOGÍA', 'VELOCIDAD BAJADA', 'VELOCIDAD SUBIDA', 'No. ACCESOS FIJOS A INTERNET']].reset_index(drop=True))
