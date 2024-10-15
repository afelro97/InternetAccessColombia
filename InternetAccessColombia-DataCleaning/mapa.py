import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster

# Configuración para usar el ancho completo de la página
st.set_page_config(layout="wide")

# Cargar el dataset unido y usar st.cache_data para mejorar el rendimiento
@st.cache_data
def cargar_datos():
    return pd.read_csv('df_unificado_limpio.csv')

df_unido_limpio = cargar_datos()

# Función para generar el mapa
def generar_mapa(departamento, municipio, accesos, limite):
    if departamento == 'Todos':
        if municipio == 'Todos':
            df_filtrado = df_unido_limpio[df_unido_limpio['No. ACCESOS FIJOS A INTERNET'] >= accesos]
        else:
            df_filtrado = df_unido_limpio[(df_unido_limpio['MUNICIPIO'] == municipio) &
                                          (df_unido_limpio['No. ACCESOS FIJOS A INTERNET'] >= accesos)]
    else:
        if municipio == 'Todos':
            df_filtrado = df_unido_limpio[(df_unido_limpio['DEPARTAMENTO'] == departamento) &
                                          (df_unido_limpio['No. ACCESOS FIJOS A INTERNET'] >= accesos)]
        else:
            df_filtrado = df_unido_limpio[(df_unido_limpio['DEPARTAMENTO'] == departamento) &
                                          (df_unido_limpio['MUNICIPIO'] == municipio) &
                                          (df_unido_limpio['No. ACCESOS FIJOS A INTERNET'] >= accesos)]

    # Limitar el número de puntos a mostrar al tamaño real de los datos filtrados
    limite = min(limite, len(df_filtrado))

    # Tomar una muestra aleatoria de los datos si el límite es menor que la cantidad de datos disponibles
    if limite > 0:
        df_filtrado = df_filtrado.sample(n=limite, random_state=42)

    # Crear mapa centrado en Colombia
    mapa = folium.Map(location=[4.570868, -74.297333], zoom_start=6)

    # Crear un cluster para los marcadores
    marker_cluster = MarkerCluster().add_to(mapa)

    for _, row in df_filtrado.iterrows():
        folium.Marker([row['Latitud'], row['Longitud']],
                      popup=f"Proveedor: {row['PROVEEDOR']}<br>Tecnología: {row['TECNOLOGÍA']}<br>Accesos: {row['No. ACCESOS FIJOS A INTERNET']}").add_to(marker_cluster)

    return mapa

# Sidebar para seleccionar departamento, municipio y accesos
st.sidebar.title("Filtrar datos del mapa")

departamentos = ['Todos'] + list(df_unido_limpio['DEPARTAMENTO'].unique())
departamento_seleccionado = st.sidebar.selectbox('Selecciona el departamento', departamentos)

if departamento_seleccionado == 'Todos':
    municipios = ['Todos']
else:
    municipios = ['Todos'] + list(df_unido_limpio[df_unido_limpio['DEPARTAMENTO'] == departamento_seleccionado]['MUNICIPIO'].unique())

municipio_seleccionado = st.sidebar.selectbox('Selecciona el municipio', municipios)

accesos_seleccionados = st.sidebar.slider('Número mínimo de accesos', min_value=int(df_unido_limpio['No. ACCESOS FIJOS A INTERNET'].min()),
                                          max_value=int(df_unido_limpio['No. ACCESOS FIJOS A INTERNET'].max()), value=0)

limite_puntos = st.sidebar.slider('Número máximo de puntos a mostrar', min_value=1, max_value=100, value=10)

# Generar el mapa con los filtros aplicados
mapa_generado = generar_mapa(departamento_seleccionado, municipio_seleccionado, accesos_seleccionados, limite_puntos)

# Mostrar el mapa en Streamlit con mayor tamaño
folium_static(mapa_generado, width=1500, height=800)

# Mostrar información adicional
st.write(f"Departamento seleccionado: {departamento_seleccionado}")
st.write(f"Municipio seleccionado: {municipio_seleccionado}")
st.write(f"Número mínimo de accesos: {accesos_seleccionados}")
st.write(f"Número máximo de puntos mostrados: {limite_puntos}")
