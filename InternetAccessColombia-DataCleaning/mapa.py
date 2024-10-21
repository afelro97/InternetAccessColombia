import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster, HeatMap

# Configuración para usar el ancho completo de la página
st.set_page_config(layout="wide")

# Agregar CSS personalizado para ajustar el layout y hacerlo responsive
st.markdown("""
    <style>
    /* Ajuste del contenedor del mapa */
    .streamlit-expanderHeader, .streamlit-expanderBody {
        padding-left: 10px;
        padding-right: 10px;
    }
    
    /* Hacer que el mapa ocupe toda la pantalla disponible */
    #map {
        width: 100%;
        height: calc(100vh - 150px);  /* Ajuste dinámico de altura según la pantalla */
    }
    
    /* Ajuste general para pantallas pequeñas */
    @media (max-width: 768px) {
        .css-1d391kg {
            margin-left: 0;
            margin-right: 0;
        }
        .css-18e3th9 {
            padding-top: 10px;
            padding-left: 5px;
            padding-right: 5px;
        }
        /* Ajustes para el mapa en pantallas pequeñas */
        iframe {
            width: 100% !important;
            height: 400px !important;
        }
    }
    
    /* Ajustes para pantallas medianas */
    @media (min-width: 768px) and (max-width: 1200px) {
        iframe {
            width: 100% !important;
            height: 600px !important;
        }
    }

    /* Ajustes para pantallas grandes */
    @media (min-width: 1200px) {
        iframe {
            width: 100% !important;
            height: 800px !important;
        }
    }
    
    </style>
""", unsafe_allow_html=True)

# Cargar el dataset unido y usar st.cache_data para mejorar el rendimiento
@st.cache_data
def cargar_datos():
    return pd.read_csv('Limpieza/data/df_unificado_limpio.csv')

df_unido_limpio = cargar_datos()

# Función para generar el heatmap
def generar_heatmap(departamento, municipio, accesos, limite):
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

    # Crear un HeatMap
    heat_data = [[row['Latitud'], row['Longitud'], row['No. ACCESOS FIJOS A INTERNET']] for _, row in df_filtrado.iterrows()]
    HeatMap(heat_data, radius=10).add_to(mapa)  # Ajusta el radio según sea necesario

    return mapa

# Sidebar para seleccionar departamento, municipio y accesos
st.sidebar.title("Filtrar datos del mapa")

# Selección de departamento
departamentos = ['Todos'] + list(df_unido_limpio['DEPARTAMENTO'].unique())
departamento_seleccionado = st.sidebar.selectbox('Selecciona el departamento', departamentos)

# Selección de municipio
if departamento_seleccionado == 'Todos':
    municipios = ['Todos']
    municipio_seleccionado = 'Todos'
    max_accesos = df_unido_limpio['No. ACCESOS FIJOS A INTERNET'].max()
else:
    municipios = ['Todos'] + list(df_unido_limpio[df_unido_limpio['DEPARTAMENTO'] == departamento_seleccionado]['MUNICIPIO'].unique())
    municipio_seleccionado = st.sidebar.selectbox('Selecciona el municipio', municipios)

    if municipio_seleccionado == 'Todos':
        max_accesos = df_unido_limpio[df_unido_limpio['DEPARTAMENTO'] == departamento_seleccionado]['No. ACCESOS FIJOS A INTERNET'].max()
    else:
        max_accesos = df_unido_limpio[(df_unido_limpio['DEPARTAMENTO'] == departamento_seleccionado) &
                                      (df_unido_limpio['MUNICIPIO'] == municipio_seleccionado)]['No. ACCESOS FIJOS A INTERNET'].max()

# Selección dinámica de accesos
accesos_seleccionados = st.sidebar.slider('Número mínimo de accesos', min_value=0, max_value=int(max_accesos), value=0)

# Límite de puntos a mostrar
limite_puntos = st.sidebar.slider('Número máximo de datos a mostrar', min_value=1, max_value=1000, value=100)

# Generar el heatmap con los filtros aplicados
mapa_generado = generar_heatmap(departamento_seleccionado, municipio_seleccionado, accesos_seleccionados, limite_puntos)

# Mostrar el heatmap en Streamlit con mayor tamaño y ajustes responsivos
folium_static(mapa_generado, width=1300, height=700)

# Mostrar información adicional
st.write(f"Departamento seleccionado: {departamento_seleccionado}")
st.write(f"Municipio seleccionado: {municipio_seleccionado}")
st.write(f"Número mínimo de accesos: {accesos_seleccionados}")
st.write(f"Número máximo de puntos mostrados: {limite_puntos}")
