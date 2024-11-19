#%% Importar librerías
import pandas as pd
import googlemaps
import time
import os
from dotenv import load_dotenv
from pathlib import Path

#%% Configurar variables de entorno
dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)

API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')
if not API_KEY:
    raise ValueError("API Key no configurada correctamente. Verifica el archivo .env.")

# Inicializar cliente de Google Maps
gmaps = googlemaps.Client(key=API_KEY)

#%% Cargar el dataset original
df = pd.read_csv('Limpieza/data/df_unificado_limpio2.csv')

# Eliminar coordenadas previas si existen
if 'Latitud' in df.columns and 'Longitud' in df.columns:
    df = df.drop(columns=['Latitud', 'Longitud'])

#%% Obtener lista única de municipios y departamentos
municipios_unicos = df[['DEPARTAMENTO', 'MUNICIPIO']].drop_duplicates()

#%% Función para obtener coordenadas usando Google Maps API
def obtener_coordenadas(departamento, municipio):
    try:
        query = f"{municipio}, {departamento}, Colombia"
        geocode_result = gmaps.geocode(query)
        if geocode_result:
            location = geocode_result[0]['geometry']['location']
            return location['lat'], location['lng']
        else:
            print(f"No se encontraron coordenadas para {municipio}, {departamento}.")
            return None, None
    except Exception as e:
        print(f"Error obteniendo coordenadas para {municipio}, {departamento}: {e}")
        return None, None

#%% Obtener coordenadas para cada municipio y departamento
coordenadas = []
for index, row in municipios_unicos.iterrows():
    departamento = row['DEPARTAMENTO']
    municipio = row['MUNICIPIO']

    # Obtener coordenadas
    lat, lng = obtener_coordenadas(departamento, municipio)
    coordenadas.append({'DEPARTAMENTO': departamento, 'MUNICIPIO': municipio, 'Latitud': lat, 'Longitud': lng})

    # Guardar progreso cada 10 solicitudes
    if (index + 1) % 10 == 0:
        df_progreso = pd.DataFrame(coordenadas)
        df_progreso.to_csv('Limpieza/data/progreso_coordenadas_nuevas.csv', index=False)
        print(f"Progreso guardado: {index + 1} municipios procesados.")

    # Respetar los límites de la API
    time.sleep(1)  # Ajustar para respetar la cuota

#%% Guardar coordenadas completas
df_coordenadas_finales = pd.DataFrame(coordenadas)
df_coordenadas_finales.to_csv('Limpieza/data/coordenadas_nuevas.csv', index=False)

#%% Unir las coordenadas nuevas con el dataset original
df_completo = df.merge(df_coordenadas_finales, on=['DEPARTAMENTO', 'MUNICIPIO'], how='left')
df_completo.to_csv('Limpieza/data/df_unificado_limpio_actualizado.csv', index=False)

print("Proceso completado. Coordenadas actualizadas.")
