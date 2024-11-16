#%%
import pandas as pd
import googlemaps
import time
import os
from dotenv import load_dotenv
from pathlib import Path

#%%
# Cargar las variables de entorno desde el archivo .env
dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)

#%%
# Obtener la API Key de las variables de entorno
API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')

# Verificar si la API Key está cargada correctamente
if not API_KEY:
    raise ValueError("La API Key de Google Maps no está configurada correctamente. Verifica el archivo .env.")

# Inicializar cliente de Google Maps con la API Key
gmaps = googlemaps.Client(key=API_KEY)

#%%
# Cargar municipios sin coordenadas
df_sin_coordenadas = pd.read_csv('Limpieza/data/municipios_sin_coordenadas.csv')

# Filtrar solo las columnas DEPARTAMENTO y MUNICIPIO
df_sin_coordenadas = df_sin_coordenadas[['DEPARTAMENTO', 'MUNICIPIO']].drop_duplicates()

#%%
# Función para obtener coordenadas
def obtener_coordenadas(departamento, municipio):
    try:
        query = f"{municipio}, {departamento}, Colombia"
        geocode_result = gmaps.geocode(query)
        if geocode_result:
            location = geocode_result[0]['geometry']['location']
            return location['lat'], location['lng']
        else:
            return None, None
    except Exception as e:
        print(f"Error obteniendo coordenadas para {municipio}, {departamento}: {e}")
        return None, None

#%%
# Lista para guardar coordenadas
coordenadas = []

# Obtener coordenadas para municipios únicos
for index, row in df_sin_coordenadas.iterrows():
    municipio = row['MUNICIPIO']
    departamento = row['DEPARTAMENTO']

    # Saltar si ya se tiene la coordenada (en caso de reanudar)
    if municipio in [item['MUNICIPIO'] for item in coordenadas]:
        continue

    # Obtener coordenadas para el municipio actual
    lat, lng = obtener_coordenadas(departamento, municipio)
    coordenadas.append({'DEPARTAMENTO': departamento, 'MUNICIPIO': municipio, 'Latitud': lat, 'Longitud': lng})

    # Guardar progreso cada 10 municipios (o cualquier número configurable)
    if len(coordenadas) % 10 == 0:
        df_progreso = pd.DataFrame(coordenadas)
        df_progreso.to_csv('Limpieza/data/progreso_coordenadas.csv', index=False)
        print(f"Progreso guardado: {len(coordenadas)} municipios procesados.")

    # Respetar los límites de la API
    time.sleep(1)

#%%
# Guardar el archivo final con todas las coordenadas obtenidas
df_coordenadas_finales = pd.DataFrame(coordenadas)
df_coordenadas_finales.to_csv('Limpieza/data/coordenadas_unicas.csv', index=False)

#%%
# Unir las coordenadas únicas al dataset original
df_sin_coordenadas_completas = df_sin_coordenadas.merge(df_coordenadas_finales, on=['DEPARTAMENTO', 'MUNICIPIO'], how='left')
df_sin_coordenadas_completas.to_csv('Limpieza/data/municipios_sin_coordenadas_completas.csv', index=False)

print("¡Proceso completo! Coordenadas obtenidas y guardadas exitosamente.")
