#%%
import pandas as pd

#%%
# Cargar los dos datasets
df_internet = pd.read_csv('Limpieza/data/INFORMACIÓN_TRIMESTRAL_DE_ACCESOS_FIJOS_A_INTERNET_POR_PROVEEDOR,_DEPARTAMENTO,_MUNICIPIO,_SEGMENTO,_TECNOLOGIA,_Y_VELOCIDAD_DE_CONEXIÓN_COMPLETO.csv')
df_coordenadas = pd.read_csv('Limpieza/data/DatasetCoordenadas.csv')

#%%
# Renombrar la columna 'Ciudad' a 'MUNICIPIO' en el dataset de coordenadas
df_coordenadas.rename(columns={'Ciudad': 'MUNICIPIO'}, inplace=True)

#%%
# Normalizar los nombres de municipios en ambos datasets
df_internet['MUNICIPIO'] = df_internet['MUNICIPIO'].str.strip().str.upper()
df_coordenadas['MUNICIPIO'] = df_coordenadas['MUNICIPIO'].str.strip().str.upper()

#%%
# Unir ambos datasets usando la columna 'MUNICIPIO' como clave
df_unido = pd.merge(df_internet, df_coordenadas, on='MUNICIPIO', how='left')

#%%
# Identificar municipios sin coordenadas
municipios_sin_coordenadas = df_unido[df_unido['Latitud'].isnull()]['MUNICIPIO'].unique()
print("Municipios sin coordenadas:", municipios_sin_coordenadas)

#%%
# Guardar los municipios sin coordenadas para posible corrección manual o uso de API
df_sin_coordenadas = df_unido[df_unido['Latitud'].isnull()]
df_sin_coordenadas.to_csv('Limpieza/data/municipios_sin_coordenadas.csv', index=False)

#%%
# Crear dataset limpio eliminando valores nulos en coordenadas
df_unificado_limpio = df_unido.dropna(subset=['Latitud', 'Longitud'])

#%%
# Validar la cantidad de datos eliminados
print(f"Total de filas antes de limpiar: {len(df_unido)}")
print(f"Total de filas después de limpiar: {len(df_unificado_limpio)}")
print(f"Porcentaje de datos eliminados: {100 * (1 - len(df_unificado_limpio) / len(df_unido)):.2f}%")

#%%
# Guardar el DataFrame limpio en un archivo CSV
df_unificado_limpio.to_csv('Limpieza/data/df_unificado_limpio.csv', index=False)

#%%
# Opcional: Guardar los datos incompletos para análisis futuro
df_incompleto = df_unido[df_unido['Latitud'].isnull() | df_unido['Longitud'].isnull()]
df_incompleto.to_csv('Limpieza/data/datos_incompletos.csv', index=False)

#%%
# Mostrar las columnas del dataset limpio para validación
df_unificado_limpio.columns
