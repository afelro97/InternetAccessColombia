#%%
import pandas as pd

#%%
# Cargar los datasets originales
df_internet = pd.read_csv('Limpieza/data/INFORMACIÓN_TRIMESTRAL_DE_ACCESOS_FIJOS_A_INTERNET_POR_PROVEEDOR,_DEPARTAMENTO,_MUNICIPIO,_SEGMENTO,_TECNOLOGIA,_Y_VELOCIDAD_DE_CONEXIÓN_COMPLETO.csv')

# Cargar las nuevas coordenadas obtenidas
df_coordenadas_actualizadas = pd.read_csv('Limpieza/data/coordenadas_unicas.csv')

#%%
# Normalizar los nombres de municipios en ambos datasets para asegurar coincidencia
df_internet['MUNICIPIO'] = df_internet['MUNICIPIO'].str.strip().str.upper()
df_coordenadas_actualizadas['MUNICIPIO'] = df_coordenadas_actualizadas['MUNICIPIO'].str.strip().str.upper()

#%%
# Unir ambos datasets usando la columna 'MUNICIPIO' como clave
df_unido = pd.merge(df_internet, df_coordenadas_actualizadas, on='MUNICIPIO', how='left')

#%%
# Resolver el problema de duplicación de columnas
# Mantener solo una columna DEPARTAMENTO
if 'DEPARTAMENTO_x' in df_unido.columns and 'DEPARTAMENTO_y' in df_unido.columns:
    # Eliminar la columna duplicada
    df_unido = df_unido.drop(columns=['DEPARTAMENTO_y'])
    df_unido.rename(columns={'DEPARTAMENTO_x': 'DEPARTAMENTO'}, inplace=True)

#%%
# Crear dataset limpio eliminando valores nulos en coordenadas
df_unificado_limpio = df_unido.dropna(subset=['Latitud', 'Longitud'])

#%%
# Validar la cantidad de datos eliminados
print(f"Total de filas antes de limpiar: {len(df_unido)}")
print(f"Total de filas después de limpiar: {len(df_unificado_limpio)}")
print(f"Porcentaje de datos eliminados: {100 * (1 - len(df_unificado_limpio) / len(df_unido)):.2f}%")

#%%
# Guardar el nuevo DataFrame limpio en un archivo CSV
df_unificado_limpio.to_csv('Limpieza/data/df_unificado_limpio.csv', index=False)

print("El nuevo archivo 'df_unificado_limpio.csv' se ha generado con éxito.")
