import pandas as pd

# Ruta de los datasets
ruta_base = "Limpieza/data/df_unificado_limpio_actualizado.csv"
ruta_generado = "Limpieza/data/df_unificado_limpio_con_proyecciones.csv"

# Leer los datasets
df_base = pd.read_csv(ruta_base)
df_generado = pd.read_csv(ruta_generado)

# Mostrar la forma (filas, columnas) de cada dataset
print("Forma del dataset base:", df_base.shape)
print("Forma del dataset generado:", df_generado.shape)

# Comparar columnas
print("\nColumnas en el dataset base:", df_base.columns.tolist())
print("Columnas en el dataset generado:", df_generado.columns.tolist())

# Descripción estadística de ambos datasets
print("\nDescripción estadística del dataset base:")
print(df_base.describe())

print("\nDescripción estadística del dataset generado:")
print(df_generado.describe())

# Comparar valores únicos por columna
print("\nValores únicos por columna en el dataset base:")
print(df_base.nunique())

print("\nValores únicos por columna en el dataset generado:")
print(df_generado.nunique())

# Comparar años en ambos datasets
if 'AÑO' in df_base.columns and 'AÑO' in df_generado.columns:
    print("\nAños en el dataset base:", df_base['AÑO'].unique())
    print("Años en el dataset generado:", df_generado['AÑO'].unique())
else:
    print("\nLa columna 'AÑO' no está presente en uno o ambos datasets.")
