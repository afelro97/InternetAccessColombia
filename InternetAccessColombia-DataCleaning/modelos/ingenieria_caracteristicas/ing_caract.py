#%%
# Importar librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#%%
# Ruta del dataset imputado
ruta_dataset = "../../Limpieza/data/df_unificado_limpio_imputado.csv"

#%%
# Cargar el dataset
df = pd.read_csv(ruta_dataset)

#%%
# Generar tasas de crecimiento de accesos por municipio y departamento
df.sort_values(by=["DEPARTAMENTO", "MUNICIPIO", "AÑO"], inplace=True)
df["Tasa_Crecimiento"] = df.groupby(["DEPARTAMENTO", "MUNICIPIO"])["No. ACCESOS FIJOS A INTERNET"].pct_change()
df["Tasa_Crecimiento"].fillna(0, inplace=True)  # Reemplazar NaN inicial por 0

#%%
# Generar densidad de accesos por área geográfica
df["Densidad_Accesos"] = df["No. ACCESOS FIJOS A INTERNET"] / (df["Latitud"].abs() + df["Longitud"].abs())
df["Densidad_Accesos"].replace([np.inf, -np.inf], np.nan, inplace=True)  # Reemplazar valores infinitos
df["Densidad_Accesos"].fillna(0, inplace=True)  # Reemplazar NaN con 0

#%%
# Crear promedios móviles para accesos
df["Promedio_Movil"] = df.groupby(["DEPARTAMENTO", "MUNICIPIO"])["No. ACCESOS FIJOS A INTERNET"].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

#%%
# Crear índice compuesto para velocidades de bajada y subida
df["Indice_Velocidad"] = (df["VELOCIDAD BAJADA"] + df["VELOCIDAD SUBIDA"]) / 2

#%%
# Codificar variables categóricas (One-Hot Encoding)
df = pd.get_dummies(df, columns=["SEGMENTO", "TECNOLOGÍA"], drop_first=True)

#%%
# Normalizar variables continuas
columnas_a_normalizar = ["Latitud", "Longitud", "VELOCIDAD BAJADA", "VELOCIDAD SUBIDA", "Densidad_Accesos", "Indice_Velocidad"]
scaler = StandardScaler()
df[columnas_a_normalizar] = scaler.fit_transform(df[columnas_a_normalizar])

#%%
# Guardar dataset con nuevas características
ruta_dataset_mejorado = "../../Limpieza/data/df_mejorado.csv"
df.to_csv(ruta_dataset_mejorado, index=False)

#%%
# Cargar el dataset mejorado
df_mejorado = pd.read_csv(ruta_dataset_mejorado)

#%%
# Revisar estructura del dataset mejorado
print("Estructura del dataset mejorado:")
print(df_mejorado.info())

#%%
# Revisar las primeras filas
print("\nPrimeras filas del dataset mejorado:")
print(df_mejorado.head())

#%%
# Verificar valores nulos
nulos_por_columna = df_mejorado.isnull().sum()
print("\nValores nulos por columna:")
print(nulos_por_columna)

#%%
# Descripción estadística de las nuevas características
nuevas_columnas = ["Tasa_Crecimiento", "Densidad_Accesos", "Promedio_Movil", "Indice_Velocidad"]
print("\nDescripción estadística de las nuevas características:")
print(df_mejorado[nuevas_columnas].describe())

#%%
# Filtrar valores infinitos o NaN en la columna "Tasa_Crecimiento"
tasa_crecimiento_valida = df_mejorado["Tasa_Crecimiento"][np.isfinite(df_mejorado["Tasa_Crecimiento"])]
#%%
# Distribución de la densidad de accesos
plt.hist(df_mejorado["Densidad_Accesos"], bins=50, alpha=0.7, color='green')
plt.title("Distribución de la Densidad de Accesos")
plt.xlabel("Densidad de Accesos")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.show()

#%%
# Reemplazar valores infinitos y NaN en la columna "Tasa_Crecimiento"
df_mejorado["Tasa_Crecimiento"] = df_mejorado["Tasa_Crecimiento"].replace([np.inf, -np.inf], np.nan).dropna()

# Filtrar valores válidos
tasa_crecimiento_valida = df_mejorado["Tasa_Crecimiento"]

# Distribución de la tasa de crecimiento
plt.hist(tasa_crecimiento_valida, bins=50, alpha=0.7, color='blue')
plt.title("Distribución de la Tasa de Crecimiento")
plt.xlabel("Tasa de Crecimiento")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.show()
#%%
