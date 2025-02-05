#%%
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor
#%%
df = pd.read_csv('Limpieza/data/df_unificado_limpio_con_proyecciones_limpio.csv')
#%%
df.columns
#%%
municipios_unicos = df['MUNICIPIO'].unique()
#%%
municipios_unicos
#%%
numero_municipios = df['MUNICIPIO'].value_counts()
#%%
# Contar valores únicos en la columna MUNICIPIO
n_municipios = df['MUNICIPIO'].nunique()
#%%
n_municipios
#%%
# 1. Crear nueva columna TRIMESTRE
df['TRIMESTRE'] = ((df['AÑO'] - 2021) * 4 + df['TRIMESTRE'])
#%%
# 2. Concatenar DEPARTAMENTO y MUNICIPIO con un punto
df['UBICACION'] = df['DEPARTAMENTO'] + '.' + df['MUNICIPIO']
#%%
# 3. Seleccionar solo las columnas que necesitamos
df_final = df[[
    'TRIMESTRE',
    'UBICACION',
    'VELOCIDAD BAJADA',
    'VELOCIDAD SUBIDA',
    'No. ACCESOS FIJOS A INTERNET'
]].copy()
#%%
# Para verificar el resultado
print("Muestra de valores únicos en TRIMESTRE:")
print(df['TRIMESTRE'].unique())
#%%
print("\nPrimeras filas del dataset transformado:")
print(df_final.head())
#%%
# Exportar a CSV
#df_final.to_csv('dataset_prediccion.csv', index=False)
#%%
# Obtener lista de ubicaciones únicas
ubicaciones = df_final['UBICACION'].unique()
#%%
# Para cada ubicación, crear y guardar su subdataset
for ubicacion in ubicaciones:
    # Filtrar el dataframe para esta ubicación
    df_ubicacion = df_final[df_final['UBICACION'] == ubicacion]

    # Seleccionar solo las columnas necesarias
    df_ubicacion = df_ubicacion[[
        'TRIMESTRE',
        'VELOCIDAD BAJADA',
        'VELOCIDAD SUBIDA',
        'No. ACCESOS FIJOS A INTERNET'
    ]]

    # Crear nombre de archivo (reemplazando caracteres problemáticos)
    nombre_archivo = ubicacion.replace('/', '_').replace('\\', '_')
    ruta_archivo = f'Limpieza/data/subdatasets/{nombre_archivo}.csv'

    # Guardar el archivo
    df_ubicacion.to_csv(ruta_archivo, index=False)

print(f"Se han creado {len(ubicaciones)} archivos en Limpieza/data/subdatasets/")
#%%
# Leer un dataset específico de la carpeta de subdatasets
nombre_municipio = 'ANTIOQUIA.AMALFI'  # Ejemplo, cámbialo por el que quieras ver
ruta_archivo = f'Limpieza/data/subdatasets/{nombre_municipio}.csv'
df_municipio = pd.read_csv(ruta_archivo)
#%%

# Crear figura y ejes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Agregar un pequeño offset para separar las velocidades
offset = 0.2

# Graficar velocidades en el primer eje
markerline1, stemlines1, baseline1 = ax1.stem(df_municipio['TRIMESTRE'] - offset,
                                              df_municipio['VELOCIDAD SUBIDA'],
                                              label='Velocidad Subida',
                                              linefmt='b-',
                                              markerfmt='bo')
markerline2, stemlines2, baseline2 = ax1.stem(df_municipio['TRIMESTRE'] + offset,
                                              df_municipio['VELOCIDAD BAJADA'],
                                              label='Velocidad Bajada',
                                              linefmt='r-',
                                              markerfmt='ro')

# Graficar accesos en el segundo eje
markerline3, stemlines3, baseline3 = ax2.stem(df_municipio['TRIMESTRE'],
                                              df_municipio['No. ACCESOS FIJOS A INTERNET'],
                                              label='Accesos Fijos',
                                              linefmt='g-',
                                              markerfmt='go')

# Configurar etiquetas y título
ax1.set_xlabel('Trimestre')
ax1.set_ylabel('Velocidad (Mbps)')
ax2.set_xlabel('Trimestre')
ax2.set_ylabel('Número de Accesos')
fig.suptitle(f'Métricas para {nombre_municipio}')

# Agregar leyendas
ax1.legend(loc='upper left')
ax2.legend(loc='upper left')

# Ajustar diseño y estilo
plt.style.use('dark_background')
ax1.grid(True, alpha=0.3)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
#%%
# Preparar datos para la regresión
X = df_municipio[['TRIMESTRE']].values
y = df_municipio['VELOCIDAD BAJADA'].values
#%%
# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%%
# Crear y entrenar el modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)
#%%
# Hacer predicciones
y_pred = modelo.predict(X)
#%%
# Visualizar resultados
plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='blue', alpha=0.5, label='Datos reales')
plt.plot(X, y_pred, color='red', label='Regresión lineal')
plt.xlabel('Trimestre')
plt.ylabel('Velocidad de Bajada (Mbps)')
plt.title(f'Regresión Lineal para {nombre_municipio}\nVelocidad de Bajada vs Trimestre')
plt.legend()
plt.grid(True)
plt.show()
#%%
# Mostrar métricas
print("\nMétricas del modelo:")
print(f"Coeficiente (pendiente): {modelo.coef_[0]:.2f}")
print(f"Intercepto: {modelo.intercept_:.2f}")
print(f"R² Score: {r2_score(y, y_pred):.4f}")
print(f"Error cuadrático medio: {mean_squared_error(y, y_pred):.2f}")
print(f"Raíz del error cuadrático medio: {np.sqrt(mean_squared_error(y, y_pred)):.2f}")
#%%
# Crear y entrenar el modelo RANSAC
ransac = RANSACRegressor(random_state=42)
ransac.fit(X, y)
#%%
# Predicciones
y_pred = ransac.predict(X)
#%%
# Obtener la máscara de inliers/outliers
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
#%%
# Visualización
plt.figure(figsize=(12, 6))
plt.scatter(X[inlier_mask], y[inlier_mask], color='blue', alpha=0.5,
            label='Inliers (datos normales)')
plt.scatter(X[outlier_mask], y[outlier_mask], color='red', alpha=0.5,
            label='Outliers (datos atípicos)')
plt.plot(X, y_pred, color='green', label='Modelo RANSAC')
plt.xlabel('Trimestre')
plt.ylabel('Velocidad de Bajada (Mbps)')
plt.title(f'RANSAC Regression para {nombre_municipio}')
plt.legend()
plt.grid(True)
plt.show()
#%%
# Métricas
print("\nMétricas del modelo RANSAC:")
print(f"R² Score: {r2_score(y, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.2f}")
print(f"Número de inliers: {np.sum(inlier_mask)}")
print(f"Número de outliers: {np.sum(outlier_mask)}")
#%%
# Definir parámetros a probar con los nombres correctos
param_grid = {
    'min_samples': [10, 50, 100],
    'residual_threshold': [50, 100, 200],
    'max_trials': [50, 100, 200],
    'loss': ['absolute_error', 'squared_error']  # Nombres corregidos
}
#%%
# Crear modelo base
ransac = RANSACRegressor(random_state=42)
#%%
# Búsqueda de mejores parámetros
grid_search = GridSearchCV(ransac, param_grid, cv=5, scoring='r2')
grid_search.fit(X, y)
#%%
# Crear modelo base
ransac = RANSACRegressor(random_state=42)
#%%
# Búsqueda de mejores parámetros
grid_search = GridSearchCV(ransac, param_grid, cv=5, scoring='r2')
grid_search.fit(X, y)
#%%
print("\nMejores parámetros encontrados:")
print(grid_search.best_params_)
print(f"Mejor score: {grid_search.best_score_:.4f}")
#%%
# Entrenar modelos
huber = HuberRegressor()
rf = RandomForestRegressor(n_estimators=100, random_state=42)

huber.fit(X, y)
rf.fit(X, y)

# Predicciones
y_pred_huber = huber.predict(X)
y_pred_rf = rf.predict(X)

# Visualización
plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='blue', alpha=0.5, label='Datos reales')
plt.plot(X, y_pred_huber, color='red', label='Huber Regression')
plt.plot(X, y_pred_rf, color='green', label='Random Forest')
plt.xlabel('Trimestre')
plt.ylabel('Velocidad de Bajada (Mbps)')
plt.title(f'Comparación de Modelos para {nombre_municipio}')
plt.legend()
plt.grid(True)
plt.show()

# Métricas
print("\nMétricas de los modelos:")
print("\nHuber Regression:")
print(f"R² Score: {r2_score(y, y_pred_huber):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred_huber)):.2f}")

print("\nRandom Forest:")
print(f"R² Score: {r2_score(y, y_pred_rf):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred_rf)):.2f}")
#%%
# Calcular promedio y desviación estándar por trimestre
df_stats = df_final.groupby('TRIMESTRE').agg({
    'VELOCIDAD BAJADA': ['mean', 'std'],
    'VELOCIDAD SUBIDA': ['mean', 'std'],
    'No. ACCESOS FIJOS A INTERNET': ['mean', 'std']
}).reset_index()
#%%
# Renombrar columnas para claridad
df_stats.columns = [
    'TRIMESTRE',
    'BAJADA_MEDIA', 'BAJADA_STD',
    'SUBIDA_MEDIA', 'SUBIDA_STD',
    'ACCESOS_MEDIA', 'ACCESOS_STD'
]
#%%
df_stats.columns
#%%
# Aplicar regresión lineal al dataset promediado
X = df_stats[['TRIMESTRE']].values
y = df_stats['BAJADA_MEDIA'].values
#%%
# Crear y entrenar el modelo
modelo = LinearRegression()
modelo.fit(X, y)
y_pred = modelo.predict(X)
#%%
# Visualización
plt.figure(figsize=(12, 8))

# Subplot 1: Datos originales
plt.subplot(2, 1, 1)
plt.scatter(df['TRIMESTRE'], df['VELOCIDAD BAJADA'], color='blue', alpha=0.5, label='Datos originales')
plt.plot(df['TRIMESTRE'], modelo.predict(df[['TRIMESTRE']]), color='red', label='Regresión datos originales')
plt.xlabel('Trimestre')
plt.ylabel('Velocidad de Bajada (Mbps)')
plt.title('Datos Originales con Regresión')
plt.yscale('log')
plt.legend()
plt.grid(True)
#%%
# Subplot 2: Datos promediados
plt.subplot(2, 1, 2)
plt.errorbar(df_stats['TRIMESTRE'], df_stats['BAJADA_MEDIA'],
             yerr=df_stats['BAJADA_STD'], fmt='o', color='blue',
             label='Media ± Desv. Est.')
plt.plot(X, y_pred, color='red', label='Regresión promedios')
plt.xlabel('Trimestre')
plt.ylabel('Velocidad de Bajada (Mbps)')
plt.title('Promedios por Trimestre con Regresión')
plt.yscale('log')
plt.legend()
plt.grid(True)
#%%
# Métricas para ambos modelos
print("\nMétricas del modelo con datos originales:")
y_pred_orig = modelo.predict(df[['TRIMESTRE']])
print(f"R² Score: {r2_score(df['VELOCIDAD BAJADA'], y_pred_orig):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(df['VELOCIDAD BAJADA'], y_pred_orig)):.2f}")

print("\nMétricas del modelo con promedios:")
print(f"R² Score: {r2_score(y, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.2f}")

# Mostrar los datos procesados
print("\nDatos procesados por trimestre:")
print(df_stats)
#%%
