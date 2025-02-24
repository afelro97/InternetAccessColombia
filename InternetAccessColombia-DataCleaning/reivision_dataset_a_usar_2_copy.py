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
df = pd.read_csv('Limpieza/data/INFORMACIÓN_TRIMESTRAL_DE_ACCESOS_FIJOS_A_INTERNET_POR_PROVEEDOR,_DEPARTAMENTO,_MUNICIPIO,_SEGMENTO,_TECNOLOGIA,_Y_VELOCIDAD_DE_CONEXIÓN_COMPLETO.csv')
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
    'No. ACCESOS FIJOS A INTERNET',
    'TECNOLOGÍA'
]].copy()
#%%
df_final.columns
#%%
print("\nPrimeras filas:")
print(df_final.head())
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
        'No. ACCESOS FIJOS A INTERNET',
        'TECNOLOGÍA'
    ]]

    # Crear nombre de archivo (reemplazando caracteres problemáticos)
    nombre_archivo = ubicacion.replace('/', '_').replace('\\', '_')
    ruta_archivo = f'Limpieza/data/subdatasets-ubicacion/{nombre_archivo}.csv'

    # Guardar el archivo
    df_ubicacion.to_csv(ruta_archivo, index=False)

print(f"Se han creado {len(ubicaciones)} archivos en Limpieza/data/subdatasets-ubicacion/")
#%%
# Leer un dataset específico de la carpeta de subdatasets
nombre_municipio = 'ANTIOQUIA.AMALFI'  # Ejemplo, cámbialo por el que quieras ver
ruta_archivo = f'Limpieza/data/subdatasets-ubicacion/{nombre_municipio}.csv'
df_municipio = pd.read_csv(ruta_archivo)
#%%
def analizar_tecnologias(df):
    # Extraer ubicación del DataFrame
    ubicacion = df['UBICACION'].iloc[0] if 'UBICACION' in df.columns else 'Ubicación No Especificada'

    # Calcular estadísticas por trimestre y tecnología
    stats_por_trimestre = df.groupby(['TRIMESTRE', 'TECNOLOGÍA']).agg({
        'VELOCIDAD BAJADA': ['mean', 'std'],
        'VELOCIDAD SUBIDA': ['mean', 'std'],
        'No. ACCESOS FIJOS A INTERNET': ['mean', 'std']
    }).reset_index()

    # Renombrar columnas para claridad
    stats_por_trimestre.columns = [
        'TRIMESTRE', 'TECNOLOGÍA',
        'VELOCIDAD_BAJADA_MEAN', 'VELOCIDAD_BAJADA_STD',
        'VELOCIDAD_SUBIDA_MEAN', 'VELOCIDAD_SUBIDA_STD',
        'ACCESOS_MEAN', 'ACCESOS_STD'
    ]

    # Asegurarse de que existe el directorio de gráficas
    ruta_graficas = 'Limpieza/data/graficas'
    if not os.path.exists(ruta_graficas):
        os.makedirs(ruta_graficas)

    # Crear figura
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # Graficar accesos por tecnología
    for tecnologia in df['TECNOLOGÍA'].unique():
        datos_tech = stats_por_trimestre[stats_por_trimestre['TECNOLOGÍA'] == tecnologia]

        # Regresión lineal para accesos
        if len(datos_tech) > 1:  # Verificar que hay suficientes datos
            modelo = LinearRegression()
            X = datos_tech['TRIMESTRE'].values.reshape(-1, 1)
            y = datos_tech['ACCESOS_MEAN'].values
            modelo.fit(X, y)
            y_pred = modelo.predict(X)

            # Graficar datos y regresión
            ax1.errorbar(datos_tech['TRIMESTRE'], datos_tech['ACCESOS_MEAN'],
                         yerr=datos_tech['ACCESOS_STD'], fmt='o-', label=f'{tecnologia}')
            ax1.plot(X, y_pred, '--', alpha=0.5)

    ax1.set_title(f'Accesos por Tecnología en {ubicacion}')
    ax1.set_xlabel('Trimestre')
    ax1.set_ylabel('Número de Accesos (Media ± Std)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True)
    ax1.set_yscale('log')

    # Graficar velocidades por tecnología
    for tecnologia in df['TECNOLOGÍA'].unique():
        datos_tech = stats_por_trimestre[stats_por_trimestre['TECNOLOGÍA'] == tecnologia]

        # Regresión lineal para velocidades
        if len(datos_tech) > 1:  # Verificar que hay suficientes datos
            modelo = LinearRegression()
            X = datos_tech['TRIMESTRE'].values.reshape(-1, 1)
            y = datos_tech['VELOCIDAD_BAJADA_MEAN'].values
            modelo.fit(X, y)
            y_pred = modelo.predict(X)

            # Graficar datos y regresión
            ax2.errorbar(datos_tech['TRIMESTRE'], datos_tech['VELOCIDAD_BAJADA_MEAN'],
                         yerr=datos_tech['VELOCIDAD_BAJADA_STD'], fmt='o-',
                         label=f'{tecnologia} - Bajada')
            ax2.plot(X, y_pred, '--', alpha=0.5)

    ax2.set_title(f'Velocidades por Tecnología en {ubicacion}')
    ax2.set_xlabel('Trimestre')
    ax2.set_ylabel('Velocidad (Mbps) (Media ± Std)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True)
    ax2.set_yscale('log')

    plt.tight_layout()

    # Guardar la figura
    ruta_grafica = f'{ruta_graficas}/{ubicacion.replace(".", "_").replace(" ", "_")}_tecnologias.png'
    plt.savefig(ruta_grafica, bbox_inches='tight', dpi=300)
    plt.close()

    # Calcular tendencias para cada tecnología
    tendencias_tecnologia = {}
    for tecnologia in df['TECNOLOGÍA'].unique():
        datos_tech = stats_por_trimestre[stats_por_trimestre['TECNOLOGÍA'] == tecnologia]
        if len(datos_tech) > 1:
            # Modelo para velocidad
            modelo_velocidad = LinearRegression()
            X = datos_tech['TRIMESTRE'].values.reshape(-1, 1)
            y = datos_tech['VELOCIDAD_BAJADA_MEAN'].values
            modelo_velocidad.fit(X, y)

            # Modelo para accesos
            modelo_accesos = LinearRegression()
            y_accesos = datos_tech['ACCESOS_MEAN'].values
            modelo_accesos.fit(X, y_accesos)

            tendencias_tecnologia[tecnologia] = {
                'pendiente_velocidad': modelo_velocidad.coef_[0],
                'r2_velocidad': r2_score(y, modelo_velocidad.predict(X)),
                'pendiente_accesos': modelo_accesos.coef_[0],
                'r2_accesos': r2_score(y_accesos, modelo_accesos.predict(X)),
                'velocidad_media': datos_tech['VELOCIDAD_BAJADA_MEAN'].mean(),
                'velocidad_std': datos_tech['VELOCIDAD_BAJADA_STD'].mean(),
                'accesos_media': datos_tech['ACCESOS_MEAN'].mean(),
                'accesos_std': datos_tech['ACCESOS_STD'].mean()
            }

    return stats_por_trimestre, ruta_grafica, tendencias_tecnologia


# Imprimir estadísticas y tendencias
for tecnologia in df_municipio['TECNOLOGÍA'].unique():
    datos_tech = stats_tecnologia[stats_tecnologia['TECNOLOGÍA'] == tecnologia]

    # Calcular tendencias
    if len(datos_tech) > 1:
        modelo_velocidad = LinearRegression()
        X = datos_tech['TRIMESTRE'].values.reshape(-1, 1)
        y = datos_tech['VELOCIDAD_BAJADA_MEAN'].values
        modelo_velocidad.fit(X, y)

        pendiente_velocidad = modelo_velocidad.coef_[0]
        r2_velocidad = r2_score(y, modelo_velocidad.predict(X))

        print(f"\n{tecnologia}:")
        print(f"Tendencia de velocidad: {pendiente_velocidad:.2f} Mbps/trimestre (R² = {r2_velocidad:.2f})")
        print(f"Velocidad promedio: {datos_tech['VELOCIDAD_BAJADA_MEAN'].mean():.2f} ± {datos_tech['VELOCIDAD_BAJADA_STD'].mean():.2f} Mbps")
        print(f"Accesos promedio: {datos_tech['ACCESOS_MEAN'].mean():.2f} ± {datos_tech['ACCESOS_STD'].mean():.2f}")
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
#%% md
# * Muestra todos los datos crudos (puntos azules) de velocidad de bajada
# * La escala es logarítmica (note los valores 10^-2 hasta 10^6)
# * Los puntos verticales azules representan todos los valores registrados en cada trimestre
# * La línea roja es la regresión lineal sobre estos datos originales
# * Se puede ver mucha dispersión en los datos (valores muy altos y muy bajos)
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
#%% md
# * Muestra el promedio de los datos por trimestre (puntos azules)
# * Las barras verticales azules representan la desviación estándar (variabilidad de los datos)
# * La línea roja es la regresión lineal sobre estos promedios
# * También usa escala logarítmica pero con menos rango (10^2 hasta 10^4)
# * Se ve una tendencia más clara al trabajar con promedios
# * Las barras de error (líneas verticales) muestran qué tan dispersos están los datos alrededor de la media
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
import warnings
warnings.filterwarnings('ignore')
#%%

def analizar_tecnologias(df):
    # Extraer ubicación del DataFrame
    ubicacion = df['UBICACION'].iloc[0] if 'UBICACION' in df.columns else 'Ubicación No Especificada'

    # Calcular estadísticas por trimestre y tecnología
    stats_por_trimestre = df.groupby(['TRIMESTRE', 'TECNOLOGÍA']).agg({
        'VELOCIDAD BAJADA': ['mean', 'std'],
        'VELOCIDAD SUBIDA': ['mean', 'std'],
        'No. ACCESOS FIJOS A INTERNET': ['mean', 'std']
    }).reset_index()

    # Renombrar columnas para claridad
    stats_por_trimestre.columns = [
        'TRIMESTRE', 'TECNOLOGÍA',
        'VELOCIDAD_BAJADA_MEAN', 'VELOCIDAD_BAJADA_STD',
        'VELOCIDAD_SUBIDA_MEAN', 'VELOCIDAD_SUBIDA_STD',
        'ACCESOS_MEAN', 'ACCESOS_STD'
    ]

    # Asegurarse de que existe el directorio de gráficas
    ruta_graficas = 'Limpieza/data/graficas'
    if not os.path.exists(ruta_graficas):
        os.makedirs(ruta_graficas)

    # Crear figura
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # Graficar accesos por tecnología
    for tecnologia in df['TECNOLOGÍA'].unique():
        datos_tech = stats_por_trimestre[stats_por_trimestre['TECNOLOGÍA'] == tecnologia]

        # Regresión lineal para accesos
        if len(datos_tech) > 1:  # Verificar que hay suficientes datos
            modelo = LinearRegression()
            X = datos_tech['TRIMESTRE'].values.reshape(-1, 1)
            y = datos_tech['ACCESOS_MEAN'].values
            modelo.fit(X, y)
            y_pred = modelo.predict(X)

            # Graficar datos y regresión
            ax1.errorbar(datos_tech['TRIMESTRE'], datos_tech['ACCESOS_MEAN'],
                         yerr=datos_tech['ACCESOS_STD'], fmt='o-', label=f'{tecnologia}')
            ax1.plot(X, y_pred, '--', alpha=0.5)

    ax1.set_title(f'Accesos por Tecnología en {ubicacion}')
    ax1.set_xlabel('Trimestre')
    ax1.set_ylabel('Número de Accesos (Media ± Std)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True)
    ax1.set_yscale('log')

    # Graficar velocidades por tecnología
    for tecnologia in df['TECNOLOGÍA'].unique():
        datos_tech = stats_por_trimestre[stats_por_trimestre['TECNOLOGÍA'] == tecnologia]

        # Regresión lineal para velocidades
        if len(datos_tech) > 1:  # Verificar que hay suficientes datos
            modelo = LinearRegression()
            X = datos_tech['TRIMESTRE'].values.reshape(-1, 1)
            y = datos_tech['VELOCIDAD_BAJADA_MEAN'].values
            modelo.fit(X, y)
            y_pred = modelo.predict(X)

            # Graficar datos y regresión
            ax2.errorbar(datos_tech['TRIMESTRE'], datos_tech['VELOCIDAD_BAJADA_MEAN'],
                         yerr=datos_tech['VELOCIDAD_BAJADA_STD'], fmt='o-',
                         label=f'{tecnologia} - Bajada')
            ax2.plot(X, y_pred, '--', alpha=0.5)

    ax2.set_title(f'Velocidades por Tecnología en {ubicacion}')
    ax2.set_xlabel('Trimestre')
    ax2.set_ylabel('Velocidad (Mbps) (Media ± Std)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True)
    ax2.set_yscale('log')

    plt.tight_layout()

    # Guardar la figura
    ruta_grafica = f'{ruta_graficas}/{ubicacion.replace(".", "_").replace(" ", "_")}_tecnologias.png'
    plt.savefig(ruta_grafica, bbox_inches='tight', dpi=300)
    plt.close()

    # Calcular tendencias para cada tecnología
    tendencias_tecnologia = {}
    for tecnologia in df['TECNOLOGÍA'].unique():
        datos_tech = stats_por_trimestre[stats_por_trimestre['TECNOLOGÍA'] == tecnologia]
        if len(datos_tech) > 1:
            # Modelo para velocidad
            modelo_velocidad = LinearRegression()
            X = datos_tech['TRIMESTRE'].values.reshape(-1, 1)
            y = datos_tech['VELOCIDAD_BAJADA_MEAN'].values
            modelo_velocidad.fit(X, y)

            # Modelo para accesos
            modelo_accesos = LinearRegression()
            y_accesos = datos_tech['ACCESOS_MEAN'].values
            modelo_accesos.fit(X, y_accesos)

            tendencias_tecnologia[tecnologia] = {
                'pendiente_velocidad': modelo_velocidad.coef_[0],
                'r2_velocidad': r2_score(y, modelo_velocidad.predict(X)),
                'pendiente_accesos': modelo_accesos.coef_[0],
                'r2_accesos': r2_score(y_accesos, modelo_accesos.predict(X)),
                'velocidad_media': datos_tech['VELOCIDAD_BAJADA_MEAN'].mean(),
                'velocidad_std': datos_tech['VELOCIDAD_BAJADA_STD'].mean(),
                'accesos_media': datos_tech['ACCESOS_MEAN'].mean(),
                'accesos_std': datos_tech['ACCESOS_STD'].mean()
            }

    return stats_por_trimestre, ruta_grafica, tendencias_tecnologia

def analizar_municipio(ruta_archivo):
    # Leer dataset
    df = pd.read_csv(ruta_archivo)
    nombre_municipio = ruta_archivo.split('/')[-1].replace('.csv', '')
    df['UBICACION'] = nombre_municipio

    # Asegurarnos que TRIMESTRE sea entero
    df['TRIMESTRE'] = df['TRIMESTRE'].astype(int)

    # Calcular estadísticas por trimestre
    df_stats = df.groupby('TRIMESTRE').agg({
        'VELOCIDAD BAJADA': ['mean', 'std'],
        'VELOCIDAD SUBIDA': ['mean', 'std'],
        'No. ACCESOS FIJOS A INTERNET': ['mean', 'std']
    }).reset_index()

    # Renombrar columnas
    df_stats.columns = [
        'TRIMESTRE',
        'BAJADA_MEDIA', 'BAJADA_STD',
        'SUBIDA_MEDIA', 'SUBIDA_STD',
        'ACCESOS_MEDIA', 'ACCESOS_STD'
    ]

    # Crear modelos de regresión para cada variable
    X = df_stats[['TRIMESTRE']].values

    # Modelo para velocidad de bajada
    modelo_bajada = LinearRegression()
    modelo_bajada.fit(X, df_stats['BAJADA_MEDIA'])

    # Modelo para velocidad de subida
    modelo_subida = LinearRegression()
    modelo_subida.fit(X, df_stats['SUBIDA_MEDIA'])

    # Modelo para accesos
    modelo_accesos = LinearRegression()
    modelo_accesos.fit(X, df_stats['ACCESOS_MEDIA'])

    # Generar proyecciones para 3 años (12 trimestres)
    ultimo_trimestre = int(df_stats['TRIMESTRE'].max())
    trimestres_futuros = np.arange(ultimo_trimestre + 1, ultimo_trimestre + 13).reshape(-1, 1)

    proyecciones_bajada = modelo_bajada.predict(trimestres_futuros)
    proyecciones_subida = modelo_subida.predict(trimestres_futuros)
    proyecciones_accesos = modelo_accesos.predict(trimestres_futuros)

    # Calcular métricas para cada modelo
    metricas = {
        'bajada': {
            'R2': r2_score(df_stats['BAJADA_MEDIA'], modelo_bajada.predict(X)),
            'RMSE': np.sqrt(mean_squared_error(df_stats['BAJADA_MEDIA'], modelo_bajada.predict(X))),
            'pendiente': modelo_bajada.coef_[0],
            'intercepto': modelo_bajada.intercept_
        },
        'subida': {
            'R2': r2_score(df_stats['SUBIDA_MEDIA'], modelo_subida.predict(X)),
            'RMSE': np.sqrt(mean_squared_error(df_stats['SUBIDA_MEDIA'], modelo_subida.predict(X))),
            'pendiente': modelo_subida.coef_[0],
            'intercepto': modelo_subida.intercept_
        },
        'accesos': {
            'R2': r2_score(df_stats['ACCESOS_MEDIA'], modelo_accesos.predict(X)),
            'RMSE': np.sqrt(mean_squared_error(df_stats['ACCESOS_MEDIA'], modelo_accesos.predict(X))),
            'pendiente': modelo_accesos.coef_[0],
            'intercepto': modelo_accesos.intercept_
        }
    }

    # Análisis por tecnología
    stats_tecnologia, ruta_grafica, tendencias_tecnologia = analizar_tecnologias(df)

    return {
        'municipio': nombre_municipio,
        'metricas': metricas,
        'estadisticas': df_stats.to_dict('records'),
        'proyecciones': list(zip(
            trimestres_futuros.flatten(),
            proyecciones_bajada,
            proyecciones_subida,
            proyecciones_accesos
        )),
        'tendencias_tecnologia': tendencias_tecnologia,
        'ruta_grafica': ruta_grafica
    }

# Inicializar el archivo Markdown
with open('resultados_analisis.md', 'w', encoding='utf-8') as f:
    f.write('# Resultados del Análisis por Municipio\n\n')

# Inicializar contadores
total_archivos = 0
archivos_exitosos = 0
archivos_con_error = 0

directorio = 'Limpieza/data/subdatasets-ubicacion/'

# Contar total de archivos a procesar
total_a_procesar = len([f for f in os.listdir(directorio) if f.endswith('.csv')])
print(f"Total de archivos a procesar: {total_a_procesar}")

for archivo in os.listdir(directorio):
    if archivo.endswith('.csv'):
        total_archivos += 1
        try:
            # Analizar el municipio
            resultado = analizar_municipio(os.path.join(directorio, archivo))
            archivos_exitosos += 1
            print(f"Procesado exitosamente: {archivo} ({total_archivos}/{total_a_procesar})")

            # Escribir resultados en el archivo Markdown
            with open('resultados_analisis.md', 'a', encoding='utf-8') as f:
                f.write(f"## {resultado['municipio']}\n\n")

                # Métricas generales
                for variable in ['bajada', 'subida', 'accesos']:
                    f.write(f"### Modelo de {variable.title()}\n")
                    f.write(f"- Pendiente: {resultado['metricas'][variable]['pendiente']:.4f}\n")
                    f.write(f"- Intercepto: {resultado['metricas'][variable]['intercepto']:.4f}\n")
                    f.write(f"- R² Score: {resultado['metricas'][variable]['R2']:.4f}\n")
                    f.write(f"- RMSE: {resultado['metricas'][variable]['RMSE']:.2f}\n\n")

                # Análisis por tecnología
                f.write("### Análisis por Tecnología\n\n")
                for tecnologia, stats in resultado['tendencias_tecnologia'].items():
                    f.write(f"#### {tecnologia}\n")
                    f.write(f"- Tendencia de velocidad: {stats['pendiente_velocidad']:.2f} Mbps/trimestre (R² = {stats['r2_velocidad']:.4f})\n")
                    f.write(f"- Tendencia de accesos: {stats['pendiente_accesos']:.2f} accesos/trimestre (R² = {stats['r2_accesos']:.4f})\n")
                    f.write(f"- Velocidad promedio: {stats['velocidad_media']:.2f} ± {stats['velocidad_std']:.2f} Mbps\n")
                    f.write(f"- Accesos promedio: {stats['accesos_media']:.2f} ± {stats['accesos_std']:.2f}\n\n")

                # Gráficas
                if 'ruta_grafica' in resultado:
                    ruta_relativa = os.path.relpath(resultado['ruta_grafica'],
                                                    start=os.path.dirname('resultados_analisis.md'))
                    f.write("### Gráficas de Evolución por Tecnología\n\n")
                    f.write(f"![Gráficas de evolución por tecnología para {resultado['municipio']}]({ruta_relativa})\n\n")
                    f.write("**Figura 1:** Evolución temporal de accesos y velocidades por tecnología.\n")
                    f.write("- Panel superior: Número de accesos por tecnología (escala logarítmica)\n")
                    f.write("- Panel inferior: Velocidad de bajada por tecnología (escala logarítmica)\n")
                    f.write("- Las líneas punteadas muestran la tendencia lineal\n")
                    f.write("- Las barras de error indican la desviación estándar\n\n")

                f.write("\n---\n\n")

        except Exception as e:
            archivos_con_error += 1
            print(f"Error procesando {archivo}: {str(e)}")
            # Registrar error en log
            with open('errores_analisis.log', 'a') as log:
                log.write(f"Error en {archivo}: {str(e)}\n")

print("\nResumen del procesamiento:")
print(f"Total de archivos procesados: {total_archivos}")
print(f"Procesados exitosamente: {archivos_exitosos}")
print(f"Errores encontrados: {archivos_con_error}")