#%% Importar librerías necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#%% Ruta del nuevo dataset mejorado
ruta_dataset_mejorado = "Limpieza/data/df_mejorado.csv"

#%% Cargar el dataset mejorado
df = pd.read_csv(ruta_dataset_mejorado)

#%% Revisar estructura del dataset
print("Estructura del dataset:")
print(df.info())

#%% Filtrar datos para garantizar que VELOCIDAD BAJADA y VELOCIDAD SUBIDA sean mayores a 0
df = df[(df['VELOCIDAD BAJADA'] > 0) & (df['VELOCIDAD SUBIDA'] > 0)]

#%% Revisar la forma del dataset después del filtrado
print(f"Total de registros después del filtrado: {len(df)}")

#%% Separar variables predictoras y objetivo
columnas_predictoras = ['AÑO', 'TRIMESTRE', 'VELOCIDAD BAJADA', 'VELOCIDAD SUBIDA', 'Latitud', 'Longitud', 'Tasa_Crecimiento', 'Densidad_Accesos', 'Promedio_Movil', 'Indice_Velocidad']
X = df[columnas_predictoras]
y = df['No. ACCESOS FIJOS A INTERNET']

#%% Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% Revisar la forma de los conjuntos
print(f"Forma de X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Forma de X_test: {X_test.shape}, y_test: {y_test.shape}")

#%% Entrenar el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X_train, y_train)

#%% Hacer predicciones y evaluar el modelo
y_pred = modelo.predict(X_test)

#%% Calcular el error cuadrático medio (MSE) y el coeficiente R²
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#%% Imprimir los resultados
print(f"Error cuadrático medio (MSE): {mse}")
print(f"Coeficiente de determinación (R²): {r2}")

#%% Imprimir los coeficientes del modelo
print("Coeficientes del modelo:")
for feature, coef in zip(columnas_predictoras, modelo.coef_):
    print(f"{feature}: {coef}")

#%% Generar un reporte de las conclusiones
# Conclusiones del Modelo de Regresión Lineal con Ingeniería de Características
print("\n# Conclusiones del Modelo de Regresión Lineal")
print("- **Error Cuadrático Medio (MSE):** {:.2f}".format(mse))
print("- **Coeficiente de Determinación (R²):** {:.4f}".format(r2))
print("- **Coeficientes de las Variables Predictoras:**")
for feature, coef in zip(columnas_predictoras, modelo.coef_):
    print(f"  - {feature}: {coef:.4f}")
