#%%
# Importar librerías necesarias
import pandas as pd

# Ruta del dataset original
ruta_dataset = "Limpieza/data/df_unificado_limpio_actualizado.csv"

# Cargar el dataset original
df = pd.read_csv(ruta_dataset)

# Revisar estructura del dataset
print("Estructura del dataset original:")
print(df.info())

# Verificar valores únicos de columnas clave
print("Valores únicos de DEPARTAMENTO:", df["DEPARTAMENTO"].unique())
print("Valores únicos de MUNICIPIO:", df["MUNICIPIO"].unique())
print("Valores únicos de AÑO:", df["AÑO"].unique())

# Asegurarse de que las columnas clave no tengan valores nulos
df = df.dropna(subset=["DEPARTAMENTO", "MUNICIPIO", "AÑO", "No. ACCESOS FIJOS A INTERNET"])

#%%
# Definir la tasa de crecimiento anual y los años futuros
tasa_crecimiento = 0.05
anios_futuros = [2022, 2023, 2024, 2025]

# Crear una lista para almacenar los nuevos datos
nuevos_datos = []

# Generar datos futuros basados en la tasa de crecimiento
for (departamento, municipio), group in df.groupby(["DEPARTAMENTO", "MUNICIPIO"]):
    print(f"Procesando: Departamento={departamento}, Municipio={municipio}")

    # Verificar el grupo procesado
    print(f"Datos del grupo:\n{group[['AÑO', 'No. ACCESOS FIJOS A INTERNET']]}")

    # Filtrar los datos más recientes por municipio y departamento
    ultimo_anio = group["AÑO"].max()
    datos_ultimo_anio = group[group["AÑO"] == ultimo_anio]

    if datos_ultimo_anio.empty:
        print(f"Sin datos para {departamento}-{municipio}")
        continue

    # Obtener accesos actuales
    accesos_actuales = datos_ultimo_anio["No. ACCESOS FIJOS A INTERNET"].sum()
    print(f"Accesos actuales para el año {ultimo_anio}: {accesos_actuales}")

    for anio in anios_futuros:
        if anio <= ultimo_anio:
            # Ignorar años pasados o iguales al último año
            continue

        # Calcular accesos para el año futuro
        accesos_futuros = int(accesos_actuales * (1 + tasa_crecimiento))
        print(f"Generando para {anio}: {accesos_futuros} accesos.")

        nuevos_datos.append({
            "AÑO": anio,
            "DEPARTAMENTO": departamento,
            "MUNICIPIO": municipio,
            "No. ACCESOS FIJOS A INTERNET": accesos_futuros
        })

        # Actualizar accesos actuales
        accesos_actuales = accesos_futuros

# Crear un DataFrame con los nuevos datos generados
df_futuro = pd.DataFrame(nuevos_datos)

#%%
# Validar los datos futuros generados
print("Datos futuros generados (primeros 5 registros):")
print(df_futuro.head())
print(f"Total de registros generados: {len(df_futuro)}")

# Combinar los datos originales con los futuros
df_combinado = pd.concat([df, df_futuro], ignore_index=True)

# Guardar los datos combinados en un nuevo archivo CSV
nueva_ruta_dataset = "Limpieza/data/df_unificado_limpio_con_proyecciones.csv"
df_combinado.to_csv(nueva_ruta_dataset, index=False)

print(f"Datos combinados guardados en: {nueva_ruta_dataset}")
print(f"Total de filas después de agregar proyecciones: {len(df_combinado)}")

#%%
# Revisar datos por año en el dataset combinado
print("Resumen por Año:")
print(df_combinado.groupby("AÑO")["No. ACCESOS FIJOS A INTERNET"].sum())
