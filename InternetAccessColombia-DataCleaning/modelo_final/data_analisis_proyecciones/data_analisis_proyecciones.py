#%%
import pandas as pd
import os
import numpy as np
from datetime import datetime
import logging
from tqdm import tqdm
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


#%%
# Set up logging
logging.basicConfig(
    filename=f'analysis_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

#%%
def setup_directories():
    """Create necessary directories for output"""
    directories = [
        'reports',
        'reports/trends',
        'reports/graphs',
        'reports/summaries'
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f'Created directory: {directory}')

#%%
def process_municipality(file_path, debug=False):
    """
    Process a single municipality file and generate its analysis

    Args:
        file_path (str): Path to the municipality CSV file
        debug (bool): Whether to print debug information

    Returns:
        tuple: (municipality_name, stats, graph_path, trends)
    """
    try:
        if debug:
            print(f"Processing {file_path}")

        # Extract municipality name from file path
        municipality_name = os.path.basename(file_path).replace('.csv', '')

        # Read the data
        df = pd.read_csv(file_path)
        if debug:
            print(f"Loaded data shape: {df.shape}")
            print("Columns:", df.columns.tolist())

        # Add UBICACION column if not present
        if 'UBICACION' not in df.columns:
            df['UBICACION'] = municipality_name

        # Run analysis
        stats, graph_path, trends = analizar_tecnologias(df)

        if debug:
            print(f"Analysis completed for {municipality_name}")
            print(f"Number of technologies analyzed: {len(trends)}")

        return municipality_name, stats, graph_path, trends

    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        if debug:
            print(f"Error processing {file_path}: {str(e)}")
        return None

#%%
def trends_to_dataframe(trends, municipality_name):
    """Convert trends dictionary to DataFrame format"""
    records = []

    for tech, stats in trends.items():
        record = {
            'municipality': municipality_name,
            'technology': tech,
            'velocity_mean': stats['velocidad_media'],
            'velocity_std': stats['velocidad_std'],
            'access_mean': stats['accesos_media'],
            'access_std': stats['accesos_std'],
            'velocity_trend': stats['pendiente_velocidad'],
            'velocity_r2': stats['r2_velocidad'],
            'access_trend': stats['pendiente_accesos'],
            'access_r2': stats['r2_accesos']
        }
        records.append(record)

    return pd.DataFrame(records)

#%%
def generate_summary(results):
    """Generate a summary DataFrame of all municipalities"""
    summary_records = []

    for mun_name, _, _, trends in results:
        if trends:  # If trends were successfully calculated
            # Process each technology
            for tech, stats in trends.items():
                record = {
                    'municipality': mun_name,
                    'technology': tech,
                    'mean_velocity': stats['velocidad_media'],
                    'std_velocity': stats['velocidad_std'],
                    'mean_access': stats['accesos_media'],
                    'std_access': stats['accesos_std'],
                    'velocity_growth': stats['pendiente_velocidad'],
                    'access_growth': stats['pendiente_accesos'],
                    'velocity_r2': stats['r2_velocidad'],
                    'access_r2': stats['r2_accesos']
                }
                summary_records.append(record)

    summary_df = pd.DataFrame(summary_records)

    # Add rankings
    summary_df['velocity_rank'] = summary_df.groupby('technology')['mean_velocity'].rank(ascending=False)
    summary_df['access_rank'] = summary_df.groupby('technology')['mean_access'].rank(ascending=False)

    return summary_df

#%%
def process_and_save_results(results, debug=False):
    """Process results and save to CSV files"""

    # Create main summary DataFrame
    summary_df = generate_summary(results)

    # Save main summary
    summary_df.to_csv('reports/summaries/complete_analysis.csv', index=False)

    # Create technology summary
    tech_summary = summary_df.groupby('technology').agg({
        'mean_velocity': ['mean', 'std', 'min', 'max'],
        'mean_access': ['mean', 'std', 'min', 'max'],
        'velocity_growth': 'mean',
        'access_growth': 'mean'
    }).round(2)

    tech_summary.to_csv('reports/summaries/technology_summary.csv')

    # Create municipality summary
    mun_summary = summary_df.groupby('municipality').agg({
        'mean_velocity': ['mean', 'std', 'min', 'max'],
        'mean_access': ['mean', 'std', 'min', 'max'],
        'velocity_growth': 'mean',
        'access_growth': 'mean'
    }).round(2)

    mun_summary.to_csv('reports/summaries/municipality_summary.csv')

    if debug:
        print("Saved summary files:")
        print("- reports/summaries/complete_analysis.csv")
        print("- reports/summaries/technology_summary.csv")
        print("- reports/summaries/municipality_summary.csv")

#%%
def main(data_directory='Limpieza/data/subdatasets-ubicacion', debug=False):
    """
    Main function to process all municipalities and generate reports
    """
    try:
        # Setup directories
        setup_directories()
        logging.info("Starting analysis process")

        # Get list of all CSV files
        csv_files = [f for f in os.listdir(data_directory) if f.endswith('.csv')]
        logging.info(f"Found {len(csv_files)} municipality files to process")

        # Process each municipality
        results = []
        for file_name in tqdm(csv_files, desc="Processing municipalities"):
            file_path = os.path.join(data_directory, file_name)
            result = process_municipality(file_path, debug)
            if result:
                results.append(result)

                # Convert trends to DataFrame and save
                mun_name, stats, _, trends = result
                trends_df = trends_to_dataframe(trends, mun_name)
                trends_df.to_csv(f'reports/trends/{mun_name}_trends.csv', index=False)

                # Save stats to CSV
                stats_df = pd.DataFrame(stats)
                stats_df.to_csv(f'reports/summaries/{mun_name}_stats.csv', index=False)

        # Process and save all results
        process_and_save_results(results, debug)

        logging.info("Analysis completed successfully")
        if debug:
            print(f"Analysis completed. Processed {len(results)} municipalities.")
            print(f"Reports saved in the 'reports' directory.")

        return results

    except Exception as e:
        logging.error(f"Error in main process: {str(e)}")
        if debug:
            print(f"Error occurred: {str(e)}")
        raise

