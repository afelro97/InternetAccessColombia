import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import glob
import logging
from typing import Dict, List, Tuple, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('analisis_internet')

# Configurar matplotlib para caracteres en español
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class AnalizadorAccesoInternet:
    """Clase para analizar datos de acceso a internet por municipio y tecnología."""
    
    def __init__(self, carpeta_datos: str = r"\\wsl.localhost\Ubuntu\home\hadoop\proyectosBigData\InternetAccessColombia\InternetAccessColombia-DataCleaning\Limpieza\data\resultados", nivel_debug: str = 'INFO'):
        """
        Inicializar el analizador con la ruta de la carpeta y nivel de depuración.
        
        Args:
            carpeta_datos: Ruta a la carpeta con los archivos JSON
            nivel_debug: Nivel de registro ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        """
        self.carpeta_datos = carpeta_datos
        self.nivel_debug = nivel_debug
        self.datos_municipios = []  # Lista para almacenar datos de cada municipio
        self.tendencias_municipios = []  # Lista para almacenar tendencias
        self.df_tecnologias = None  # DataFrame consolidado de tecnologías
        self.df_tendencias = None  # DataFrame consolidado de tendencias
        
        # Establecer nivel de depuración
        nivel_numerico = getattr(logging, nivel_debug.upper(), None)
        if isinstance(nivel_numerico, int):
            logger.setLevel(nivel_numerico)
    
    def cargar_datos_json(self) -> None:
        """
        Cargar datos desde múltiples archivos JSON en la carpeta especificada.
        """
        try:
            patron_archivos = os.path.join(self.carpeta_datos, "*.json")
            archivos_json = glob.glob(patron_archivos)
            
            if not archivos_json:
                logger.error(f"No se encontraron archivos JSON en {self.carpeta_datos}")
                raise FileNotFoundError(f"No se encontraron archivos JSON en {self.carpeta_datos}")
            
            logger.info(f"Encontrados {len(archivos_json)} archivos JSON para procesar")
            
            # Listas para almacenar datos
            datos_tecnologias = []
            datos_tendencias = []
            
            # Procesar cada archivo JSON
            for archivo in archivos_json:
                try:
                    logger.info(f"Procesando archivo: {archivo}")
                    
                    # Leer el archivo JSON
                    with open(archivo, 'r', encoding='utf-8') as f:
                        datos_json = json.load(f)
                    
                    # Extraer municipio
                    if 'municipio' in datos_json:
                        municipio = datos_json['municipio']
                    else:
                        # Extraer del nombre del archivo si no está en los datos
                        nombre_archivo = os.path.basename(archivo)
                        municipio = nombre_archivo.replace('.json', '')
                    
                    # Procesar datos de tecnologías
                    if 'tecnologias' in datos_json and 'stats' in datos_json['tecnologias']:
                        for stat in datos_json['tecnologias']['stats']:
                            # Añadir municipio a cada registro
                            stat['municipio'] = municipio
                            datos_tecnologias.append(stat)
                    
                    # Procesar datos de tendencias
                    if 'trends' in datos_json:
                        # Crear registro para tendencias de velocidad de bajada
                        if 'bajada' in datos_json['trends']:
                            tendencia_bajada = {
                                'municipio': municipio,
                                'tipo': 'bajada',
                                'media': datos_json['trends']['bajada'].get('media', np.nan),
                                'tendencia': datos_json['trends']['bajada'].get('tendencia', np.nan),
                                'r2': datos_json['trends']['bajada'].get('r2', np.nan),
                                'modelo': datos_json['trends']['bajada'].get('modelo', 'desconocido')
                            }
                            datos_tendencias.append(tendencia_bajada)
                        
                        # Crear registro para tendencias de velocidad de subida
                        if 'subida' in datos_json['trends']:
                            tendencia_subida = {
                                'municipio': municipio,
                                'tipo': 'subida',
                                'media': datos_json['trends']['subida'].get('media', np.nan),
                                'tendencia': datos_json['trends']['subida'].get('tendencia', np.nan),
                                'r2': datos_json['trends']['subida'].get('r2', np.nan),
                                'modelo': datos_json['trends']['subida'].get('modelo', 'desconocido')
                            }
                            datos_tendencias.append(tendencia_subida)
                    
                    # Guardar datos brutos del municipio para referencia
                    self.datos_municipios.append(datos_json)
                    
                except Exception as e:
                    logger.error(f"Error al procesar el archivo {archivo}: {str(e)}")
                    # Continuar con el siguiente archivo
            
            # Convertir a DataFrames
            if datos_tecnologias:
                self.df_tecnologias = pd.DataFrame(datos_tecnologias)
                # Renombrar columnas a nombres más estándar
                self.df_tecnologias = self.df_tecnologias.rename(columns={
                    'TRIMESTRE': 'trimestre',
                    'TECNOLOGÍA': 'tecnologia',
                    'VELOCIDAD_BAJADA_MEAN': 'velocidad_bajada_media',
                    'VELOCIDAD_BAJADA_STD': 'velocidad_bajada_std',
                    'VELOCIDAD_SUBIDA_MEAN': 'velocidad_subida_media',
                    'VELOCIDAD_SUBIDA_STD': 'velocidad_subida_std',
                    'ACCESOS_MEAN': 'accesos_media',
                    'ACCESOS_STD': 'accesos_std'
                })
                logger.info(f"DataFrame de tecnologías creado con {len(self.df_tecnologias)} registros")
            else:
                logger.warning("No se encontraron datos de tecnologías en los archivos JSON")
            
            if datos_tendencias:
                self.df_tendencias = pd.DataFrame(datos_tendencias)
                logger.info(f"DataFrame de tendencias creado con {len(self.df_tendencias)} registros")
            else:
                logger.warning("No se encontraron datos de tendencias en los archivos JSON")
                
        except Exception as e:
            logger.error(f"Error al cargar los datos JSON: {str(e)}")
            raise
    
    def analizar_por_tecnologia(self) -> pd.DataFrame:
        """
        Analiza los datos agrupados por municipio y tecnología.
        
        Returns:
            DataFrame con el análisis por municipio y tecnología
        """
        if self.df_tecnologias is None:
            logger.error("Datos de tecnologías no cargados. Llame a cargar_datos_json() primero.")
            return pd.DataFrame()
            
        try:
            logger.info("Analizando datos por municipio y tecnología")
            
            # Obtener las columnas necesarias
            if 'municipio' not in self.df_tecnologias.columns or 'tecnologia' not in self.df_tecnologias.columns:
                # Verificar si existen columnas equivalentes
                if 'municipio' not in self.df_tecnologias.columns:
                    columnas_posibles = [col for col in self.df_tecnologias.columns if 'muni' in col.lower()]
                    if columnas_posibles:
                        self.df_tecnologias = self.df_tecnologias.rename(columns={columnas_posibles[0]: 'municipio'})
                    else:
                        logger.error("No se encontró una columna para municipio")
                        return pd.DataFrame()
                
                if 'tecnologia' not in self.df_tecnologias.columns:
                    columnas_posibles = [col for col in self.df_tecnologias.columns if 'tecno' in col.lower() or 'tech' in col.lower()]
                    if columnas_posibles:
                        self.df_tecnologias = self.df_tecnologias.rename(columns={columnas_posibles[0]: 'tecnologia'})
                    else:
                        logger.error("No se encontró una columna para tecnología")
                        return pd.DataFrame()
            
            # Agrupar por municipio y tecnología, calcular promedios
            columnas_acceso = [col for col in self.df_tecnologias.columns if 'acceso' in col.lower()]
            if not columnas_acceso:
                columnas_acceso = [col for col in self.df_tecnologias.columns if 'acces' in col.lower()]
            
            if columnas_acceso:
                columna_acceso = columnas_acceso[0]
                logger.info(f"Usando columna '{columna_acceso}' para análisis de acceso")
                
                # Agrupar y calcular estadísticas
                analisis = self.df_tecnologias.groupby(['municipio', 'tecnologia'])[columna_acceso].agg([
                    ('acceso_promedio', 'mean'),
                    ('total_registros', 'count')
                ]).reset_index()
                
                # Obtener el promedio general de acceso por municipio
                acceso_por_municipio = self.df_tecnologias.groupby('municipio')[columna_acceso].mean().reset_index()
                acceso_por_municipio.columns = ['municipio', 'acceso_general_media']
                
                # Fusionar con el análisis por tecnología
                analisis = pd.merge(analisis, acceso_por_municipio, on='municipio')
                
                logger.info(f"Análisis por tecnología completado para {len(analisis)} combinaciones de municipio-tecnología")
                return analisis
            else:
                logger.warning("No se encontró una columna adecuada para accesos o penetración")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error durante el análisis por tecnología: {str(e)}")
            raise
    
    def analizar_por_velocidad(self) -> pd.DataFrame:
        """
        Analiza los datos de tendencias para obtener clasificación por velocidad de internet.
        
        Returns:
            DataFrame con el análisis por velocidad
        """
        if self.df_tendencias is None:
            logger.error("Datos de tendencias no cargados. Llame a cargar_datos_json() primero.")
            return pd.DataFrame()
            
        try:
            logger.info("Analizando datos por velocidad de internet")
            
            # Filtrar solo datos de tendencia de bajada para clasificación
            tendencias_bajada = self.df_tendencias[self.df_tendencias['tipo'] == 'bajada'].copy()
            
            if len(tendencias_bajada) > 0:
                # Ordenar por media de velocidad
                tendencias_bajada = tendencias_bajada.sort_values(by='media', ascending=False)
                
                # Añadir ranking
                tendencias_bajada['ranking'] = range(1, len(tendencias_bajada) + 1)
                
                logger.info(f"Análisis por velocidad completado para {len(tendencias_bajada)} municipios")
                return tendencias_bajada
            else:
                logger.warning("No se encontraron datos de tendencia de bajada para analizar")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error durante el análisis por velocidad: {str(e)}")
            raise
    
    def generar_grafica_mejores_peores(self, n: int = 10) -> None:
        """
        Genera gráficas para los 10 municipios con mejores y peores resultados de acceso a internet.
        
        Args:
            n: Número de municipios a mostrar en cada categoría
        """
        try:
            logger.info(f"Generando gráficas para los {n} mejores y peores municipios")
            
            # 1. Análisis por tecnología (basado en accesos)
            if self.df_tecnologias is not None:
                # Calcular el promedio de accesos por municipio
                columnas_acceso = [col for col in self.df_tecnologias.columns if 'acceso' in col.lower()]
                if not columnas_acceso:
                    columnas_acceso = [col for col in self.df_tecnologias.columns if 'acces' in col.lower()]
                
                if columnas_acceso:
                    columna_acceso = columnas_acceso[0]
                    
                    # Agrupar por municipio y calcular la media de accesos
                    accesos_por_municipio = self.df_tecnologias.groupby('municipio')[columna_acceso].mean().reset_index()
                    accesos_por_municipio.columns = ['municipio', 'acceso_promedio']
                    
                    # Ordenar y obtener los mejores y peores
                    accesos_ordenados = accesos_por_municipio.sort_values(by='acceso_promedio', ascending=False)
                    mejores_acceso = accesos_ordenados.head(n)
                    peores_acceso = accesos_ordenados.tail(n).iloc[::-1]  # Invertir para mostrar de peor a menos peor
                    
                    # Generar gráfica
                    plt.figure(figsize=(14, 10))
                    
                    # Gráfica de mejores municipios por acceso
                    plt.subplot(2, 1, 1)
                    sns.barplot(x='acceso_promedio', y='municipio', data=mejores_acceso, palette='viridis')
                    plt.title(f'Top {n} Municipios con Mayor Acceso a Internet')
                    plt.xlabel('Promedio de Accesos')
                    plt.ylabel('Municipio')
                    
                    # Gráfica de peores municipios por acceso
                    plt.subplot(2, 1, 2)
                    sns.barplot(x='acceso_promedio', y='municipio', data=peores_acceso, palette='viridis')
                    plt.title(f'Top {n} Municipios con Menor Acceso a Internet')
                    plt.xlabel('Promedio de Accesos')
                    plt.ylabel('Municipio')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.carpeta_datos, 'mejores_peores_acceso.png'), dpi=300)
                    logger.info(f"Gráfica de acceso guardada en {os.path.join(self.carpeta_datos, 'mejores_peores_acceso.png')}")
                    plt.close()
            
            # 2. Análisis por velocidad (basado en tendencias)
            if self.df_tendencias is not None:
                # Filtrar solo datos de tendencia de bajada
                tendencias_bajada = self.df_tendencias[self.df_tendencias['tipo'] == 'bajada'].copy()
                
                if len(tendencias_bajada) > 0:
                    # Ordenar por media de velocidad
                    tendencias_ordenadas = tendencias_bajada.sort_values(by='media', ascending=False)
                    mejores_velocidad = tendencias_ordenadas.head(n)
                    peores_velocidad = tendencias_ordenadas.tail(n).iloc[::-1]  # Invertir para mostrar de peor a menos peor
                    
                    # Generar gráfica
                    plt.figure(figsize=(14, 10))
                    
                    # Gráfica de mejores municipios por velocidad
                    plt.subplot(2, 1, 1)
                    sns.barplot(x='media', y='municipio', data=mejores_velocidad, palette='viridis')
                    plt.title(f'Top {n} Municipios con Mayor Velocidad de Internet (Bajada)')
                    plt.xlabel('Velocidad Media (Mbps)')
                    plt.ylabel('Municipio')
                    
                    # Añadir tendencia como anotación
                    for i, row in enumerate(mejores_velocidad.itertuples()):
                        plt.text(row.media + 0.5, i, f'Tendencia: {row.tendencia:.2f}', va='center')
                    
                    # Gráfica de peores municipios por velocidad
                    plt.subplot(2, 1, 2)
                    sns.barplot(x='media', y='municipio', data=peores_velocidad, palette='viridis')
                    plt.title(f'Top {n} Municipios con Menor Velocidad de Internet (Bajada)')
                    plt.xlabel('Velocidad Media (Mbps)')
                    plt.ylabel('Municipio')
                    
                    # Añadir tendencia como anotación
                    for i, row in enumerate(peores_velocidad.itertuples()):
                        plt.text(row.media + 0.5, i, f'Tendencia: {row.tendencia:.2f}', va='center')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.carpeta_datos, 'mejores_peores_velocidad.png'), dpi=300)
                    logger.info(f"Gráfica de velocidad guardada en {os.path.join(self.carpeta_datos, 'mejores_peores_velocidad.png')}")
                    plt.close()
            
            logger.info("Generación de gráficas completada con éxito")
            
        except Exception as e:
            logger.error(f"Error durante la generación de gráficas: {str(e)}")
            raise

# Función principal para ejecutar el análisis
def ejecutar_analisis_completo(carpeta_datos='resultados', nivel_debug='INFO'):
    """
    Ejecuta el análisis completo de los datos de acceso a internet.
    
    Args:
        carpeta_datos: Ruta a la carpeta con los archivos JSON
        nivel_debug: Nivel de registro ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    try:
        logger.info("Iniciando análisis completo de acceso a internet")
        
        # Crear instancia del analizador
        analizador = AnalizadorAccesoInternet(carpeta_datos, nivel_debug)
        
        # Cargar datos JSON
        analizador.cargar_datos_json()
        
        # Realizar análisis
        analisis_tecnologia = analizador.analizar_por_tecnologia()
        analisis_velocidad = analizador.analizar_por_velocidad()
        
        # Generar gráficas de mejores y peores municipios
        analizador.generar_grafica_mejores_peores(n=10)
        
        # Guardar resultados en CSV para referencia
        if not analisis_tecnologia.empty:
            analisis_tecnologia.to_csv(os.path.join(carpeta_datos, 'analisis_tecnologia.csv'), index=False)
            logger.info(f"Resultados de análisis por tecnología guardados en {os.path.join(carpeta_datos, 'analisis_tecnologia.csv')}")
        
        if not analisis_velocidad.empty:
            analisis_velocidad.to_csv(os.path.join(carpeta_datos, 'analisis_velocidad.csv'), index=False)
            logger.info(f"Resultados de análisis por velocidad guardados en {os.path.join(carpeta_datos, 'analisis_velocidad.csv')}")
        
        logger.info("Análisis completo finalizado con éxito")
        
    except Exception as e:
        logger.error(f"Error durante el análisis completo: {str(e)}")
        raise

# Si se ejecuta como script principal
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analizar datos de acceso a internet desde archivos JSON.')
    parser.add_argument('--carpeta', type=str, default='resultados', help='Carpeta con archivos JSON (default: resultados)')
    parser.add_argument('--debug', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Nivel de depuración (default: INFO)')
    args = parser.parse_args()
    
    ejecutar_analisis_completo(args.carpeta, args.debug)
