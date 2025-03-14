import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import glob
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

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

    def __init__(self, carpeta_datos: str = "./resultados", nivel_debug: str = 'INFO'):
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
        self.resultados_markdown = []  # Lista para almacenar el contenido de markdown

        # Lista de municipios a excluir (considerados atípicos)
        self.municipios_excluir = [
            "CAUCA_EL_TAMBO",
            "CALDAS_BELALCÁZAR"
        ]

        logger.info(f"Se excluirán los siguientes municipios por tener valores atípicos: {', '.join(self.municipios_excluir)}")

        # Establecer nivel de depuración
        nivel_numerico = getattr(logging, nivel_debug.upper(), None)
        if isinstance(nivel_numerico, int):
            logger.setLevel(nivel_numerico)

        # Crear carpeta para figuras si no existe
        self.carpeta_figuras = os.path.join(self.carpeta_datos, 'figuras')
        if not os.path.exists(self.carpeta_figuras):
            os.makedirs(self.carpeta_figuras)

    def agregar_a_markdown(self, contenido: str) -> None:
        """
        Agrega contenido al reporte markdown.

        Args:
            contenido: Texto a agregar al reporte
        """
        self.resultados_markdown.append(contenido)

    def cargar_datos_json(self) -> None:
        """
        Cargar datos desde múltiples archivos JSON en la carpeta especificada,
        excluyendo los municipios marcados como atípicos.
        """
        try:
            patron_archivos = os.path.join(self.carpeta_datos, "*.json")
            archivos_json = glob.glob(patron_archivos)

            if not archivos_json:
                logger.error(f"No se encontraron archivos JSON en {self.carpeta_datos}")
                raise FileNotFoundError(f"No se encontraron archivos JSON en {self.carpeta_datos}")

            # Filtrar municipios a excluir
            archivos_json_filtrados = []
            for archivo in archivos_json:
                nombre_base = os.path.basename(archivo)
                excluido = False
                for municipio in self.municipios_excluir:
                    if municipio in nombre_base:
                        logger.info(f"Excluyendo municipio con valores atípicos: {nombre_base}")
                        excluido = True
                        break

                if not excluido:
                    archivos_json_filtrados.append(archivo)

            total_archivos = len(archivos_json)
            archivos_filtrados = len(archivos_json_filtrados)

            logger.info(f"Encontrados {total_archivos} archivos JSON. Se excluyen {total_archivos - archivos_filtrados} por valores atípicos.")
            self.agregar_a_markdown(f"## Datos Cargados\n\nSe encontraron **{total_archivos} archivos JSON** para análisis. "
                                    f"Se excluyeron **{total_archivos - archivos_filtrados} municipios** por tener valores atípicos.")

            # Listas para almacenar datos
            datos_tecnologias = []
            datos_tendencias = []

            # Procesar cada archivo JSON
            for archivo in archivos_json_filtrados:
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
                        municipio = municipio.replace('_results', '')

                    # Verificar si el municipio debe ser excluido
                    if any(excluir in municipio for excluir in self.municipios_excluir):
                        logger.info(f"Saltando municipio con valores atípicos: {municipio}")
                        continue

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
                self.agregar_a_markdown(f"\nSe procesaron **{len(self.df_tecnologias)} registros** de datos de tecnologías.")
            else:
                logger.warning("No se encontraron datos de tecnologías en los archivos JSON")
                self.agregar_a_markdown("\n⚠️ No se encontraron datos de tecnologías en los archivos JSON.")

            if datos_tendencias:
                self.df_tendencias = pd.DataFrame(datos_tendencias)
                logger.info(f"DataFrame de tendencias creado con {len(self.df_tendencias)} registros")
                self.agregar_a_markdown(f"\nSe procesaron **{len(self.df_tendencias)} registros** de datos de tendencias.")
            else:
                logger.warning("No se encontraron datos de tendencias en los archivos JSON")
                self.agregar_a_markdown("\n⚠️ No se encontraron datos de tendencias en los archivos JSON.")

        except Exception as e:
            logger.error(f"Error al cargar los datos JSON: {str(e)}")
            self.agregar_a_markdown(f"\n❌ **Error al cargar los datos JSON**: {str(e)}")
            raise



    def eliminar_outliers_estadisticos(self, df: pd.DataFrame, columna: str, umbral: float = 3.0) -> pd.DataFrame:
        """
        Elimina outliers estadísticos basados en Z-score.

        Args:
            df: DataFrame con los datos
            columna: Nombre de la columna para detectar outliers
            umbral: Umbral de Z-score para considerar un dato como outlier

        Returns:
            DataFrame sin outliers
        """
        if df is None or df.empty or columna not in df.columns:
            return df

        # Calcular Z-score
        z_scores = np.abs(stats.zscore(df[columna].dropna()))

        # Crear máscara para filtrar outliers
        mask = np.ones(len(df), dtype=bool)
        mask_indices = np.where(~np.isnan(df[columna]))[0]
        mask[mask_indices[z_scores > umbral]] = False

        # Filtrar DataFrame
        df_filtrado = df[mask].copy()

        outliers_count = len(df) - len(df_filtrado)
        if outliers_count > 0:
            logger.info(f"Se eliminaron {outliers_count} outliers en la columna '{columna}' usando Z-score > {umbral}")

        return df_filtrado

    def visualizar_tecnologias_populares(self, analisis: pd.DataFrame) -> None:
        """
        Crea una visualización de las tecnologías más utilizadas.

        Args:
            analisis: DataFrame con el análisis por municipio y tecnología
        """
        try:
            # Agrupar por tecnología
            tech_counts = analisis.groupby('tecnologia').agg({
                'municipio': 'nunique',
                'acceso_promedio': 'mean'
            }).reset_index()

            # Renombrar columnas
            tech_counts.columns = ['tecnologia', 'num_municipios', 'acceso_promedio']

            # Ordenar por número de municipios
            tech_counts = tech_counts.sort_values('num_municipios', ascending=False)

            # Tomar las 10 principales tecnologías
            top_techs = tech_counts.head(10)

            plt.figure(figsize=(12, 8))
            bars = plt.bar(top_techs['tecnologia'], top_techs['num_municipios'], color='skyblue')

            # Añadir etiquetas a las barras
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                         f'{int(height)}',
                         ha='center', va='bottom')

            plt.title('Top 10 Tecnologías por Número de Municipios', fontsize=15)
            plt.xlabel('Tecnología', fontsize=12)
            plt.ylabel('Número de Municipios', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            # Guardar la figura
            ruta_figura = os.path.join(self.carpeta_figuras, 'tecnologias_populares.png')
            plt.savefig(ruta_figura, dpi=300)
            plt.close()

            # Agregar al markdown
            self.agregar_a_markdown("\n### Tecnologías más Utilizadas\n")
            self.agregar_a_markdown(f"![Top 10 Tecnologías por Municipios](figuras/tecnologias_populares.png)\n")
            self.agregar_a_markdown("La gráfica muestra las 10 tecnologías más utilizadas en los municipios analizados. "
                                    "Cada barra representa el número de municipios que cuentan con esa tecnología. "
                                    "Esto permite identificar cuáles son las tecnologías de conectividad más extendidas en el territorio colombiano.")

            logger.info(f"Visualización de tecnologías populares guardada en {ruta_figura}")

        except Exception as e:
            logger.error(f"Error al crear visualización de tecnologías populares: {str(e)}")
            self.agregar_a_markdown(f"\n⚠️ No se pudo crear la visualización de tecnologías populares debido a un error: {str(e)}")

    def analizar_por_velocidad(self) -> pd.DataFrame:
        """
        Analiza los datos de tendencias para obtener clasificación por velocidad de internet.

        Returns:
            DataFrame con el análisis por velocidad
        """
        if self.df_tendencias is None:
            logger.error("Datos de tendencias no cargados. Llame a cargar_datos_json() primero.")
            self.agregar_a_markdown("\n❌ **Error**: Datos de tendencias no cargados. Es necesario cargar los datos primero.")
            return pd.DataFrame()

        try:
            logger.info("Analizando datos por velocidad de internet")
            self.agregar_a_markdown("\n## Análisis por Velocidad de Internet")

            # Filtrar solo datos de tendencia de bajada para clasificación
            tendencias_bajada = self.df_tendencias[self.df_tendencias['tipo'] == 'bajada'].copy()

            if len(tendencias_bajada) > 0:
                # Eliminar datos atípicos adicionales usando Z-score para:
                # 1. Velocidad media (media)
                # 2. Tendencia de cambio (tendencia)
                tendencias_bajada = self.eliminar_outliers_estadisticos(tendencias_bajada, 'media', 3.0)
                tendencias_bajada = self.eliminar_outliers_estadisticos(tendencias_bajada, 'tendencia', 3.0)

                # Ordenar por media de velocidad
                tendencias_bajada = tendencias_bajada.sort_values(by='media', ascending=False)

                # Añadir ranking
                tendencias_bajada['ranking'] = range(1, len(tendencias_bajada) + 1)

                logger.info(f"Análisis por velocidad completado para {len(tendencias_bajada)} municipios")
                self.agregar_a_markdown(f"\nSe analizaron las velocidades de internet de **{len(tendencias_bajada)} municipios**, "
                                        f"excluyendo valores atípicos.")

                # Generar visualización de distribución de velocidades
                self.visualizar_distribucion_velocidades(tendencias_bajada)

                # Generar visualización de tendencias vs velocidades
                self.visualizar_tendencia_vs_velocidad(tendencias_bajada)

                return tendencias_bajada
            else:
                logger.warning("No se encontraron datos de tendencia de bajada para analizar")
                self.agregar_a_markdown("\n⚠️ **Advertencia**: No se encontraron datos de tendencia de bajada para analizar.")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error durante el análisis por velocidad: {str(e)}")
            self.agregar_a_markdown(f"\n❌ **Error durante el análisis por velocidad**: {str(e)}")
            raise

    def visualizar_distribucion_velocidades(self, tendencias: pd.DataFrame) -> None:
        """
        Crea una visualización de la distribución de velocidades de internet.

        Args:
            tendencias: DataFrame con las tendencias de velocidad
        """
        try:
            plt.figure(figsize=(12, 8))

            # Crear histograma con KDE
            sns.histplot(tendencias['media'], kde=True, bins=20, color='skyblue')

            # Añadir líneas verticales para estadísticas descriptivas
            media = tendencias['media'].mean()
            mediana = tendencias['media'].median()

            plt.axvline(media, color='red', linestyle='--', label=f'Media: {media:.2f} Mbps')
            plt.axvline(mediana, color='green', linestyle='-.', label=f'Mediana: {mediana:.2f} Mbps')

            plt.title('Distribución de Velocidades de Internet (Bajada) - Sin Valores Atípicos', fontsize=15)
            plt.xlabel('Velocidad Media (Mbps)', fontsize=12)
            plt.ylabel('Frecuencia', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Guardar la figura
            ruta_figura = os.path.join(self.carpeta_figuras, 'distribucion_velocidades.png')
            plt.savefig(ruta_figura, dpi=300)
            plt.close()

            # Agregar al markdown
            self.agregar_a_markdown("\n### Distribución de Velocidades de Internet\n")
            self.agregar_a_markdown(f"![Distribución de Velocidades](figuras/distribucion_velocidades.png)\n")

            # Calcular estadísticas descriptivas para el reporte
            min_vel = tendencias['media'].min()
            max_vel = tendencias['media'].max()
            std_vel = tendencias['media'].std()
            q1 = tendencias['media'].quantile(0.25)
            q3 = tendencias['media'].quantile(0.75)

            self.agregar_a_markdown("La gráfica muestra la distribución de velocidades de internet (bajada) en los municipios analizados, "
                                    "excluyendo valores atípicos para una mejor visualización. "
                                    f"La velocidad media es de **{media:.2f} Mbps** y la mediana es de **{mediana:.2f} Mbps**. "
                                    f"La velocidad mínima registrada es de **{min_vel:.2f} Mbps** y la máxima de **{max_vel:.2f} Mbps**. "
                                    f"El 50% central de los municipios tiene velocidades entre **{q1:.2f}** y **{q3:.2f} Mbps**.")

            logger.info(f"Visualización de distribución de velocidades guardada en {ruta_figura}")

        except Exception as e:
            logger.error(f"Error al crear visualización de distribución de velocidades: {str(e)}")
            self.agregar_a_markdown(f"\n⚠️ No se pudo crear la visualización de distribución de velocidades debido a un error: {str(e)}")

    def visualizar_tendencia_vs_velocidad(self, tendencias: pd.DataFrame) -> None:
        """
        Crea una visualización de la relación entre tendencia y velocidad de internet.

        Args:
            tendencias: DataFrame con las tendencias de velocidad
        """
        try:
            plt.figure(figsize=(12, 8))

            # Crear gráfico de dispersión
            scatter = plt.scatter(tendencias['media'], tendencias['tendencia'],
                                  c=tendencias['r2'], cmap='viridis', alpha=0.7, s=80)

            # Añadir líneas de referencia
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Sin cambio')
            plt.axvline(x=tendencias['media'].median(), color='blue', linestyle='--',
                        alpha=0.7, label=f'Mediana: {tendencias["media"].median():.2f} Mbps')

            # Marcar puntos interesantes - ajustando el criterio para no mostrar demasiados
            umbral_tendencia = tendencias['tendencia'].quantile(0.90)  # Mostrar solo top 10%
            umbral_velocidad = tendencias['media'].quantile(0.90)      # Mostrar solo top 10%

            municipios_destacados = tendencias[
                (tendencias['tendencia'] > umbral_tendencia) |
                (tendencias['tendencia'] < -umbral_tendencia) |
                (tendencias['media'] > umbral_velocidad)
                ]

            # Limitar a máximo 10 municipios destacados
            if len(municipios_destacados) > 10:
                municipios_destacados = pd.concat([
                    municipios_destacados.nlargest(5, 'media'),
                    municipios_destacados.nlargest(5, 'tendencia')
                ]).drop_duplicates()

            # Etiquetar algunos puntos destacados
            for idx, row in municipios_destacados.iterrows():
                plt.annotate(row['municipio'], xy=(row['media'], row['tendencia']),
                             xytext=(5, 5), textcoords='offset points',
                             fontsize=9, color='darkblue')

            # Añadir colorbar para R²
            cbar = plt.colorbar(scatter)
            cbar.set_label('R² (Calidad del modelo)', fontsize=10)

            # Dividir el gráfico en cuadrantes
            media_vel = tendencias['media'].median()

            plt.text(media_vel*0.2, tendencias['tendencia'].max()*0.8,
                     'Baja velocidad\nMejorando',
                     ha='center', bbox=dict(facecolor='lightgreen', alpha=0.4))

            plt.text(media_vel*1.8, tendencias['tendencia'].max()*0.8,
                     'Alta velocidad\nMejorando',
                     ha='center', bbox=dict(facecolor='green', alpha=0.4))

            plt.text(media_vel*0.2, tendencias['tendencia'].min()*0.8,
                     'Baja velocidad\nEmpeorando',
                     ha='center', bbox=dict(facecolor='salmon', alpha=0.4))

            plt.text(media_vel*1.8, tendencias['tendencia'].min()*0.8,
                     'Alta velocidad\nEmpeorando',
                     ha='center', bbox=dict(facecolor='lightcoral', alpha=0.4))

            plt.title('Relación entre Velocidad Media y Tendencia por Municipio (Sin Valores Atípicos)', fontsize=15)
            plt.xlabel('Velocidad Media (Mbps)', fontsize=12)
            plt.ylabel('Tendencia (Cambio en Mbps)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best')
            plt.tight_layout()

            # Guardar la figura
            ruta_figura = os.path.join(self.carpeta_figuras, 'tendencia_vs_velocidad.png')
            plt.savefig(ruta_figura, dpi=300)
            plt.close()

            # Agregar al markdown
            self.agregar_a_markdown("\n### Relación entre Velocidad y Tendencia\n")
            self.agregar_a_markdown(f"![Velocidad vs Tendencia](figuras/tendencia_vs_velocidad.png)\n")

            # Calcular estadísticas para el reporte
            municipios_mejorando = len(tendencias[tendencias['tendencia'] > 0])
            municipios_empeorando = len(tendencias[tendencias['tendencia'] < 0])
            pct_mejorando = (municipios_mejorando / len(tendencias)) * 100

            self.agregar_a_markdown("Esta gráfica muestra la relación entre la velocidad actual de internet y su tendencia de cambio para cada municipio, "
                                    "excluyendo valores atípicos para una mejor visualización. "
                                    "Los puntos están coloreados según el valor R² que indica la calidad del modelo de predicción.\n\n"
                                    "Los cuadrantes representan:\n"
                                    "- **Alta velocidad, Mejorando**: Municipios con buena conectividad que sigue mejorando\n"
                                    "- **Alta velocidad, Empeorando**: Municipios con buena conectividad pero en deterioro\n"
                                    "- **Baja velocidad, Mejorando**: Municipios rezagados pero que están mejorando\n"
                                    "- **Baja velocidad, Empeorando**: Municipios en situación crítica\n\n"
                                    f"De los municipios analizados, **{municipios_mejorando}** ({pct_mejorando:.1f}%) muestran tendencia positiva (mejorando) "
                                    f"y **{municipios_empeorando}** ({100-pct_mejorando:.1f}%) muestran tendencia negativa (empeorando).")

            logger.info(f"Visualización de tendencia vs velocidad guardada en {ruta_figura}")

        except Exception as e:
            logger.error(f"Error al crear visualización de tendencia vs velocidad: {str(e)}")
            self.agregar_a_markdown(f"\n⚠️ No se pudo crear la visualización de tendencia vs velocidad debido a un error: {str(e)}")

    def generar_grafica_mejores_peores(self, n: int = 10) -> None:
        """
        Genera gráficas para los 10 municipios con mejores y peores resultados de acceso a internet.

        Args:
            n: Número de municipios a mostrar en cada categoría
        """
        try:
            logger.info(f"Generando gráficas para los {n} mejores y peores municipios")
            self.agregar_a_markdown(f"\n## Top {n} Municipios con Mejor y Peor Acceso a Internet")

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

                    # Filtrar municipios excluidos y valores atípicos
                    for municipio in self.municipios_excluir:
                        accesos_por_municipio = accesos_por_municipio[~accesos_por_municipio['municipio'].str.contains(municipio, case=False)]

                    # Eliminar outliers estadísticos
                    accesos_por_municipio = self.eliminar_outliers_estadisticos(accesos_por_municipio, 'acceso_promedio', 2.5)

                    # Ordenar y obtener los mejores y peores
                    accesos_ordenados = accesos_por_municipio.sort_values(by='acceso_promedio', ascending=False)
                    mejores_acceso = accesos_ordenados.head(n)
                    peores_acceso = accesos_ordenados.tail(n).iloc[::-1]  # Invertir para mostrar de peor a menos peor

                    # Generar gráfica
                    plt.figure(figsize=(14, 10))

                    # Gráfica de mejores municipios por acceso
                    plt.subplot(2, 1, 1)
                    ax1 = sns.barplot(x='acceso_promedio', y='municipio', data=mejores_acceso, palette='viridis')
                    plt.title(f'Top {n} Municipios con Mayor Acceso a Internet', fontsize=15)
                    plt.xlabel('Promedio de Accesos', fontsize=12)
                    plt.ylabel('Municipio', fontsize=12)

                    # Agregar valores en las barras
                    for i, v in enumerate(mejores_acceso['acceso_promedio']):
                        ax1.text(v + 0.1, i, f"{v:.2f}", va='center')

                    # Gráfica de peores municipios por acceso
                    plt.subplot(2, 1, 2)
                    ax2 = sns.barplot(x='acceso_promedio', y='municipio', data=peores_acceso, palette='viridis')
                    plt.title(f'Top {n} Municipios con Menor Acceso a Internet', fontsize=15)
                    plt.xlabel('Promedio de Accesos', fontsize=12)
                    plt.ylabel('Municipio', fontsize=12)

                    # Agregar valores en las barras
                    for i, v in enumerate(peores_acceso['acceso_promedio']):
                        ax2.text(v + 0.1, i, f"{v:.2f}", va='center')

                    plt.tight_layout()

                    # Guardar la figura
                    ruta_figura = os.path.join(self.carpeta_figuras, 'mejores_peores_acceso.png')
                    plt.savefig(ruta_figura, dpi=300)
                    plt.close()

                    # Agregar al markdown
                    self.agregar_a_markdown("\n### Municipios por Nivel de Acceso\n")
                    self.agregar_a_markdown(f"![Mejores y Peores Municipios por Acceso](figuras/mejores_peores_acceso.png)\n")
                    self.agregar_a_markdown(f"Esta gráfica muestra los {n} municipios con mayor y menor acceso a internet según los datos analizados "
                                            f"(excluyendo valores atípicos). "
                                            f"El municipio con mayor acceso es **{mejores_acceso.iloc[0]['municipio']}** con un valor de "
                                            f"**{mejores_acceso.iloc[0]['acceso_promedio']:.2f}**, mientras que el municipio con menor acceso es "
                                            f"**{peores_acceso.iloc[-1]['municipio']}** con un valor de **{peores_acceso.iloc[-1]['acceso_promedio']:.2f}**.")

                    logger.info(f"Gráfica de mejores y peores municipios por acceso guardada en {ruta_figura}")

            # 2. Análisis por velocidad (basado en tendencias)
            if self.df_tendencias is not None:
                # Filtrar solo datos de tendencia de bajada
                tendencias_bajada = self.df_tendencias[self.df_tendencias['tipo'] == 'bajada'].copy()

                # Excluir municipios específicos
                for municipio in self.municipios_excluir:
                    tendencias_bajada = tendencias_bajada[~tendencias_bajada['municipio'].str.contains(municipio, case=False)]

                # Eliminar outliers estadísticos
                tendencias_bajada = self.eliminar_outliers_estadisticos(tendencias_bajada, 'media', 2.5)

                if len(tendencias_bajada) > 0:
                    # Ordenar por media de velocidad
                    tendencias_ordenadas = tendencias_bajada.sort_values(by='media', ascending=False)
                    mejores_velocidad = tendencias_ordenadas.head(n)
                    peores_velocidad = tendencias_ordenadas.tail(n).iloc[::-1]  # Invertir para mostrar de peor a menos peor

                    # Generar gráfica
                    plt.figure(figsize=(14, 10))

                    # Gráfica de mejores municipios por velocidad
                    plt.subplot(2, 1, 1)
                    ax1 = sns.barplot(x='media', y='municipio', data=mejores_velocidad, palette='viridis')
                    plt.title(f'Top {n} Municipios con Mayor Velocidad de Internet (Bajada)', fontsize=15)
                    plt.xlabel('Velocidad Media (Mbps)', fontsize=12)
                    plt.ylabel('Municipio', fontsize=12)

                    # Añadir tendencia como anotación
                    for i, row in enumerate(mejores_velocidad.itertuples()):
                        ax1.text(row.media + 0.5, i, f'Tendencia: {row.tendencia:.2f}', va='center')

                    # Gráfica de peores municipios por velocidad
                    plt.subplot(2, 1, 2)
                    ax2 = sns.barplot(x='media', y='municipio', data=peores_velocidad, palette='viridis')
                    plt.title(f'Top {n} Municipios con Menor Velocidad de Internet (Bajada)', fontsize=15)
                    plt.xlabel('Velocidad Media (Mbps)', fontsize=12)
                    plt.ylabel('Municipio', fontsize=12)

                    # Añadir tendencia como anotación
                    for i, row in enumerate(peores_velocidad.itertuples()):
                        ax2.text(row.media + 0.5, i, f'Tendencia: {row.tendencia:.2f}', va='center')

                    plt.tight_layout()

                    # Guardar la figura
                    ruta_figura = os.path.join(self.carpeta_figuras, 'mejores_peores_velocidad.png')
                    plt.savefig(ruta_figura, dpi=300)
                    plt.close()

                    # Agregar al markdown
                    self.agregar_a_markdown("\n### Municipios por Velocidad de Internet\n")
                    self.agregar_a_markdown(f"![Mejores y Peores Municipios por Velocidad](figuras/mejores_peores_velocidad.png)\n")
                    self.agregar_a_markdown(f"Esta gráfica muestra los {n} municipios con mayor y menor velocidad de internet (bajada), "
                                            f"excluyendo valores atípicos. "
                                            f"El municipio con mayor velocidad es **{mejores_velocidad.iloc[0]['municipio']}** con "
                                            f"**{mejores_velocidad.iloc[0]['media']:.2f} Mbps** y una tendencia de "
                                            f"**{mejores_velocidad.iloc[0]['tendencia']:.2f}**. El municipio con menor velocidad es "
                                            f"**{peores_velocidad.iloc[-1]['municipio']}** con **{peores_velocidad.iloc[-1]['media']:.2f} Mbps** "
                                            f"y una tendencia de **{peores_velocidad.iloc[-1]['tendencia']:.2f}**.\n\n"
                                            f"Es importante notar que la tendencia indica la dirección de cambio en la velocidad: "
                                            f"valores positivos significan mejora y valores negativos indican deterioro.")

                    logger.info(f"Gráfica de mejores y peores municipios por velocidad guardada en {ruta_figura}")

            logger.info("Generación de gráficas de mejores y peores municipios completada con éxito")

        except Exception as e:
            logger.error(f"Error durante la generación de gráficas de mejores y peores municipios: {str(e)}")
            self.agregar_a_markdown(f"\n❌ **Error durante la generación de gráficas de mejores y peores municipios**: {str(e)}")

    def generar_mapa_calor_tecnologias(self) -> None:
        """
        Genera un mapa de calor que muestra la relación entre tecnologías y velocidad de internet.
        """
        try:
            if self.df_tecnologias is None:
                logger.warning("No hay datos de tecnologías para generar el mapa de calor")
                return

            logger.info("Generando mapa de calor de tecnologías y velocidad")
            self.agregar_a_markdown("\n## Relación entre Tecnologías y Velocidad de Internet")

            # Calcular la velocidad media por tecnología
            if 'tecnologia' in self.df_tecnologias.columns and 'velocidad_bajada_media' in self.df_tecnologias.columns:
                # Filtrar municipios a excluir
                df_filtrado = self.df_tecnologias.copy()
                for municipio in self.municipios_excluir:
                    df_filtrado = df_filtrado[~df_filtrado['municipio'].str.contains(municipio, case=False)]

                # Agrupar por tecnología y calcular estadísticas
                velocidad_por_tecnologia = df_filtrado.groupby('tecnologia').agg({
                    'velocidad_bajada_media': ['mean', 'std', 'count'],
                    'velocidad_subida_media': ['mean', 'std', 'count'] if 'velocidad_subida_media' in df_filtrado.columns else None
                }).reset_index()

                # Limpiar y renombrar columnas
                velocidad_por_tecnologia.columns = ['_'.join(col).strip('_') for col in velocidad_por_tecnologia.columns.values]

                # Ordenar por velocidad de bajada
                velocidad_por_tecnologia = velocidad_por_tecnologia.sort_values('velocidad_bajada_media_mean', ascending=False)

                # Tomar las 15 principales tecnologías para el mapa de calor
                top_tecnologias = velocidad_por_tecnologia.head(15)

                # Crear un DataFrame más adecuado para el mapa de calor
                heatmap_data = pd.DataFrame({
                    'Tecnología': top_tecnologias['tecnologia'],
                    'Velocidad Bajada (Mbps)': top_tecnologias['velocidad_bajada_media_mean'],
                    'Desv. Estándar Bajada': top_tecnologias['velocidad_bajada_media_std']
                })

                if 'velocidad_subida_media_mean' in top_tecnologias.columns:
                    heatmap_data['Velocidad Subida (Mbps)'] = top_tecnologias['velocidad_subida_media_mean']
                    heatmap_data['Desv. Estándar Subida'] = top_tecnologias['velocidad_subida_media_std']

                # Crear gráfica
                plt.figure(figsize=(14, 8))

                # Crear tabla coloreada
                ax = plt.subplot(111, frame_on=False)
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)

                # Crear tabla coloreada basada en valores
                tabla_data = heatmap_data.set_index('Tecnología')
                tabla = ax.table(
                    cellText=np.round(tabla_data.values, 2),
                    rowLabels=tabla_data.index,
                    colLabels=tabla_data.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0.2, 0.2, 0.7, 0.6]
                )

                # Colorear celdas según valores (solo para columnas de velocidad)
                for i in range(len(tabla_data.index)):
                    for j in range(len(tabla_data.columns)):
                        if 'Velocidad' in tabla_data.columns[j] and 'Desv.' not in tabla_data.columns[j]:
                            cell = tabla[i+1, j]
                            val = tabla_data.iloc[i, j]
                            max_val = tabla_data[tabla_data.columns[j]].max()

                            # Normalize and create color
                            normalized = val / max_val
                            cell.set_facecolor(plt.cm.viridis(normalized))
                            cell.set_text_props(color='white' if normalized > 0.5 else 'black')

                # Ajustar tamaño de tabla
                tabla.auto_set_font_size(False)
                tabla.set_fontsize(10)
                tabla.scale(1.2, 1.5)

                plt.title('Velocidades de Internet por Tecnología (Sin Valores Atípicos)', fontsize=16)
                plt.tight_layout()

                # Guardar figura
                ruta_figura = os.path.join(self.carpeta_figuras, 'tecnologias_velocidad.png')
                plt.savefig(ruta_figura, dpi=300)
                plt.close()

                # Agregar al markdown
                self.agregar_a_markdown("\n### Velocidades por Tecnología\n")
                self.agregar_a_markdown(f"![Velocidades por Tecnología](figuras/tecnologias_velocidad.png)\n")
                self.agregar_a_markdown("Esta tabla muestra las velocidades promedio (bajada y subida) para las principales tecnologías de internet, "
                                        f"excluyendo valores atípicos. "
                                        f"La tecnología **{top_tecnologias.iloc[0]['tecnologia']}** ofrece la mayor velocidad de bajada con "
                                        f"**{top_tecnologias.iloc[0]['velocidad_bajada_media_mean']:.2f} Mbps**, mientras que "
                                        f"**{top_tecnologias.iloc[-1]['tecnologia']}** presenta la menor velocidad entre las tecnologías principales "
                                        f"con **{top_tecnologias.iloc[-1]['velocidad_bajada_media_mean']:.2f} Mbps**.")

                logger.info(f"Mapa de calor de tecnologías y velocidad guardado en {ruta_figura}")
            else:
                logger.warning("No se encontraron las columnas necesarias para generar el mapa de calor")

        except Exception as e:
            logger.error(f"Error al generar mapa de calor de tecnologías: {str(e)}")
            self.agregar_a_markdown(f"\n⚠️ No se pudo crear el mapa de calor de tecnologías debido a un error: {str(e)}")

    def generar_conclusiones(self) -> None:
        """
        Genera conclusiones basadas en los análisis realizados.
        """
        try:
            logger.info("Generando conclusiones del análisis")
            self.agregar_a_markdown("\n## Conclusiones del Análisis")

            self.agregar_a_markdown("\n*Nota: Las siguientes conclusiones se basan en el análisis realizado excluyendo valores atípicos "
                                    "como CAUCA_EL_TAMBO y CALDAS_BELALCÁZAR para obtener una representación más precisa.*\n")

            conclusiones = []

            # Conclusiones sobre tecnologías
            if self.df_tecnologias is not None:
                # Filtrar datos para excluir municipios atípicos
                df_tech_filtrado = self.df_tecnologias.copy()
                for municipio in self.municipios_excluir:
                    df_tech_filtrado = df_tech_filtrado[~df_tech_filtrado['municipio'].str.contains(municipio, case=False)]

                # Identificar tecnología más común
                if 'tecnologia' in df_tech_filtrado.columns:
                    tecnologia_comun = df_tech_filtrado['tecnologia'].value_counts().index[0]
                    conclusiones.append(f"- La tecnología de acceso a internet más utilizada en los municipios analizados es **{tecnologia_comun}**.")

                # Identificar tecnología con mayor velocidad
                if 'tecnologia' in df_tech_filtrado.columns and 'velocidad_bajada_media' in df_tech_filtrado.columns:
                    tech_velocidad = df_tech_filtrado.groupby('tecnologia')['velocidad_bajada_media'].mean()
                    mejor_tech = tech_velocidad.idxmax()
                    peor_tech = tech_velocidad.idxmin()
                    conclusiones.append(f"- La tecnología que ofrece mayor velocidad de bajada es **{mejor_tech}** con "
                                        f"**{tech_velocidad.max():.2f} Mbps** en promedio, mientras que **{peor_tech}** "
                                        f"ofrece la menor velocidad con **{tech_velocidad.min():.2f} Mbps**.")

            # Conclusiones sobre velocidades y tendencias
            if self.df_tendencias is not None:
                tendencias_bajada = self.df_tendencias[self.df_tendencias['tipo'] == 'bajada'].copy()

                # Filtrar municipios excluidos
                for municipio in self.municipios_excluir:
                    tendencias_bajada = tendencias_bajada[~tendencias_bajada['municipio'].str.contains(municipio, case=False)]

                # Eliminar otros outliers estadísticos
                tendencias_bajada = self.eliminar_outliers_estadisticos(tendencias_bajada, 'media', 2.5)
                tendencias_bajada = self.eliminar_outliers_estadisticos(tendencias_bajada, 'tendencia', 2.5)

                if len(tendencias_bajada) > 0:
                    # Porcentaje de municipios con tendencia positiva
                    municipios_mejorando = len(tendencias_bajada[tendencias_bajada['tendencia'] > 0])
                    pct_mejorando = (municipios_mejorando / len(tendencias_bajada)) * 100
                    conclusiones.append(f"- Un **{pct_mejorando:.1f}%** de los municipios muestra una tendencia positiva en la velocidad de internet, "
                                        f"lo que indica mejoras en la conectividad.")

                    # Municipio con mayor mejora
                    if 'tendencia' in tendencias_bajada.columns and len(tendencias_bajada) > 0:
                        mayor_mejora = tendencias_bajada.loc[tendencias_bajada['tendencia'].idxmax()]
                        mayor_deterioro = tendencias_bajada.loc[tendencias_bajada['tendencia'].idxmin()]

                        conclusiones.append(f"- El municipio con mayor mejora en velocidad es **{mayor_mejora['municipio']}** "
                                            f"con una tendencia de **+{mayor_mejora['tendencia']:.2f} Mbps**, mientras que "
                                            f"**{mayor_deterioro['municipio']}** muestra el mayor deterioro con una tendencia "
                                            f"de **{mayor_deterioro['tendencia']:.2f} Mbps**.")

                    # Velocidad promedio
                    velocidad_promedio = tendencias_bajada['media'].mean()
                    velocidad_mediana = tendencias_bajada['media'].median()
                    conclusiones.append(f"- La velocidad promedio de internet en los municipios analizados es de **{velocidad_promedio:.2f} Mbps**, "
                                        f"con una mediana de **{velocidad_mediana:.2f} Mbps**, lo que indica una distribución {('sesgada' if abs(velocidad_promedio - velocidad_mediana) > 1 else 'bastante simétrica')}.")

            # Agregar conclusiones al markdown
            if conclusiones:
                for conclusion in conclusiones:
                    self.agregar_a_markdown(conclusion + "\n")
            else:
                self.agregar_a_markdown("No se pudieron generar conclusiones significativas con los datos disponibles.")

            logger.info("Generación de conclusiones completada")

        except Exception as e:
            logger.error(f"Error al generar conclusiones: {str(e)}")
            self.agregar_a_markdown(f"\n⚠️ No se pudieron generar todas las conclusiones debido a un error: {str(e)}")

    def exportar_markdown(self) -> None:
        """
        Exporta todos los resultados a un archivo markdown.
        """
        try:
            # Crear encabezado del documento
            fecha_actual = datetime.now().strftime("%d/%m/%Y %H:%M")
            encabezado = [
                "# Análisis de Acceso a Internet en Colombia\n",
                f"*Informe generado automáticamente el {fecha_actual}*\n",
                "## Resumen Ejecutivo\n",
                "Este informe presenta un análisis completo del acceso a internet en diferentes municipios de Colombia, ",
                "incluyendo tecnologías utilizadas, velocidades promedio y tendencias de cambio. Las visualizaciones ",
                "muestran los municipios con mejor y peor conectividad, así como las tecnologías predominantes.\n\n",
                f"**Nota:** Se han excluido municipios con valores atípicos ({', '.join(self.municipios_excluir)}) para obtener ",
                "visualizaciones más representativas de la tendencia general en el país.\n"
            ]

            # Combinar todo el contenido
            contenido_completo = encabezado + self.resultados_markdown

            # Guardar archivo markdown
            ruta_markdown = os.path.join(self.carpeta_datos, 'informe_acceso_internet.md')
            with open(ruta_markdown, 'w', encoding='utf-8') as f:
                f.write('\n'.join(contenido_completo))

            logger.info(f"Informe markdown guardado en {ruta_markdown}")

        except Exception as e:
            logger.error(f"Error al exportar markdown: {str(e)}")

# Importar scipy.stats para el método de eliminación de outliers
import scipy.stats as stats

# Función principal para ejecutar el análisis
def ejecutar_analisis_completo(carpeta_datos="./resultados", nivel_debug='INFO'):
    """
    Ejecuta el análisis completo de los datos de acceso a internet.

    Args:
        carpeta_datos: Ruta a la carpeta con los archivos JSON
        nivel_debug: Nivel de registro ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    try:
        logger.info("Iniciando análisis completo de acceso a internet")
        logger.info(f"Se utilizará la carpeta de datos: {carpeta_datos}")
        logger.info(f"Se excluirán municipios con valores atípicos: CAUCA_EL_TAMBO, CALDAS_BELALCÁZAR")

        # Crear instancia del analizador
        analizador = AnalizadorAccesoInternet(carpeta_datos, nivel_debug)

        # Cargar datos JSON
        analizador.cargar_datos_json()

        # Realizar análisis
        analizador.analizar_por_tecnologia()
        analizador.analizar_por_velocidad()

        # Generar gráficas de mejores y peores municipios
        analizador.generar_grafica_mejores_peores(n=10)

        # Generar mapa de calor de tecnologías
        analizador.generar_mapa_calor_tecnologias()

        # Generar conclusiones
        analizador.generar_conclusiones()

        # Exportar resultados a markdown
        analizador.exportar_markdown()

        logger.info("Análisis completo finalizado con éxito")

    except Exception as e:
        logger.error(f"Error durante el análisis completo: {str(e)}")
        raise



# Si se ejecuta como script principal
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Analizar datos de acceso a internet desde archivos JSON y exportar a Markdown.')
    parser.add_argument('--carpeta', type=str, default="./resultados",
                        help='Carpeta con archivos JSON (default: ./resultados)')
    parser.add_argument('--debug', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Nivel de depuración (default: INFO)')
    args = parser.parse_args()

    ejecutar_analisis_completo(args.carpeta, args.debug)