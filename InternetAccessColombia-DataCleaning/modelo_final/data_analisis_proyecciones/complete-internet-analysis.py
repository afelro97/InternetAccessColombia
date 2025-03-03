import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RANSACRegressor, HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import logging
from datetime import datetime
import glob
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import sys

#%%
# Try to import tqdm
try:
    from tqdm.notebook import tqdm as notebook_tqdm
    from tqdm import tqdm as terminal_tqdm
    has_tqdm = True
except ImportError:
    has_tqdm = False
    # Create dummy tqdm function
    def notebook_tqdm(iterable, **kwargs):
        return iterable
    terminal_tqdm = notebook_tqdm

#%%
# Determine if running in Jupyter
def is_jupyter():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type
    except NameError:
        return False      # Probably standard Python interpreter

# Set appropriate tqdm based on environment
if is_jupyter() and has_tqdm:
    tqdm = notebook_tqdm
else:
    tqdm = terminal_tqdm

#%%
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"internet_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("internet_analysis")

def setup_logging(output_dir, debug=False):
    """Configure logging with custom settings"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    logging.basicConfig(
        level=logging.INFO if debug else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, f"internet_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("internet_analysis")

#%%
class DataValidator:
    """Class to validate the dataset before analysis"""
    
    @staticmethod
    def validate_dataset(df, debug=False):
        """
        Validates the dataset for required columns and data quality
        
        Args:
            df (DataFrame): Dataset to validate
            debug (bool): Flag to enable debug output
            
        Returns:
            tuple: (is_valid, issues_found)
        """
        issues = []
        required_columns = ['TRIMESTRE', 'VELOCIDAD BAJADA', 'VELOCIDAD SUBIDA', 
                           'No. ACCESOS FIJOS A INTERNET', 'TECNOLOGÍA']
        
        # Check if DataFrame is empty
        if df.empty:
            issues.append("Dataset is empty")
            return False, issues
            
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
            return False, issues
            
        # Check for null values in critical columns
        for col in required_columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                issues.append(f"Column {col} has {null_count} null values")
                
        # Check for data type consistency
        try:
            # Ensure numeric columns are actually numeric
            pd.to_numeric(df['VELOCIDAD BAJADA'])
            pd.to_numeric(df['VELOCIDAD SUBIDA'])
            pd.to_numeric(df['No. ACCESOS FIJOS A INTERNET'])
            
            # Ensure TRIMESTRE is numeric and sequential
            trimesters = pd.to_numeric(df['TRIMESTRE']).sort_values().unique()
            if len(trimesters) < 2:
                issues.append("Not enough unique time periods for trend analysis")
                # This is a special case - not a critical validation error
                # We'll handle it separately in procesar_municipio
                
        except Exception as e:
            issues.append(f"Data type consistency error: {str(e)}")
            
        # Check for negative values in metrics that should be positive
        if (df['VELOCIDAD BAJADA'] < 0).any():
            issues.append("Found negative values in VELOCIDAD BAJADA")
        if (df['VELOCIDAD SUBIDA'] < 0).any():
            issues.append("Found negative values in VELOCIDAD SUBIDA")
        if (df['No. ACCESOS FIJOS A INTERNET'] < 0).any():
            issues.append("Found negative values in No. ACCESOS FIJOS A INTERNET")
            
        # For trend analysis timing issue, we'll treat this differently
        # We'll consider it valid but flag the issue
        critical_issues = [issue for issue in issues 
                           if "Not enough unique time periods" not in issue]
        
        is_valid = len(critical_issues) == 0
        
        if debug and issues:
            logger.warning(f"Data validation issues found: {issues}")
            
        return is_valid, issues

#%%
class ModelSelector:
    """Class to select and train prediction models"""
    
    @staticmethod
    def train_model(X, y, model_type='linear', debug=False):
        """
        Trains a model based on the specified type
        
        Args:
            X (array): Features (typically time/trimesters)
            y (array): Target values to predict
            model_type (str): Type of model ('linear', 'ransac', 'huber', 'forest')
            debug (bool): Flag to enable debug output
            
        Returns:
            tuple: (model, metrics) where metrics is a dict of performance metrics
        """
        if len(X) < 3:
            if debug:
                logger.warning(f"Not enough data points ({len(X)}) for reliable model training")
            return None, {"error": "Not enough data points"}
            
        # Reshape X if it's 1D
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        # Dictionary to map model_type to actual model
        models = {
            'linear': LinearRegression(),
            'ransac': RANSACRegressor(random_state=42),
            'huber': HuberRegressor(epsilon=1.35),
            'forest': RandomForestRegressor(n_estimators=50, random_state=42)
        }
        
        try:
            # Get the model
            model = models.get(model_type, models['linear'])
            
            # Fit the model
            model.fit(X, y)
            
            # Make predictions
            y_pred = model.predict(X)
            
            # Calculate metrics
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            
            # Get slope if it's a linear model
            slope = None
            if hasattr(model, 'coef_'):
                slope = model.coef_[0]
            
            metrics = {
                'r2': r2,
                'mse': mse,
                'rmse': rmse,
                'slope': slope,
                'model_type': model_type
            }
            
            if debug:
                logger.info(f"Model trained: {model_type}, R² = {r2:.4f}, RMSE = {rmse:.4f}")
                if slope is not None:
                    logger.info(f"Slope: {slope:.4f}")
            
            return model, metrics
            
        except Exception as e:
            if debug:
                logger.error(f"Error training model: {str(e)}")
            return None, {"error": str(e)}

#%%
def analizar_tecnologias(df, output_dir='output', debug=False):
    """
    Analyze technology trends by municipality
    
    Args:
        df (DataFrame): Dataset to analyze
        output_dir (str): Directory to save output files
        debug (bool): Flag to enable debug output
        
    Returns:
        tuple: (stats_dataframe, plot_path, trends_dict)
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Extract location from DataFrame or use a default
    ubicacion = df['UBICACION'].iloc[0] if 'UBICACION' in df.columns else 'Ubicación No Especificada'
    
    # Validate dataset
    validator = DataValidator()
    is_valid, issues = validator.validate_dataset(df, debug=debug)
    
    if not is_valid:
        logger.error(f"Invalid dataset for {ubicacion}: {issues}")
        return None, None, None
    
    # Check if dataset has enough time series points
    num_trimestres = df['TRIMESTRE'].nunique()
    if num_trimestres < 2:
        if debug:
            logger.warning(f"{ubicacion}: Not enough time series points for technology analysis (found {num_trimestres})")
        return None, None, None
    
    if debug:
        logger.info(f"Analyzing technologies for {ubicacion}")
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Unique technologies: {df['TECNOLOGÍA'].unique()}")
        logger.info(f"Time periods: {df['TRIMESTRE'].unique()}")
    
    try:
        # Calculate statistics by trimester and technology
        stats_por_trimestre = df.groupby(['TRIMESTRE', 'TECNOLOGÍA']).agg({
            'VELOCIDAD BAJADA': ['mean', 'std'],
            'VELOCIDAD SUBIDA': ['mean', 'std'],
            'No. ACCESOS FIJOS A INTERNET': ['mean', 'std']
        }).reset_index()
        
        # Rename columns for clarity
        stats_por_trimestre.columns = [
            'TRIMESTRE', 'TECNOLOGÍA',
            'VELOCIDAD_BAJADA_MEAN', 'VELOCIDAD_BAJADA_STD',
            'VELOCIDAD_SUBIDA_MEAN', 'VELOCIDAD_SUBIDA_STD',
            'ACCESOS_MEAN', 'ACCESOS_STD'
        ]
        
        # Replace NaN with 0 for std when there's only one data point
        stats_por_trimestre = stats_por_trimestre.fillna(0)
        
        if debug:
            logger.info(f"Generated stats table with shape: {stats_por_trimestre.shape}")
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Dictionary to store model results
        tendencias_tecnologia = {}
        
        # Graficar accesos por tecnología
        for tecnologia in df['TECNOLOGÍA'].unique():
            datos_tech = stats_por_trimestre[stats_por_trimestre['TECNOLOGÍA'] == tecnologia]
            
            # Skip if not enough data points
            if len(datos_tech) <= 1:
                if debug:
                    logger.warning(f"Not enough data points for {tecnologia} in {ubicacion}")
                continue
                
            # Plot data points
            ax1.errorbar(datos_tech['TRIMESTRE'], datos_tech['ACCESOS_MEAN'],
                         yerr=datos_tech['ACCESOS_STD'], fmt='o-', label=f'{tecnologia}')
            
            # Train regression model
            X = datos_tech['TRIMESTRE'].values
            y = datos_tech['ACCESOS_MEAN'].values
            
            # Try different models and pick the best one
            model_types = ['linear', 'ransac', 'huber']
            best_model = None
            best_metrics = {"r2": -float('inf')}
            
            for model_type in model_types:
                model, metrics = ModelSelector.train_model(X, y, model_type=model_type, debug=debug)
                if model is not None and metrics.get('r2', -float('inf')) > best_metrics['r2']:
                    best_model = model
                    best_metrics = metrics
            
            if best_model is not None:
                X_reshaped = X.reshape(-1, 1)
                y_pred = best_model.predict(X_reshaped)
                ax1.plot(X, y_pred, '--', alpha=0.5)
                
                # Store in tendencies dictionary
                tendencias_tecnologia[tecnologia] = {
                    'model_accesos': best_model,
                    **best_metrics,
                    'accesos_media': datos_tech['ACCESOS_MEAN'].mean(),
                    'accesos_std': datos_tech['ACCESOS_STD'].mean()
                }
                
                if debug:
                    logger.info(f"Technology {tecnologia} - Access model: {best_metrics['model_type']}, R² = {best_metrics['r2']:.4f}")
        
        ax1.set_title(f'Accesos por Tecnología en {ubicacion}')
        ax1.set_xlabel('Trimestre')
        ax1.set_ylabel('Número de Accesos (Media ± Std)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True)
        # Use log scale only if data spans multiple orders of magnitude
        if df['No. ACCESOS FIJOS A INTERNET'].max() / max(df['No. ACCESOS FIJOS A INTERNET'].min(), 1) > 100:
            ax1.set_yscale('log')
            
        # Graficar velocidades por tecnología
        for tecnologia in df['TECNOLOGÍA'].unique():
            datos_tech = stats_por_trimestre[stats_por_trimestre['TECNOLOGÍA'] == tecnologia]
            
            # Skip if not enough data points
            if len(datos_tech) <= 1:
                continue
                
            # Plot data points
            ax2.errorbar(datos_tech['TRIMESTRE'], datos_tech['VELOCIDAD_BAJADA_MEAN'],
                         yerr=datos_tech['VELOCIDAD_BAJADA_STD'], fmt='o-',
                         label=f'{tecnologia} - Bajada')
            
            # Train regression model
            X = datos_tech['TRIMESTRE'].values
            y = datos_tech['VELOCIDAD_BAJADA_MEAN'].values
            
            # Try different models and pick the best one
            model_types = ['linear', 'ransac', 'huber']
            best_model = None
            best_metrics = {"r2": -float('inf')}
            
            for model_type in model_types:
                model, metrics = ModelSelector.train_model(X, y, model_type=model_type, debug=debug)
                if model is not None and metrics.get('r2', -float('inf')) > best_metrics['r2']:
                    best_model = model
                    best_metrics = metrics
            
            if best_model is not None:
                X_reshaped = X.reshape(-1, 1)
                y_pred = best_model.predict(X_reshaped)
                ax2.plot(X, y_pred, '--', alpha=0.5)
                
                # Update tendencies dictionary
                if tecnologia in tendencias_tecnologia:
                    tendencias_tecnologia[tecnologia].update({
                        'model_velocidad': best_model,
                        'r2_velocidad': best_metrics['r2'],
                        'velocidad_media': datos_tech['VELOCIDAD_BAJADA_MEAN'].mean(),
                        'velocidad_std': datos_tech['VELOCIDAD_BAJADA_STD'].mean()
                    })
                else:
                    tendencias_tecnologia[tecnologia] = {
                        'model_velocidad': best_model,
                        'r2_velocidad': best_metrics['r2'],
                        'velocidad_media': datos_tech['VELOCIDAD_BAJADA_MEAN'].mean(),
                        'velocidad_std': datos_tech['VELOCIDAD_BAJADA_STD'].mean()
                    }
                
                if debug:
                    logger.info(f"Technology {tecnologia} - Speed model: {best_metrics['model_type']}, R² = {best_metrics['r2']:.4f}")
        
        ax2.set_title(f'Velocidades por Tecnología en {ubicacion}')
        ax2.set_xlabel('Trimestre')
        ax2.set_ylabel('Velocidad Bajada (Mbps) (Media ± Std)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True)
        # Use log scale only if data spans multiple orders of magnitude
        if df['VELOCIDAD BAJADA'].max() / max(df['VELOCIDAD BAJADA'].min(), 1) > 100:
            ax2.set_yscale('log')
            
        plt.tight_layout()
        
        # Save the figure
        safe_ubicacion = ubicacion.replace(".", "_").replace(" ", "_").replace("/", "_")
        plot_path = os.path.join(output_dir, f"{safe_ubicacion}_tecnologias.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        if debug:
            logger.info(f"Plot saved to {plot_path}")
            
        return stats_por_trimestre, plot_path, tendencias_tecnologia
        
    except Exception as e:
        if debug:
            logger.error(f"Error in analizar_tecnologias for {ubicacion}: {str(e)}")
        return None, None, None

#%%
def analizar_velocidades(df, output_dir='output', debug=False):
    """
    Analyze velocity trends by municipality
    
    Args:
        df (DataFrame): Dataset to analyze
        output_dir (str): Directory to save output files
        debug (bool): Flag to enable debug output
        
    Returns:
        tuple: (stats_dataframe, plot_path, trends_dict)
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Extract location from DataFrame or use a default
    ubicacion = df['UBICACION'].iloc[0] if 'UBICACION' in df.columns else 'Ubicación No Especificada'
    
    # Validate dataset
    validator = DataValidator()
    is_valid, issues = validator.validate_dataset(df, debug=debug)
    
    if not is_valid:
        logger.error(f"Invalid dataset for {ubicacion}: {issues}")
        return None, None, None
    
    # Check if dataset has enough time series points
    num_trimestres = df['TRIMESTRE'].nunique()
    if num_trimestres < 2:
        if debug:
            logger.warning(f"{ubicacion}: Not enough time series points for velocity analysis (found {num_trimestres})")
        return None, None, None
    
    if debug:
        logger.info(f"Analyzing velocities for {ubicacion}")
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Time periods: {df['TRIMESTRE'].unique()}")
    
    try:
        # Calculate statistics by trimester, focusing on velocities
        stats_por_trimestre = df.groupby(['TRIMESTRE']).agg({
            'VELOCIDAD BAJADA': ['mean', 'std', 'min', 'max'],
            'VELOCIDAD SUBIDA': ['mean', 'std', 'min', 'max'],
            'No. ACCESOS FIJOS A INTERNET': ['sum']
        }).reset_index()
        
        # Rename columns for clarity
        stats_por_trimestre.columns = [
            'TRIMESTRE',
            'VELOCIDAD_BAJADA_MEAN', 'VELOCIDAD_BAJADA_STD', 'VELOCIDAD_BAJADA_MIN', 'VELOCIDAD_BAJADA_MAX',
            'VELOCIDAD_SUBIDA_MEAN', 'VELOCIDAD_SUBIDA_STD', 'VELOCIDAD_SUBIDA_MIN', 'VELOCIDAD_SUBIDA_MAX',
            'TOTAL_ACCESOS'
        ]
        
        # Replace NaN with 0 for std when there's only one data point
        stats_por_trimestre = stats_por_trimestre.fillna(0)
        
        if debug:
            logger.info(f"Generated velocity stats table with shape: {stats_por_trimestre.shape}")
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot download speed trends
        trimestres = stats_por_trimestre['TRIMESTRE'].values
        velocidad_bajada = stats_por_trimestre['VELOCIDAD_BAJADA_MEAN'].values
        velocidad_bajada_std = stats_por_trimestre['VELOCIDAD_BAJADA_STD'].values
        
        # Plot data points for download speed
        ax1.errorbar(trimestres, velocidad_bajada, yerr=velocidad_bajada_std, 
                     fmt='o-', color='blue', label='Velocidad Bajada')
        ax1.fill_between(trimestres, 
                        stats_por_trimestre['VELOCIDAD_BAJADA_MIN'], 
                        stats_por_trimestre['VELOCIDAD_BAJADA_MAX'], 
                        alpha=0.2, color='blue')
        
        # Train model for download speed
        if len(trimestres) > 2:
            # Try different models and pick the best one
            model_types = ['linear', 'ransac', 'huber']
            best_model = None
            best_metrics = {"r2": -float('inf')}
            
            for model_type in model_types:
                model, metrics = ModelSelector.train_model(
                    trimestres, velocidad_bajada, model_type=model_type, debug=debug
                )
                if model is not None and metrics.get('r2', -float('inf')) > best_metrics['r2']:
                    best_model = model
                    best_metrics = metrics
            
            if best_model is not None:
                # Predict for visualization
                X_pred = np.linspace(min(trimestres), max(trimestres), 100).reshape(-1, 1)
                y_pred = best_model.predict(X_pred)
                ax1.plot(X_pred, y_pred, '--', color='blue', alpha=0.7)
                
                # Add projection for next two trimesters
                next_trimesters = np.array([max(trimestres) + 1, max(trimestres) + 2]).reshape(-1, 1)
                future_preds = best_model.predict(next_trimesters)
                ax1.plot(next_trimesters.flatten(), future_preds, 'o--', color='red', alpha=0.7, 
                        label='Predicción (2 trimestres)')
                
                # Add model info to plot
                model_info = f"Modelo: {best_metrics['model_type'].capitalize()}, R² = {best_metrics['r2']:.3f}"
                if best_metrics.get('slope') is not None:
                    model_info += f", Pendiente = {best_metrics['slope']:.2f} Mbps/trimestre"
                ax1.text(0.02, 0.95, model_info, transform=ax1.transAxes, 
                        bbox=dict(facecolor='white', alpha=0.8))
                
                if debug:
                    logger.info(f"Download speed model: {best_metrics['model_type']}, R² = {best_metrics['r2']:.4f}")
                    if best_metrics.get('slope') is not None:
                        logger.info(f"Speed slope: {best_metrics['slope']:.4f} Mbps/trimestre")
        
        ax1.set_title(f'Tendencia de Velocidad de Bajada en {ubicacion}')
        ax1.set_xlabel('Trimestre')
        ax1.set_ylabel('Velocidad Bajada (Mbps)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot upload speed trends
        velocidad_subida = stats_por_trimestre['VELOCIDAD_SUBIDA_MEAN'].values
        velocidad_subida_std = stats_por_trimestre['VELOCIDAD_SUBIDA_STD'].values
        
        # Plot data points for upload speed
        ax2.errorbar(trimestres, velocidad_subida, yerr=velocidad_subida_std, 
                     fmt='o-', color='green', label='Velocidad Subida')
        ax2.fill_between(trimestres, 
                        stats_por_trimestre['VELOCIDAD_SUBIDA_MIN'], 
                        stats_por_trimestre['VELOCIDAD_SUBIDA_MAX'], 
                        alpha=0.2, color='green')
        
        # Train model for upload speed
        if len(trimestres) > 2:
            # Try different models and pick the best one
            model_types = ['linear', 'ransac', 'huber']
            best_model = None
            best_metrics = {"r2": -float('inf')}
            
            for model_type in model_types:
                model, metrics = ModelSelector.train_model(
                    trimestres, velocidad_subida, model_type=model_type, debug=debug
                )
                if model is not None and metrics.get('r2', -float('inf')) > best_metrics['r2']:
                    best_model = model
                    best_metrics = metrics
            
            if best_model is not None:
                # Predict for visualization
                X_pred = np.linspace(min(trimestres), max(trimestres), 100).reshape(-1, 1)
                y_pred = best_model.predict(X_pred)
                ax2.plot(X_pred, y_pred, '--', color='green', alpha=0.7)
                
                # Add projection for next two trimesters
                next_trimesters = np.array([max(trimestres) + 1, max(trimestres) + 2]).reshape(-1, 1)
                future_preds = best_model.predict(next_trimesters)
                ax2.plot(next_trimesters.flatten(), future_preds, 'o--', color='red', alpha=0.7, 
                        label='Predicción (2 trimestres)')
                
                # Add model info to plot
                model_info = f"Modelo: {best_metrics['model_type'].capitalize()}, R² = {best_metrics['r2']:.3f}"
                if best_metrics.get('slope') is not None:
                    model_info += f", Pendiente = {best_metrics['slope']:.2f} Mbps/trimestre"
                ax2.text(0.02, 0.95, model_info, transform=ax2.transAxes, 
                        bbox=dict(facecolor='white', alpha=0.8))
                
                if debug:
                    logger.info(f"Upload speed model: {best_metrics['model_type']}, R² = {best_metrics['r2']:.4f}")
                    if best_metrics.get('slope') is not None:
                        logger.info(f"Speed slope: {best_metrics['slope']:.4f} Mbps/trimestre")
        
        ax2.set_title(f'Tendencia de Velocidad de Subida en {ubicacion}')
        ax2.set_xlabel('Trimestre')
        ax2.set_ylabel('Velocidad Subida (Mbps)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save the figure
        safe_ubicacion = ubicacion.replace(".", "_").replace(" ", "_").replace("/", "_")
        plot_path = os.path.join(output_dir, f"{safe_ubicacion}_velocidades.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        if debug:
            logger.info(f"Velocity plot saved to {plot_path}")
            
        # Create dictionary with trend analysis results
        tendencias_velocidad = {
            'bajada': {
                'media': np.mean(velocidad_bajada),
                'tendencia': best_metrics.get('slope', 0) if 'best_metrics' in locals() else 0,
                'r2': best_metrics.get('r2', 0) if 'best_metrics' in locals() else 0,
                'modelo': best_metrics.get('model_type', 'N/A') if 'best_metrics' in locals() else 'N/A',
                'prediccion_2_trimestres': future_preds.tolist() if 'future_preds' in locals() else []
            },
            'subida': {
                'media': np.mean(velocidad_subida),
                'tendencia': best_metrics.get('slope', 0) if 'best_metrics' in locals() else 0,
                'r2': best_metrics.get('r2', 0) if 'best_metrics' in locals() else 0,
                'modelo': best_metrics.get('model_type', 'N/A') if 'best_metrics' in locals() else 'N/A',
                'prediccion_2_trimestres': future_preds.tolist() if 'future_preds' in locals() else []
            }
        }
        
        return stats_por_trimestre, plot_path, tendencias_velocidad
        
    except Exception as e:
        if debug:
            logger.error(f"Error in analizar_velocidades for {ubicacion}: {str(e)}")
        return None, None, None

#%%
def procesar_municipio(ruta_archivo, output_dir='output', debug=False):
    """
    Process a single municipality file with both analysis methods
    
    Args:
        ruta_archivo (str): Path to the municipality CSV file
        output_dir (str): Directory to save output files
        debug (bool): Flag to enable debug output
        
    Returns:
        dict: Results of both analyses
    """
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Load the dataset
        df = pd.read_csv(ruta_archivo)
        
        # Extract municipality name from filename
        nombre_municipio = os.path.basename(ruta_archivo).replace('.csv', '')
        
        if debug:
            logger.info(f"Processing municipality: {nombre_municipio}")
            logger.info(f"Data shape: {df.shape}")
        
        # Add UBICACION column if not present
        if 'UBICACION' not in df.columns:
            df['UBICACION'] = nombre_municipio
            
        # Check if dataset has enough unique time series points
        num_trimestres = df['TRIMESTRE'].nunique()
        
        # Create basic result structure that will be returned regardless of outcome
        results = {
            'municipio': nombre_municipio,
            'status': 'success',  # Will be updated if needed
            'tecnologias': {
                'stats': None,
                'plot_path': None,
                'trends': None
            },
            'velocidades': {
                'stats': None,
                'plot_path': None,
                'trends': None
            }
        }
        
        if num_trimestres < 2:
            warning_msg = "Not enough data: dataset contains only one unique time series point"
            logger.warning(f"{nombre_municipio}: {warning_msg}")
            
            # Create basic stats even without time series
            results.update({
                'status': 'warning',
                'message': warning_msg,
                'tecnologias': {
                    'stats': df.groupby('TECNOLOGÍA').agg({
                        'VELOCIDAD BAJADA': 'mean', 
                        'VELOCIDAD SUBIDA': 'mean',
                        'No. ACCESOS FIJOS A INTERNET': 'sum'
                    }).reset_index().to_dict('records') if 'TECNOLOGÍA' in df.columns else None,
                    'plot_path': None,
                    'trends': None
                },
                'velocidades': {
                    'stats': {col: df[col].agg(['mean', 'min', 'max']).to_dict() 
                             for col in ['VELOCIDAD BAJADA', 'VELOCIDAD SUBIDA']}
                             if not df.empty else None,
                    'plot_path': None,
                    'trends': None
                }
            })
        else:
            # Run both analyses
            try:
                tech_stats, tech_plot, tech_trends = analizar_tecnologias(df, output_dir=output_dir, debug=debug)
                if tech_stats is not None:
                    results['tecnologias']['stats'] = tech_stats.to_dict('records')
                    results['tecnologias']['plot_path'] = tech_plot
                    results['tecnologias']['trends'] = tech_trends
            except Exception as e:
                if debug:
                    logger.error(f"Error in technology analysis for {nombre_municipio}: {str(e)}")
                results['status'] = 'partial'
                results['tech_error'] = str(e)
            
            try:
                vel_stats, vel_plot, vel_trends = analizar_velocidades(df, output_dir=output_dir, debug=debug)
                if vel_stats is not None:
                    results['velocidades']['stats'] = vel_stats.to_dict('records')
                    results['velocidades']['plot_path'] = vel_plot
                    results['velocidades']['trends'] = vel_trends
            except Exception as e:
                if debug:
                    logger.error(f"Error in velocity analysis for {nombre_municipio}: {str(e)}")
                results['status'] = 'partial'
                results['vel_error'] = str(e)
        
        # Save results to JSON
        import json
        results_file = os.path.join(output_dir, f"{nombre_municipio.replace('.', '_').replace(' ', '_').replace('-', '_')}_results.json")
        
        # We need to remove model objects which are not JSON serializable
        serializable_results = results.copy()
        if serializable_results['tecnologias']['trends']:
            for tech, trend_data in serializable_results['tecnologias']['trends'].items():
                if 'model_accesos' in trend_data:
                    del trend_data['model_accesos']
                if 'model_velocidad' in trend_data:
                    del trend_data['model_velocidad']
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        if debug:
            logger.info(f"Results saved to {results_file}")
        
        return results
    
    except Exception as e:
        if debug:
            logger.error(f"Critical error processing {ruta_archivo}: {str(e)}")
        
        # Try to create a minimal result file even in case of error
        try:
            nombre_municipio = os.path.basename(ruta_archivo).replace('.csv', '')
            error_results = {
                'municipio': nombre_municipio,
                'status': 'error',
                'error': str(e)
            }
            
            import json
            error_file = os.path.join(output_dir, f"{nombre_municipio.replace('.', '_').replace(' ', '_').replace('-', '_')}_results.json")
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(error_results, f, ensure_ascii=False, indent=2)
                
            if debug:
                logger.info(f"Error results saved to {error_file}")
        except Exception as inner_e:
            if debug:
                logger.error(f"Failed to save error results: {str(inner_e)}")
        
        return {
            'municipio': os.path.basename(ruta_archivo).replace('.csv', ''),
            'status': 'error',
            'error': str(e)
        }

#%% 
# New functions for batch processing
def procesar_todos_municipios(data_dir, output_dir='resultados', 
                             parallel=True, n_workers=None, debug=False):
    """
    Process all municipality CSV files in the specified directory
    
    Args:
        data_dir (str): Directory containing municipality CSV files
        output_dir (str): Directory to save output files
        parallel (bool): Whether to process files in parallel
        n_workers (int): Number of worker processes to use (None = auto)
        debug (bool): Flag to enable debug output
        
    Returns:
        dict: Summary of processing results
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Find all CSV files in the directory
    pattern = os.path.join(data_dir, '**', '*.csv')
    csv_files = glob.glob(pattern, recursive=True)
    
    if debug:
        logger.info(f"Found {len(csv_files)} CSV files in {data_dir}")
        
    if len(csv_files) == 0:
        logger.warning(f"No CSV files found in {data_dir}")
        return {"status": "error", "message": "No CSV files found"}
    
    # Initialize results
    results_summary = {
        "total_files": len(csv_files),
        "success": 0,
        "warning": 0,
        "partial": 0,
        "error": 0,
        "start_time": time.time(),
        "end_time": None,
        "processing_time": None,
        "results": {}
    }
    
    # Process files
    if parallel and len(csv_files) > 1:
        # Determine number of workers
        if n_workers is None:
            n_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
        
        if debug:
            logger.info(f"Processing {len(csv_files)} files in parallel with {n_workers} workers")
            
        # Process files in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(procesar_municipio, csv_file, output_dir, debug): csv_file 
                for csv_file in csv_files
            }
            
            # Process results as they complete
            for i, future in enumerate(tqdm(as_completed(future_to_file), total=len(csv_files), 
                                         desc="Processing municipalities", disable=not debug)):
                csv_file = future_to_file[future]
                try:
                    result = future.result()
                    municipio = result.get('municipio', os.path.basename(csv_file).replace('.csv', ''))
                    
                    # Update statistics
                    status = result.get('status', 'error')
                    results_summary[status] = results_summary.get(status, 0) + 1
                    
                    # Store result
                    results_summary['results'][municipio] = {
                        'status': status,
                        'file': csv_file
                    }
                    
                    # Log progress
                    if debug:
                        logger.info(f"Processed {municipio}: {status}")
                        
                except Exception as e:
                    municipio = os.path.basename(csv_file).replace('.csv', '')
                    logger.error(f"Error processing {municipio}: {str(e)}")
                    
                    # Update statistics
                    results_summary['error'] += 1
                    
                    # Store error
                    results_summary['results'][municipio] = {
                        'status': 'error',
                        'file': csv_file,
                        'error': str(e)
                    }
    else:
        # Process files sequentially
        if debug:
            logger.info(f"Processing {len(csv_files)} files sequentially")
            
        for i, csv_file in enumerate(tqdm(csv_files, desc="Processing municipalities", disable=not debug)):
            try:
                # Process the municipality
                result = procesar_municipio(csv_file, output_dir, debug)
                municipio = result.get('municipio', os.path.basename(csv_file).replace('.csv', ''))
                
                # Update statistics
                status = result.get('status', 'error')
                results_summary[status] = results_summary.get(status, 0) + 1
                
                # Store result
                results_summary['results'][municipio] = {
                    'status': status,
                    'file': csv_file
                }
                
                # Log progress
                if debug:
                    logger.info(f"Processed {municipio}: {status}")
                    
            except Exception as e:
                municipio = os.path.basename(csv_file).replace('.csv', '')
                logger.error(f"Error processing {municipio}: {str(e)}")
                
                # Update statistics
                results_summary['error'] += 1
                
                # Store error
                results_summary['results'][municipio] = {
                    'status': 'error',
                    'file': csv_file,
                    'error': str(e)
                }
    
    # Finalize results
    results_summary['end_time'] = time.time()
    results_summary['processing_time'] = results_summary['end_time'] - results_summary['start_time']
    
    # Save summary
    summary_file = os.path.join(output_dir, 'processing_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
        
    if debug:
        logger.info(f"Processing complete. Summary saved to {summary_file}")
        logger.info(f"Results: Success={results_summary['success']}, "
                   f"Warning={results_summary['warning']}, "
                   f"Partial={results_summary['partial']}, "
                   f"Error={results_summary['error']}")
        logger.info(f"Total processing time: {results_summary['processing_time']:.1f}s")
        
    return results_summary

#%%
def generar_reporte_consolidado(output_dir, debug=False):
    """
    Generate a consolidated report of all processed municipalities
    
    Args:
        output_dir (str): Directory containing municipality result files
        debug (bool): Flag to enable debug output
        
    Returns:
        dict: Consolidated statistics
    """
    if debug:
        logger.info(f"Generating consolidated report from {output_dir}")
        
    # Find all JSON result files
    result_files = glob.glob(os.path.join(output_dir, '*_results.json'))
    
    if debug:
        logger.info(f"Found {len(result_files)} result files")
        
    if len(result_files) == 0:
        logger.warning(f"No result files found in {output_dir}")
        return {"status": "error", "message": "No result files found"}
    
    # Initialize consolidated statistics
    consolidated = {
        "municipios_count": len(result_files),
        "tecnologias": {},
        "velocidades": {
            "bajada": {
                "media_general": [],
                "tendencia_media": [],
                "r2_media": []
            },
            "subida": {
                "media_general": [],
                "tendencia_media": [],
                "r2_media": []
            }
        },
        "status": {
            "success": 0,
            "warning": 0,
            "partial": 0,
            "error": 0
        }
    }
    
    # Process each result file
    for result_file in tqdm(result_files, desc="Analyzing results", disable=not debug):
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
                
            # Update status count
            status = result.get('status', 'error')
            consolidated["status"][status] = consolidated["status"].get(status, 0) + 1
            
            # Skip files with errors
            if status == 'error':
                continue
                
            # Process technology trends
            if 'tecnologias' in result and result['tecnologias'].get('trends'):
                for tech, trends in result['tecnologias']['trends'].items():
                    if tech not in consolidated["tecnologias"]:
                        consolidated["tecnologias"][tech] = {
                            "accesos_media": [],
                            "velocidad_media": [],
                            "r2_accesos": [],
                            "r2_velocidad": []
                        }
                    
                    # Collect statistics
                    if 'accesos_media' in trends:
                        consolidated["tecnologias"][tech]["accesos_media"].append(trends['accesos_media'])
                    if 'velocidad_media' in trends:
                        consolidated["tecnologias"][tech]["velocidad_media"].append(trends['velocidad_media'])
                    if 'r2' in trends:
                        consolidated["tecnologias"][tech]["r2_accesos"].append(trends['r2'])
                    if 'r2_velocidad' in trends:
                        consolidated["tecnologias"][tech]["r2_velocidad"].append(trends['r2_velocidad'])
            
            # Process velocity trends
            if 'velocidades' in result and result['velocidades'].get('trends'):
                vel_trends = result['velocidades']['trends']
                
                # Collect bajada statistics
                if 'bajada' in vel_trends:
                    if 'media' in vel_trends['bajada']:
                        consolidated["velocidades"]["bajada"]["media_general"].append(vel_trends['bajada']['media'])
                    if 'tendencia' in vel_trends['bajada']:
                        consolidated["velocidades"]["bajada"]["tendencia_media"].append(vel_trends['bajada']['tendencia'])
                    if 'r2' in vel_trends['bajada']:
                        consolidated["velocidades"]["bajada"]["r2_media"].append(vel_trends['bajada']['r2'])
                
                # Collect subida statistics
                if 'subida' in vel_trends:
                    if 'media' in vel_trends['subida']:
                        consolidated["velocidades"]["subida"]["media_general"].append(vel_trends['subida']['media'])
                    if 'tendencia' in vel_trends['subida']:
                        consolidated["velocidades"]["subida"]["tendencia_media"].append(vel_trends['subida']['tendencia'])
                    if 'r2' in vel_trends['subida']:
                        consolidated["velocidades"]["subida"]["r2_media"].append(vel_trends['subida']['r2'])
                        
        except Exception as e:
            if debug:
                logger.error(f"Error processing result file {result_file}: {str(e)}")
    
    # Calculate averages for each collected statistic
    for tech, stats in consolidated["tecnologias"].items():
        for stat_name, values in list(stats.items()):
            if values:
                stats[f"{stat_name}_avg"] = sum(values) / len(values)
                stats[f"{stat_name}_min"] = min(values)
                stats[f"{stat_name}_max"] = max(values)
    
    for speed_type in ["bajada", "subida"]:
        for stat_name, values in list(consolidated["velocidades"][speed_type].items()):
            if values:
                consolidated["velocidades"][speed_type][f"{stat_name}_avg"] = sum(values) / len(values)
                consolidated["velocidades"][speed_type][f"{stat_name}_min"] = min(values)
                consolidated["velocidades"][speed_type][f"{stat_name}_max"] = max(values)
    
    # Save consolidated report
    report_file = os.path.join(output_dir, 'consolidated_report.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(consolidated, f, ensure_ascii=False, indent=2)
        
    if debug:
        logger.info(f"Consolidated report saved to {report_file}")
        
    # Generate summary plots
    try:
        generar_graficos_consolidados(consolidated, output_dir, debug)
    except Exception as e:
        if debug:
            logger.error(f"Error generating consolidated plots: {str(e)}")
    
    return consolidated

#%%
def generar_graficos_consolidados(consolidated, output_dir, debug=False):
    """
    Generate consolidated plots summarizing results across all municipalities
    
    Args:
        consolidated (dict): Consolidated statistics
        output_dir (str): Directory to save output files
        debug (bool): Flag to enable debug output
        
    Returns:
        list: Paths to generated plot files
    """
    plot_files = []
    
    # 1. Technology comparison plot (accesses and speeds)
    if consolidated["tecnologias"]:
        try:
            # Prepare data
            technologies = list(consolidated["tecnologias"].keys())
            accesses = [consolidated["tecnologias"][tech].get("accesos_media_avg", 0) for tech in technologies]
            speeds = [consolidated["tecnologias"][tech].get("velocidad_media_avg", 0) for tech in technologies]
            
            # Sort by accesses
            sorted_indices = np.argsort(accesses)[::-1]  # Descending order
            sorted_technologies = [technologies[i] for i in sorted_indices]
            sorted_accesses = [accesses[i] for i in sorted_indices]
            sorted_speeds = [speeds[i] for i in sorted_indices]
            
            # Create plot with two subplots sharing x-axis
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
            # Plot accesses
            bars1 = ax1.bar(sorted_technologies, sorted_accesses, color='blue', alpha=0.7)
            ax1.set_title('Promedio de Accesos por Tecnología')
            ax1.set_ylabel('Número de Accesos (Promedio)')
            ax1.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height):,}',
                        ha='center', va='bottom', rotation=0)
            
            # Plot speeds
            bars2 = ax2.bar(sorted_technologies, sorted_speeds, color='green', alpha=0.7)
            ax2.set_title('Velocidad Promedio por Tecnología')
            ax2.set_xlabel('Tecnología')
            ax2.set_ylabel('Velocidad Bajada (Mbps)')
            ax2.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.2f}',
                        ha='center', va='bottom', rotation=0)
            
            # Rotate x labels for better readability
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            
            # Save plot
            tech_plot_path = os.path.join(output_dir, 'tecnologias_consolidado.png')
            plt.savefig(tech_plot_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            plot_files.append(tech_plot_path)
            
            if debug:
                logger.info(f"Technology comparison plot saved to {tech_plot_path}")
                
        except Exception as e:
            if debug:
                logger.error(f"Error generating technology comparison plot: {str(e)}")
    
    # 2. Speed trends plot
    try:
        # Create bar chart comparing download and upload speeds and trends
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Speed averages
        speed_types = ['Bajada', 'Subida']
        speed_avgs = [
            consolidated["velocidades"]["bajada"].get("media_general_avg", 0),
            consolidated["velocidades"]["subida"].get("media_general_avg", 0)
        ]
        
        bars1 = ax1.bar(speed_types, speed_avgs, color=['blue', 'green'], alpha=0.7)
        ax1.set_title('Velocidad Promedio (Mbps)')
        ax1.set_ylabel('Velocidad (Mbps)')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        # Speed trends
        speed_trends = [
            consolidated["velocidades"]["bajada"].get("tendencia_media_avg", 0),
            consolidated["velocidades"]["subida"].get("tendencia_media_avg", 0)
        ]
        
        bars2 = ax2.bar(speed_types, speed_trends, color=['blue', 'green'], alpha=0.7)
        ax2.set_title('Tendencia de Velocidad Promedio (Mbps/trimestre)')
        ax2.set_ylabel('Cambio por Trimestre (Mbps)')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.4f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        speed_plot_path = os.path.join(output_dir, 'velocidades_consolidado.png')
        plt.savefig(speed_plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        plot_files.append(speed_plot_path)
        
        if debug:
            logger.info(f"Speed trends plot saved to {speed_plot_path}")
            
    except Exception as e:
        if debug:
            logger.error(f"Error generating speed trends plot: {str(e)}")
    
    return plot_files

#%%
def generar_mapa_velocidades(output_dir, geodata_path=None, debug=False):
    """
    Generate choropleth map of internet speeds by municipality
    
    Args:
        output_dir (str): Directory containing municipality result files
        geodata_path (str): Path to GeoJSON file with municipality boundaries
        debug (bool): Flag to enable debug output
        
    Returns:
        str: Path to generated map file
    """
    try:
        # This function requires geopandas, which may need to be installed
        import geopandas as gpd
        
        if debug:
            logger.info("Generating choropleth map of internet speeds")
        
        # Find all JSON result files
        result_files = glob.glob(os.path.join(output_dir, '*_results.json'))
        
        # Extract municipality data
        municipalities_data = []
        for result_file in result_files:
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                
                # Skip files with errors or without velocity data
                if result.get('status') != 'success' or 'velocidades' not in result or not result['velocidades'].get('trends'):
                    continue
                
                # Extract municipality name from filename
                municipio = os.path.basename(result_file).replace('_results.json', '')
                
                # Get velocity data
                vel_trends = result['velocidades']['trends']
                
                municipalities_data.append({
                    'municipio': municipio,
                    'velocidad_bajada': vel_trends['bajada'].get('media', 0),
                    'velocidad_subida': vel_trends['subida'].get('media', 0),
                    'tendencia_bajada': vel_trends['bajada'].get('tendencia', 0),
                    'tendencia_subida': vel_trends['subida'].get('tendencia', 0)
                })
            except Exception as e:
                if debug:
                    logger.error(f"Error processing result file for map: {str(e)}")
        
        # Create DataFrame
        muni_df = pd.DataFrame(municipalities_data)
        
        # If no geodata provided, save CSV and return
        if geodata_path is None or not os.path.exists(geodata_path):
            csv_path = os.path.join(output_dir, 'velocidades_municipios.csv')
            muni_df.to_csv(csv_path, index=False)
            if debug:
                logger.info(f"No GeoJSON provided. Saved data to {csv_path}")
            return csv_path
        
        # Load GeoJSON
        gdf = gpd.read_file(geodata_path)
        
        # Ensure municipality names match between dataframes
        # This may require preprocessing depending on your data format
        gdf['municipio'] = gdf['municipio'].str.upper() if 'municipio' in gdf.columns else gdf['MUNICIPIO'].str.upper()
        muni_df['municipio'] = muni_df['municipio'].str.upper()
        
        # Merge data
        merged_gdf = gdf.merge(muni_df, on='municipio', how='left')
        
        # Create map
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        
        # Plot download speed
        merged_gdf.plot(column='velocidad_bajada', 
                       ax=ax,
                       legend=True,
                       cmap='viridis',
                       legend_kwds={'label': "Velocidad Bajada (Mbps)"},
                       missing_kwds={'color': 'lightgrey'})
        
        ax.set_title('Velocidad de Internet por Municipio')
        ax.axis('off')
        
        # Save map
        map_path = os.path.join(output_dir, 'mapa_velocidades.png')
        plt.savefig(map_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        if debug:
            logger.info(f"Choropleth map saved to {map_path}")
            
        return map_path
        
    except ImportError as e:
        if debug:
            logger.error(f"Error importing geopandas for map generation: {str(e)}")
        return None
    except Exception as e:
        if debug:
            logger.error(f"Error generating choropleth map: {str(e)}")
        return None

#%%
def run_analysis(data_dir=None, single_file=None, output_dir='resultados', 
                parallel=True, n_workers=None, debug=True, report_only=False,
                generate_map=False, geodata_path=None):
    """
    Run the automated internet analysis with flexible options
    
    Args:
        data_dir (str): Directory containing municipality CSV files
        single_file (str): Process a single file instead of a directory
        output_dir (str): Directory to save output files
        parallel (bool): Whether to process files in parallel
        n_workers (int): Number of worker processes to use (None = auto)
        debug (bool): Flag to enable debug output
        report_only (bool): Generate consolidated report from existing results
        generate_map (bool): Generate choropleth map of internet speeds
        geodata_path (str): Path to GeoJSON file with municipality boundaries
        
    Returns:
        dict: Summary of processing results
    """
    print("Starting Internet Analysis")
    
    # Process data
    if report_only:
        print("Generating consolidated report from existing results...")
        report = generar_reporte_consolidado(output_dir, debug=debug)
        
        print("\nConsolidated Report Summary:")
        print(f"Total municipalities: {report['municipios_count']}")
        print(f"Technology distribution: {len(report['tecnologias'])} technologies found")
        
        print("\nVelocity Statistics:")
        print(f"  - Download: {report['velocidades']['bajada'].get('media_general_avg', 0):.2f} Mbps (avg)")
        print(f"    Trend: {report['velocidades']['bajada'].get('tendencia_media_avg', 0):.4f} Mbps/trimester")
        print(f"  - Upload: {report['velocidades']['subida'].get('media_general_avg', 0):.2f} Mbps (avg)")
        print(f"    Trend: {report['velocidades']['subida'].get('tendencia_media_avg', 0):.4f} Mbps/trimester")
        
        # Generate map if requested
        if generate_map:
            map_path = generar_mapa_velocidades(output_dir, geodata_path=geodata_path, debug=debug)
            if map_path:
                print(f"\nChoropleth map saved to: {map_path}")
        
        return report
    
    elif single_file:
        # Process a single file
        print(f"Processing single file: {single_file}")
        result = procesar_municipio(single_file, output_dir=output_dir, debug=debug)
        
        # Print summary
        if result['status'] == 'success':
            print(f"Successfully processed {result['municipio']}")
            
            # Technology trends summary
            print("\nTechnology Trends:")
            for tech, trend in result['tecnologias']['trends'].items():
                print(f"  - {tech}:")
                if 'r2_velocidad' in trend:
                    print(f"    Velocidad: R²={trend['r2_velocidad']:.4f}, Media={trend['velocidad_media']:.2f} Mbps")
                if 'r2' in trend:
                    print(f"    Accesos: R²={trend['r2']:.4f}, Media={trend['accesos_media']:.2f}")
            
            # Velocity trends summary
            print("\nVelocity Trends:")
            vel_trends = result['velocidades']['trends']
            print(f"  - Bajada: Media={vel_trends['bajada']['media']:.2f} Mbps, Tendencia={vel_trends['bajada']['tendencia']:.4f}, R²={vel_trends['bajada']['r2']:.4f}")
            print(f"  - Subida: Media={vel_trends['subida']['media']:.2f} Mbps, Tendencia={vel_trends['subida']['tendencia']:.4f}, R²={vel_trends['subida']['r2']:.4f}")
            
            print(f"\nPlots saved to:")
            print(f"  - {result['tecnologias']['plot_path']}")
            print(f"  - {result['velocidades']['plot_path']}")
        else:
            print(f"Error or warning processing {result['municipio']}: {result.get('message', result.get('error', 'Unknown error/warning'))}")
        
        return result
    
    elif data_dir:
        # Process multiple files
        print(f"Processing all CSV files in {data_dir}")
        start_time = time.time()
        
        results = procesar_todos_municipios(
            data_dir, 
            output_dir=output_dir, 
            parallel=parallel,
            n_workers=n_workers,
            debug=debug
        )
        
        elapsed = time.time() - start_time
        
        # Print summary
        print("\nProcessing Summary:")
        print(f"Total files: {results['total_files']}")
        print(f"Success: {results['success']}")
        print(f"Warning: {results['warning']}")
        print(f"Partial: {results['partial']}")
        print(f"Error: {results['error']}")
        print(f"Processing time: {elapsed:.2f} seconds")
        
        # Generate consolidated report
        print("\nGenerating consolidated report...")
        report = generar_reporte_consolidado(output_dir, debug=debug)
        
        # Print report summary
        print("\nConsolidated Report Summary:")
        print(f"Total municipalities: {report['municipios_count']}")
        print(f"Technology distribution: {len(report['tecnologias'])} technologies found")
        
        print("\nVelocity Statistics:")
        print(f"  - Download: {report['velocidades']['bajada'].get('media_general_avg', 0):.2f} Mbps (avg)")
        print(f"    Trend: {report['velocidades']['bajada'].get('tendencia_media_avg', 0):.4f} Mbps/trimester")
        print(f"  - Upload: {report['velocidades']['subida'].get('media_general_avg', 0):.2f} Mbps (avg)")
        print(f"    Trend: {report['velocidades']['subida'].get('tendencia_media_avg', 0):.4f} Mbps/trimester")
        
        print("\nSee detailed results in the output directory:")
        print(f"  {os.path.abspath(output_dir)}")
        
        # Generate map if requested
        if generate_map:
            map_path = generar_mapa_velocidades(output_dir, geodata_path=geodata_path, debug=debug)
            if map_path:
                print(f"\nChoropleth map saved to: {map_path}")
        
        return results
    
    else:
        print("Error: You must specify either data_dir, single_file, or report_only")
        return {"status": "error", "message": "No input specified"}

#%% 
# Main script execution code
if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process internet access data by municipality')
    parser.add_argument('--data_dir', type=str, required=False, 
                        help='Directory containing municipality CSV files')
    parser.add_argument('--output_dir', type=str, default='resultados', 
                        help='Directory to save output files')
    parser.add_argument('--parallel', action='store_true', default=True,
                        help='Process files in parallel')
    parser.add_argument('--sequential', action='store_false', dest='parallel',
                        help='Process files sequentially')
    parser.add_argument('--workers', type=int, default=None, 
                        help='Number of worker processes (default: CPU count - 1)')
    parser.add_argument('--debug', action='store_true', 
                        help='Enable debug output')
    parser.add_argument('--single_file', type=str, default=None,
                        help='Process a single file instead of a directory')
    parser.add_argument('--geojson', type=str, default=None,
                        help='Path to GeoJSON file for map generation')
    parser.add_argument('--report_only', action='store_true',
                        help='Generate consolidated report from existing results')
    
    # Only parse args if not running in Jupyter
    if not is_jupyter():
        # Parse arguments
        args = parser.parse_args()
        
        # Run appropriate function based on arguments
        if args.report_only:
            generar_reporte_consolidado(args.output_dir, debug=args.debug)
        elif args.single_file:
            procesar_municipio(args.single_file, output_dir=args.output_dir, debug=args.debug)
        elif args.data_dir:
            procesar_todos_municipios(
                args.data_dir, 
                output_dir=args.output_dir, 
                parallel=args.parallel,
                n_workers=args.workers,
                debug=args.debug
            )
            generar_reporte_consolidado(args.output_dir, debug=args.debug)
            if args.geojson:
                generar_mapa_velocidades(args.output_dir, geodata_path=args.geojson, debug=args.debug)
        else:
            parser.print_help()

#%% 
# Example usage in a notebook:
# To run the full analysis on a directory:
# results = run_analysis(data_dir="your/data/directory", debug=True)
#
# To process a single file:
# result = run_analysis(single_file="path/to/municipality.csv", debug=True)
#
# To generate report only:
# report = run_analysis(report_only=True, output_dir="resultados")
