# ANÁLISIS COMPLETO DEL PROYECTO - INVENTARIO Y LIMPIEZA

## 📋 RESUMEN EJECUTIVO

Este documento contiene un inventario completo de todos los archivos del proyecto, identifica duplicados, explica las conexiones entre componentes y proporciona recomendaciones para la limpieza del proyecto.

**Fecha de análisis:** 2025-01-27  
**Proyecto:** InternetAccessColombia - Análisis de Acceso a Internet en Colombia

---

## 1. ESTRUCTURA GENERAL DEL PROYECTO

```
InternetAccessColombia/
├── main.py (vacío)
├── notebook.ipynb
├── 3.ipynb
├── Transformación datasets.ipynb
├── Transformacion_datasets_2.ipynb
├── requirements.txt
├── pyvenv.cfg
└── InternetAccessColombia-DataCleaning/
    ├── Limpieza/
    ├── modelos/
    ├── modelo_final/
    └── [múltiples archivos .py e .ipynb]
```

---

## 2. INVENTARIO COMPLETO DE ARCHIVOS

### 2.1. ARCHIVOS EN LA RAÍZ DEL PROYECTO

| Archivo | Tipo | Estado | Descripción |
|---------|------|--------|-------------|
| `main.py` | Python | ⚠️ VACÍO | Archivo principal (sin contenido) |
| `notebook.ipynb` | Jupyter | ❓ | Notebook sin contexto claro |
| `3.ipynb` | Jupyter | ❓ | Notebook con nombre genérico |
| `Transformación datasets.ipynb` | Jupyter | ✅ | Transformación de datasets |
| `Transformacion_datasets_2.ipynb` | Jupyter | ⚠️ DUPLICADO | Versión 2 de transformación |
| `requirements.txt` | Config | ✅ | Dependencias del proyecto |
| `pyvenv.cfg` | Config | ✅ | Configuración de entorno virtual |

### 2.2. ARCHIVOS EN `InternetAccessColombia-DataCleaning/`

#### A. Archivos de Revisión de Datasets (DUPLICADOS POTENCIALES)

| Archivo | Tipo | Estado | Descripción |
|---------|------|--------|-------------|
| `reivision_dataset_a_usar.py` | Python | ⚠️ | Revisión dataset (versión 1) |
| `reivision_dataset_a_usar.ipynb` | Jupyter | ⚠️ | Notebook equivalente |
| `reivision_dataset_a_usar_2.py` | Python | ⚠️ | Revisión dataset (versión 2) |
| `reivision_dataset_a_usar_2.ipynb` | Jupyter | ⚠️ | Notebook equivalente |
| `reivision_dataset_a_usar_2_copy.py` | Python | ❌ DUPLICADO | Copia de versión 2 |
| `reivision_dataset_a_usar_3.py` | Python | ⚠️ | Revisión dataset (versión 3) |
| `reivision_dataset_a_usar_3.ipynb` | Jupyter | ⚠️ | Notebook en `modelo_final/` |

**⚠️ PROBLEMA:** 4 versiones del mismo script + copias. Necesita consolidación.

#### B. Archivos de Datos Unificados (DUPLICADOS POTENCIALES)

| Archivo | Tipo | Estado | Descripción |
|---------|------|--------|-------------|
| `df_unificado_limpio.ipynb` | Jupyter | ⚠️ | Versión 1 |
| `df_unificado_limpio2.ipynb` | Jupyter | ⚠️ | Versión 2 |
| `df_unificado_limpio2.py` | Python | ⚠️ | Script versión 2 |
| `df_unificado_limpio3.ipynb` | Jupyter | ⚠️ | Versión 3 |
| `df_unificado_limpio3.py` | Python | ⚠️ | Script versión 3 |

**⚠️ PROBLEMA:** 3 versiones del mismo proceso. Necesita determinar cuál es la versión final.

#### C. Archivos de Coordenadas

| Archivo | Tipo | Estado | Descripción |
|---------|------|--------|-------------|
| `Coordenadas_api.py` | Python | ✅ | Obtiene coordenadas vía API |
| `Coordenadas_api.ipynb` | Jupyter | ⚠️ DUPLICADO | Notebook equivalente |

#### D. Archivos de Mapas

| Archivo | Tipo | Estado | Descripción |
|---------|------|--------|-------------|
| `mapa.py` | Python | ✅ | Generación de mapas |
| `map.ipynb` | Jupyter | ⚠️ DUPLICADO | Notebook equivalente |

#### E. Archivos de Proyección

| Archivo | Tipo | Estado | Descripción |
|---------|------|--------|-------------|
| `proyeccion_datos.py` | Python | ✅ | Proyección de datos |
| `proyeccion_datos.ipynb` | Jupyter | ⚠️ DUPLICADO | Notebook equivalente |

#### F. Archivos de Comparación

| Archivo | Tipo | Estado | Descripción |
|---------|------|--------|-------------|
| `comparacion_datasets.py` | Python | ✅ | Compara diferentes datasets |

#### G. Archivos de Carga de Datos

| Archivo | Tipo | Estado | Descripción |
|---------|------|--------|-------------|
| `upload-data.ipynb` | Jupyter | ✅ | Carga datos a sistema |

---

### 2.3. ARCHIVOS EN `modelo_final/data_analisis_proyecciones/`

#### ⚠️ ZONA CRÍTICA: Múltiples archivos de análisis similares

| Archivo | Tipo | Estado | Descripción |
|---------|------|--------|-------------|
| `complete-internet-analysis.py` | Python | ✅ **PRINCIPAL** | Análisis completo según prompt |
| `complete-internet-analysis.ipynb` | Jupyter | ⚠️ DUPLICADO | Notebook equivalente |
| `data_analisis_proyecciones.py` | Python | ⚠️ | Análisis de proyecciones (diferente enfoque) |
| `data_analisis_proyecciones.ipynb` | Jupyter | ⚠️ DUPLICADO | Notebook equivalente |
| `analisis-municipios-internet (1).py` | Python | ❌ DUPLICADO | Archivo con "(1)" - probable duplicado |
| `heatmaps.ipynb` | Jupyter | ✅ | Genera mapas de calor |
| `mini-heatmaps.ipynb` | Jupyter | ⚠️ | Versión mini de heatmaps |
| `count_reports.ipynb` | Jupyter | ✅ | Cuenta reportes generados |
| `test.ipynb` | Jupyter | ❌ TEST | Archivo de prueba |
| `test2.ipynb` | Jupyter | ❌ TEST | Archivo de prueba 2 |

**🔍 OBSERVACIÓN:** 
- `complete-internet-analysis.py` parece ser el archivo principal mencionado en el prompt como `internet_analysis.py`
- `data_analisis_proyecciones.py` tiene un enfoque diferente pero similar
- `analisis-municipios-internet (1).py` es claramente un duplicado

---

### 2.4. ARCHIVOS EN `modelos/`

#### Estructura por Modelo de ML:

```
modelos/
├── catboost/
│   ├── catboost.ipynb
│   └── catboost_ing_caract.ipynb
├── gbm/
│   ├── gbm.ipynb
│   └── gbm_ing_caract.ipynb
├── lightgbm/
│   ├── lightgbm.ipynb
│   └── lightgbm_ing_caract.ipynb
├── xgboost/
│   ├── xgboost.ipynb
│   └── xgboost_ing_caracteristicas.ipynb
├── randomforest/
│   ├── randomforest.ipynb
│   ├── randomforest_ing_caract.ipynb
│   ├── randomforest_ing_caract2.ipynb ⚠️
│   ├── randomforest_ing_caract.py
│   ├── test_modelo_randomforest.ipynb ❌
│   └── características.json ✅
├── regresion_lineal/
│   ├── regresion_lineal.ipynb
│   ├── regresion_lineal_ing_caract.ipynb
│   └── regresion_lineal_ing_caract.py
├── ingenieria_caracteristicas/
│   ├── ing_caract.ipynb
│   ├── ing_caract_2.ipynb ⚠️
│   └── ing_caract.py
└── Resumen_modelos.ipynb ✅
```

**⚠️ PATRÓN OBSERVADO:**
- Cada modelo tiene versión base y versión con ingeniería de características
- Algunos tienen versiones numeradas (2, _2) que pueden ser duplicados
- Archivos de prueba que deberían eliminarse

---

### 2.5. ARCHIVOS EN `Limpieza/`

| Archivo | Tipo | Estado | Descripción |
|---------|------|--------|-------------|
| `Limpieza_por_tipo_conexión.ipynb` | Jupyter | ✅ | Limpieza por tipo de conexión |
| `Limpieza_proveedores_floridablanca.ipynb` | Jupyter | ✅ | Limpieza específica de Floridablanca |
| `simple.ipynb` | Jupyter | ❓ | Notebook simple (sin contexto) |

---

## 3. IDENTIFICACIÓN DE DUPLICADOS

### 3.1. DUPLICADOS EXACTOS (Mismo código en .py e .ipynb)

| Par .py / .ipynb | Ubicación | Acción Recomendada |
|------------------|-----------|-------------------|
| `Coordenadas_api.py` / `Coordenadas_api.ipynb` | Raíz DataCleaning | Mantener solo .py o solo .ipynb según uso |
| `mapa.py` / `map.ipynb` | Raíz DataCleaning | Mantener solo .py o solo .ipynb según uso |
| `proyeccion_datos.py` / `proyeccion_datos.ipynb` | Raíz DataCleaning | Mantener solo .py o solo .ipynb según uso |
| `complete-internet-analysis.py` / `complete-internet-analysis.ipynb` | modelo_final/ | Mantener ambos si se usan, sino consolidar |
| `data_analisis_proyecciones.py` / `data_analisis_proyecciones.ipynb` | modelo_final/ | Mantener ambos si se usan, sino consolidar |

### 3.2. VERSIONES NUMERADAS (Probables duplicados)

| Grupo | Archivos | Acción Recomendada |
|-------|----------|-------------------|
| **Revisión datasets** | `reivision_dataset_a_usar.py` (v1, v2, v2_copy, v3) | **Determinar versión final y eliminar resto** |
| **Datos unificados** | `df_unificado_limpio*.ipynb` (v1, v2, v3) | **Determinar versión final y eliminar resto** |
| **Ingeniería características** | `ing_caract.ipynb`, `ing_caract_2.ipynb` | Revisar diferencias, mantener solo la última |
| **Random Forest** | `randomforest_ing_caract.ipynb`, `randomforest_ing_caract2.ipynb` | Revisar diferencias, mantener solo la última |

### 3.3. ARCHIVOS CON NOMBRES PROBLEMÁTICOS

| Archivo | Problema | Acción |
|---------|----------|--------|
| `analisis-municipios-internet (1).py` | Nombre con "(1)" indica duplicado | **ELIMINAR** (duplicado de otro archivo) |
| `reivision_dataset_a_usar_2_copy.py` | Nombre indica copia | **ELIMINAR** (copia explícita) |
| `test.ipynb`, `test2.ipynb` | Archivos de prueba | **ELIMINAR** o mover a carpeta `tests/` |
| `test_modelo_randomforest.ipynb` | Archivo de prueba | **ELIMINAR** o mover a carpeta `tests/` |
| `3.ipynb` | Nombre genérico | **RENOMBRAR** o eliminar si no se usa |
| `simple.ipynb` | Nombre genérico | **RENOMBRAR** o eliminar si no se usa |

### 3.4. ARCHIVOS VACÍOS O SIN CONTENIDO

| Archivo | Estado | Acción |
|---------|--------|--------|
| `main.py` | Vacío | **ELIMINAR** o implementar punto de entrada |

---

## 4. CONEXIONES ENTRE ARCHIVOS

### 4.1. FLUJO DE DATOS PRINCIPAL

```
1. DATOS ORIGINALES
   └── Limpieza/data/INFORMACIÓN_TRIMESTRAL_DE_ACCESOS_FIJOS...COMPLETO.csv
       │
       ├──> reivision_dataset_a_usar*.py (Revisa qué dataset usar)
       │
       ├──> df_unificado_limpio*.py (Crea dataset unificado)
       │    └──> Limpieza/data/df_unificado_limpio*.csv
       │
       └──> data_analisis_proyecciones.py (Crea subdatasets por ubicación)
            └──> Limpieza/data/subdatasets-ubicacion/*.csv
                 │
                 └──> complete-internet-analysis.py (Análisis principal)
                      └──> resultados/*.json, *.png, *.csv
```

### 4.2. DEPENDENCIAS DE CÓDIGO

#### `complete-internet-analysis.py` (Archivo Principal)
- **Lee de:** `Limpieza/data/subdatasets-ubicacion/*.csv`
- **Genera:** `resultados/*.json`, `resultados/*.png`, `resultados/*.csv`
- **No importa otros archivos del proyecto** (es autocontenido)

#### `data_analisis_proyecciones.py`
- **Lee de:** `Limpieza/data/subdatasets-ubicacion/*.csv`
- **Genera:** Reportes y resúmenes
- **Relación:** Similar a `complete-internet-analysis.py` pero enfoque diferente

#### `reivision_dataset_a_usar*.py`
- **Lee de:** `Limpieza/data/INFORMACIÓN_TRIMESTRAL_DE_ACCESOS_FIJOS...COMPLETO.csv`
- **Propósito:** Determinar qué dataset usar para análisis
- **Genera:** Decisiones/documentación (no archivos de salida claros)

#### `df_unificado_limpio*.py`
- **Lee de:** 
  - `Limpieza/data/INFORMACIÓN_TRIMESTRAL_DE_ACCESOS_FIJOS...COMPLETO.csv`
  - `Limpieza/data/coordenadas_unicas.csv`
- **Genera:** `Limpieza/data/df_unificado_limpio*.csv`

#### `Coordenadas_api.py`
- **Lee de:** `Limpieza/data/df_unificado_limpio*.csv`
- **Genera:** `Limpieza/data/coordenadas_unicas.csv` o similar

### 4.3. ARCHIVOS INDEPENDIENTES (Sin dependencias claras)

- `modelos/*/` - Modelos de ML independientes
- `mapa.py` - Generación de mapas (probablemente usa resultados)
- `proyeccion_datos.py` - Proyecciones (probablemente usa resultados)
- `comparacion_datasets.py` - Compara datasets

---

## 5. ARCHIVOS FALTANTES MENCIONADOS EN EL PROMPT

El prompt menciona estos archivos que **NO EXISTEN** en el proyecto:

| Archivo Mencionado | Archivo Real Encontrado | Estado |
|-------------------|-------------------------|--------|
| `internet_analysis.py` | `complete-internet-analysis.py` | ✅ **EQUIVALENTE** |
| `internet_automation.py` | No existe | ❌ **FALTANTE** o integrado en `complete-internet-analysis.py` |
| `create_municipality_dataset.py` | `data_analisis_proyecciones.py` (líneas 67-87) | ✅ **FUNCIONALIDAD INCLUIDA** |

**🔍 CONCLUSIÓN:** 
- `complete-internet-analysis.py` ES el `internet_analysis.py` mencionado en el prompt
- La funcionalidad de `create_municipality_dataset.py` está en `data_analisis_proyecciones.py`
- `internet_automation.py` puede estar integrado en `complete-internet-analysis.py` (función `procesar_todos_municipios`)

---

## 6. ESTRUCTURA DE DATOS

### 6.1. DIRECTORIOS DE DATOS

```
Limpieza/data/
├── [Archivos CSV originales del MinTIC]
├── subdatasets-ubicacion/  ← GENERADO POR data_analisis_proyecciones.py
│   └── DEPARTAMENTO.MUNICIPIO.csv (1121 archivos según logs)
├── df_unificado_limpio*.csv (múltiples versiones)
├── coordenadas_unicas.csv
└── resultados/ (si existe, generado por análisis)
```

### 6.2. ARCHIVOS DE DATOS DUPLICADOS

| Archivo | Versiones | Acción |
|---------|-----------|--------|
| `df_unificado_limpio.csv` | v1, v2, v3, actualizado, imputado, con_proyecciones | **Determinar cuál es la versión final** |

---

## 6.3. INVENTARIO COMPLETO DE ARCHIVOS DE DATOS

### A. DATOS ORIGINALES DEL MINTIC (Fuente Principal)

**Ubicación:** `Limpieza/data/`

| Archivo | Tipo | Descripción | Estado |
|---------|------|-------------|--------|
| `INFORMACIÓN_TRIMESTRAL_DE_ACCESOS_FIJOS_A_INTERNET_POR_PROVEEDOR,_DEPARTAMENTO,_MUNICIPIO,_SEGMENTO,_TECNOLOGIA,_Y_VELOCIDAD_DE_CONEXIÓN_COMPLETO.csv` | CSV | **ARCHIVO PRINCIPAL** - Dataset completo | ✅ **MANTENER** |
| `INFORMACIÓN_TRIMESTRAL_DE_ACCESOS_FIJOS_A_INTERNET_POR_PROVEEDOR,_DEPARTAMENTO,_MUNICIPIO,_SEGMENTO,_TECNOLOGIA,_Y_VELOCIDAD_DE_CONEXIÓN_4.csv` | CSV | Versión parcial (trimestre 4?) | ⚠️ **REVISAR** |
| `INFORMACIÓN_TRIMESTRAL_DE_ACCESOS_FIJOS_A_INTERNET_POR_PROVEEDOR,_DEPARTAMENTO,_MUNICIPIO,_SEGMENTO,_TECNOLOGIA,_Y_VELOCIDAD_DE_CONEXIÓN_5.csv` | CSV | Versión parcial (trimestre 5?) | ⚠️ **REVISAR** |
| `INFORMACIÓN_TRIMESTRAL_DE_ACCESOS_FIJOS_A_INTERNET_POR_PROVEEDOR,_DEPARTAMENTO,_MUNICIPIO,_SEGMENTO,_TECNOLOGIA,_Y_VELOCIDAD_DE_CONEXIÓN_6.csv` | CSV | Versión parcial (trimestre 6?) | ⚠️ **REVISAR** |
| `INFORMACIÓN_TRIMESTRAL_DE_ACCESOS_FIJOS_A_INTERNET_POR_PROVEEDOR,_DEPARTAMENTO,_MUNICIPIO,_SEGMENTO,_TECNOLOGIA,_Y_VELOCIDAD_DE_CONEXIÓN_7.csv` | CSV | Versión parcial (trimestre 7?) | ⚠️ **REVISAR** |
| `INFORMACIÓN_TRIMESTRAL_DE_ACCESOS_FIJOS_A_INTERNET_POR_PROVEEDOR__1.csv` | CSV | Versión parcial (proveedor?) | ⚠️ **REVISAR** |
| `INFORMACIÓN_TRIMESTRAL_DE_ACCESOS_FIJOS_A_INTERNET_POR_DEPARTAMENTO_Y_POBLACIÓN_2.csv` | CSV | Datos agregados por departamento | ✅ **MANTENER** |
| `INFORMACIÓN_TRIMESTRAL_DE_ACCESOS_FIJOS_A_INTERNET_POR_DEPARTAMENTO,_MUNICIPIO_Y_POBLACIÓN_3.csv` | CSV | Datos agregados por municipio | ✅ **MANTENER** |
| `INFORMACIÓN_TRIMESTRAL_DE_INGRESOS_DEL_SERVICIO_DE_ACCESO_FIJO_A_INTERNET_POR_PROVEEDOR__8.csv` | CSV | Ingresos por proveedor | ✅ **MANTENER** |
| `INFORMACIÓN_TRIMESTRAL_ABONADOS_DE_INTERNET_MÓVIL__DEMANDA,_POR_PROVEEDOR_9.csv` | CSV | Abonados internet móvil (demanda) | ✅ **MANTENER** |
| `INFORMACIÓN_TRIMESTRAL_DE_INGRESOS_DE_INTERNET_MÓVIL_DEMANDA,_POR_PROVEEDOR_10.csv` | CSV | Ingresos internet móvil (demanda) | ✅ **MANTENER** |
| `INFORMACIÓN_TRIMESTRAL_DE_TRÁFICO_DE_INTERNET_MÓVIL_DEMANDA,_POR_PROVEEDOR_11.csv` | CSV | Tráfico internet móvil (demanda) | ✅ **MANTENER** |
| `INFORMACIÓN_TRIMESTRAL_SUSCRIPTORES_DE_INTERNET_MÓVIL_SUSCRIPCIÓN,_POR_PROVEEDOR_12.csv` | CSV | Suscriptores internet móvil | ✅ **MANTENER** |
| `INFORMACIÓN_TRIMESTRAL_DE_INGRESOS_DE_INTERNET_MÓVIL_SUSCRIPCIÓN,_POR_PROVEEDOR_13.csv` | CSV | Ingresos internet móvil (suscripción) | ✅ **MANTENER** |
| `INFORMACIÓN_TRIMESTRAL_DE_TRÁFICO_DE_INTERNET_MÓVIL_SUSCRIPCIÓN,_POR_PROVEEDOR_14.csv` | CSV | Tráfico internet móvil (suscripción) | ✅ **MANTENER** |
| `INFORMACIÓN_TRIMESTRAL_ABONADOS_DE_TELEFONÍA_MÓVIL_15.csv` | CSV | Abonados telefonía móvil | ✅ **MANTENER** |
| `INFORMACIÓN_TRIMESTRAL_-_TRÁFICO_DE_VOZ_EN_MINUTOS_DE_PROVEEDORES_DE_REDES_Y_SERVICIOS_MÓVILES,_POR_PROVEEDOR_16.csv` | CSV | Tráfico de voz móvil | ✅ **MANTENER** |
| `INFORMACIÓN_TRIMESTRAL_DE_INGRESOS_POR_TRÁFICO_DE_VOZ_DE_PROVEEDORES_DE_REDES_Y_SERVICIOS_MÓVILES_17.csv` | CSV | Ingresos por tráfico de voz | ✅ **MANTENER** |
| `INFORMACIÓN_TRIMESTRAL_DE_COBERTURA_MUNICIPAL_DEL_SERVICIO_MÓVIL_18.csv` | CSV | Cobertura móvil municipal | ✅ **MANTENER** |
| `LÍNEAS_EN_SERVICIO_POR_ESTRATO_Y_MUNICIPIO_DE_TELEFONÍA_PÚBLICA_BÁSICA_CONMUTADA_19.csv` | CSV | Líneas telefonía pública | ✅ **MANTENER** |
| `INFORMACIÓN_TRIMESTRAL_DE_INGRESOS_DE_TELEFONÍA_FIJA_20.csv` | CSV | Ingresos telefonía fija | ✅ **MANTENER** |

**⚠️ NOTA:** Los archivos numerados (4, 5, 6, 7) pueden ser versiones parciales del archivo COMPLETO. Revisar si son necesarios o si están incluidos en el archivo COMPLETO.

### B. DATASETS PROCESADOS Y UNIFICADOS (MÚLTIPLES VERSIONES)

| Archivo | Descripción | Estado | Acción Recomendada |
|---------|-------------|--------|-------------------|
| `df_unificado_limpio.csv` | Versión 1 - Dataset unificado limpio | ⚠️ | **Determinar versión final** |
| `df_unificado_limpio2.csv` | Versión 2 - Dataset unificado limpio | ⚠️ | **Determinar versión final** |
| `df_unificado_limpio_actualizado.csv` | Versión actualizada | ⚠️ | **Determinar versión final** |
| `df_unificado_limpio_imputado.csv` | Versión con imputación de valores faltantes | ⚠️ | **Determinar versión final** |
| `df_unificado_limpio_con_proyecciones.csv` | Versión con proyecciones | ⚠️ | **Determinar versión final** |
| `df_unificado_limpio_con_proyecciones_limpio.csv` | Versión con proyecciones limpias | ⚠️ | **Determinar versión final** |
| `df_mejorado.csv` | Dataset mejorado | ⚠️ | Revisar relación con df_unificado |
| `df_mejorado_ing_caract.csv` | Dataset mejorado con ingeniería de características | ⚠️ | Revisar relación con df_unificado |
| `df_mejorado_ing_caract_2.csv` | Versión 2 con ingeniería de características | ⚠️ | Revisar relación con df_unificado |
| `datos_con_proyecciones.csv` | Datos con proyecciones (nombre alternativo) | ⚠️ | **POSIBLE DUPLICADO** |
| `datos_incompletos.csv` | Datos incompletos (probablemente para análisis) | ✅ | Mantener para referencia |
| `filas_con_nulos.csv` | Filas con valores nulos (probablemente para análisis) | ✅ | Mantener para referencia |

**🔍 PROBLEMA CRÍTICO:** Hay al menos 9 versiones diferentes del dataset unificado. Necesita consolidación urgente.

### C. DATOS DE COORDENADAS GEOGRÁFICAS

| Archivo | Descripción | Estado |
|---------|-------------|--------|
| `DatasetCoordenadas.csv` | Dataset original de coordenadas | ✅ **MANTENER** |
| `coordenadas_unicas.csv` | Coordenadas únicas procesadas | ✅ **MANTENER** |
| `coordenadas_nuevas.csv` | Coordenadas nuevas obtenidas | ⚠️ **REVISAR** - ¿Reemplaza a coordenadas_unicas? |
| `municipios_sin_coordenadas.csv` | Municipios sin coordenadas | ✅ **MANTENER** (referencia) |
| `municipios_sin_coordenadas_completas.csv` | Municipios con coordenadas incompletas | ✅ **MANTENER** (referencia) |
| `progreso_coordenadas.csv` | Progreso de obtención de coordenadas | ✅ **MANTENER** (referencia) |
| `progreso_coordenadas_nuevas.csv` | Progreso de coordenadas nuevas | ⚠️ **REVISAR** - ¿Duplicado? |

**⚠️ NOTA:** `coordenadas_nuevas.csv` y `coordenadas_unicas.csv` pueden ser duplicados o versiones diferentes. Revisar.

### D. DATOS POR DEPARTAMENTO

**Ubicación:** `Limpieza/data/departamentos/`

**Total:** 33 archivos CSV (uno por cada departamento de Colombia)

| Ejemplos | Descripción |
|----------|-------------|
| `AMAZONAS.csv`, `ANTIOQUIA.csv`, `BOGOTÁ D.C..csv`, etc. | Datos separados por departamento |

**Estado:** ✅ **MANTENER** - Útiles para análisis por departamento

### E. SUBDATASETS POR UBICACIÓN (GENERADOS)

**Ubicación:** `Limpieza/data/subdatasets-ubicacion/` (según prompt, pero no encontrado en exploración)

**Descripción:** Archivos CSV generados con formato `DEPARTAMENTO.MUNICIPIO.csv`

**Cantidad estimada:** ~1121 archivos (según logs del análisis)

**Estado:** ⚠️ **GENERADOS** - Pueden regenerarse desde el dataset principal

**⚠️ NOTA:** Este directorio no se encontró en la exploración. Puede estar en otra ubicación o haber sido eliminado. Verificar según el prompt menciona: `../../Limpieza/data/subdatasets-ubicacion/`

### F. RESULTADOS DEL ANÁLISIS

**Ubicación:** `Limpieza/data/resultados/`

#### Archivos JSON (Resultados del análisis)
- **Cantidad:** ~1123 archivos (uno por municipio)
- **Formato:** `DEPARTAMENTO_MUNICIPIO_results.json`
- **Contenido:** Estadísticas, tendencias, predicciones por municipio
- **Estado:** ✅ **MANTENER** - Resultados del análisis principal

#### Archivos PNG (Gráficas)
- **Cantidad:** ~2000+ archivos (2 por municipio: tecnologias.png y velocidades.png)
- **Formato:** 
  - `DEPARTAMENTO_MUNICIPIO_tecnologias.png`
  - `DEPARTAMENTO_MUNICIPIO_velocidades.png`
- **Estado:** ✅ **MANTENER** - Visualizaciones importantes

**⚠️ NOTA:** También existe un directorio `Limpieza/data/graficas/` con archivos similares. Verificar si hay duplicación.

### G. OTROS ARCHIVOS DE DATOS

| Archivo | Descripción | Estado |
|---------|-------------|--------|
| `proveedores_floridablanca_santander_fibra_detalle.csv` | Detalle de proveedores en Floridablanca | ✅ **MANTENER** (específico) |
| `articles-383732_archivo_xls.xlsx` | Archivo Excel original (probablemente del MinTIC) | ✅ **MANTENER** (fuente) |
| `articles-383732_archivo_xls.xlsx:Zone.Identifier` | Metadatos de Windows | ❌ **ELIMINAR** (metadatos) |
| `DatasetCoordenadas.csv:Zone.Identifier` | Metadatos de Windows | ❌ **ELIMINAR** (metadatos) |

### H. MODELOS ENTRENADOS

**Ubicación:** `Limpieza/data/models/`

| Archivo | Descripción | Estado |
|---------|-------------|--------|
| `modelo_completo_datos_base.joblib` | Modelo entrenado con datos base | ✅ **MANTENER** |
| `modelo_completo_datos_prueba.joblib` | Modelo entrenado con datos de prueba | ✅ **MANTENER** |

---

## 6.4. RESUMEN DE DUPLICADOS EN DATOS

### Duplicados Confirmados o Probables:

1. **Datasets Unificados:** 9 versiones diferentes
   - `df_unificado_limpio.csv` (v1)
   - `df_unificado_limpio2.csv` (v2)
   - `df_unificado_limpio_actualizado.csv`
   - `df_unificado_limpio_imputado.csv`
   - `df_unificado_limpio_con_proyecciones.csv`
   - `df_unificado_limpio_con_proyecciones_limpio.csv`
   - `df_mejorado.csv` (posiblemente relacionado)
   - `df_mejorado_ing_caract.csv` (posiblemente relacionado)
   - `df_mejorado_ing_caract_2.csv` (posiblemente relacionado)

2. **Coordenadas:** Posibles duplicados
   - `coordenadas_unicas.csv` vs `coordenadas_nuevas.csv`
   - `progreso_coordenadas.csv` vs `progreso_coordenadas_nuevas.csv`

3. **Datos con Proyecciones:** Posibles duplicados
   - `df_unificado_limpio_con_proyecciones.csv` vs `datos_con_proyecciones.csv`

4. **Archivos Parciales del MinTIC:** 
   - Archivos numerados (4, 5, 6, 7) pueden estar incluidos en COMPLETO

5. **Gráficas:** 
   - `resultados/*.png` vs `graficas/*.png` (verificar duplicación)

### Archivos a Eliminar (Seguro):

- `articles-383732_archivo_xls.xlsx:Zone.Identifier` (metadatos Windows)
- `DatasetCoordenadas.csv:Zone.Identifier` (metadatos Windows)

---

## 7. RECOMENDACIONES DE LIMPIEZA

### 7.1. ELIMINACIÓN INMEDIATA (Bajo Riesgo)

✅ **Archivos a eliminar sin dudar:**
1. `analisis-municipios-internet (1).py` - Duplicado explícito
2. `reivision_dataset_a_usar_2_copy.py` - Copia explícita
3. `test.ipynb`, `test2.ipynb` - Archivos de prueba
4. `test_modelo_randomforest.ipynb` - Archivo de prueba
5. `main.py` - Vacío

### 7.2. CONSOLIDACIÓN REQUERIDA (Revisar Antes)

⚠️ **Grupos que necesitan revisión:**

#### Grupo 1: Revisión de Datasets
- **Mantener:** La versión más reciente y completa de `reivision_dataset_a_usar*.py`
- **Eliminar:** Las versiones anteriores
- **Acción:** Revisar `reivision_dataset_a_usar_3.py` vs `reivision_dataset_a_usar_2.py` y mantener solo una

#### Grupo 2: Datos Unificados
- **Mantener:** La versión final de `df_unificado_limpio*.py`
- **Eliminar:** Versiones anteriores
- **Acción:** Determinar cuál es la versión final (probablemente v3)

#### Grupo 3: Análisis Principal
- **Mantener:** `complete-internet-analysis.py` (es el archivo principal)
- **Revisar:** `data_analisis_proyecciones.py` - ¿Es necesario o duplicado?
- **Acción:** Comparar ambos y determinar si se pueden consolidar

### 7.3. DECISIONES SOBRE .py vs .ipynb

**Estrategia recomendada:**
- Si el archivo se ejecuta como script: **Mantener solo .py**
- Si el archivo se usa para exploración: **Mantener solo .ipynb**
- Si ambos se usan activamente: **Mantener ambos**

**Archivos a revisar:**
- `Coordenadas_api.py` / `.ipynb`
- `mapa.py` / `map.ipynb`
- `proyeccion_datos.py` / `.ipynb`
- `complete-internet-analysis.py` / `.ipynb`

### 7.4. RENOMBRADO RECOMENDADO

| Archivo Actual | Nombre Sugerido | Razón |
|----------------|-----------------|-------|
| `complete-internet-analysis.py` | `internet_analysis.py` | Coincide con prompt |
| `3.ipynb` | `[nombre_descriptivo].ipynb` o eliminar | Nombre genérico |
| `simple.ipynb` | `[nombre_descriptivo].ipynb` o eliminar | Nombre genérico |

---

## 8. MAPA DE CONEXIONES VISUAL

```
┌─────────────────────────────────────────────────────────────┐
│                    DATOS ORIGINALES                          │
│  INFORMACIÓN_TRIMESTRAL_DE_ACCESOS_FIJOS...COMPLETO.csv     │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐
│  reivision_  │ │ df_unificado │ │ data_analisis_        │
│  dataset_    │ │ _limpio*.py  │ │ proyecciones.py      │
│  a_usar*.py  │ │              │ │                      │
└──────────────┘ └──────┬───────┘ └──────────┬───────────┘
                       │                      │
                       ▼                      ▼
              ┌─────────────────┐  ┌──────────────────────┐
              │ df_unificado_   │  │ subdatasets-        │
              │ limpio*.csv     │  │ ubicacion/*.csv     │
              └─────────────────┘  └──────────┬──────────┘
                                               │
                                               ▼
                                    ┌──────────────────────┐
                                    │ complete-internet-    │
                                    │ analysis.py           │
                                    │ (internet_analysis.py)│
                                    └──────────┬─────────────┘
                                               │
                                               ▼
                                    ┌──────────────────────┐
                                    │ resultados/          │
                                    │ *.json, *.png, *.csv │
                                    └──────────────────────┘
```

---

## 9. CHECKLIST DE LIMPIEZA

### Fase 1: Eliminación Segura

#### Archivos de Código
- [ ] Eliminar `analisis-municipios-internet (1).py`
- [ ] Eliminar `reivision_dataset_a_usar_2_copy.py`
- [ ] Eliminar `test.ipynb`, `test2.ipynb`
- [ ] Eliminar `test_modelo_randomforest.ipynb`
- [ ] Eliminar o implementar `main.py`

#### Archivos de Datos
- [ ] Eliminar `articles-383732_archivo_xls.xlsx:Zone.Identifier` (metadatos Windows)
- [ ] Eliminar `DatasetCoordenadas.csv:Zone.Identifier` (metadatos Windows)

### Fase 2: Revisión y Consolidación

#### Archivos de Código
- [ ] Revisar `reivision_dataset_a_usar*.py` y mantener solo la versión final
- [ ] Revisar `df_unificado_limpio*.py` y mantener solo la versión final
- [ ] Comparar `complete-internet-analysis.py` vs `data_analisis_proyecciones.py`
- [ ] Revisar versiones numeradas en `modelos/` (ing_caract_2, randomforest_ing_caract2)

#### Archivos de Datos
- [ ] **CRÍTICO:** Revisar las 9 versiones de `df_unificado_limpio*.csv` y determinar cuál es la versión final
- [ ] Comparar `coordenadas_unicas.csv` vs `coordenadas_nuevas.csv` - ¿Son duplicados?
- [ ] Comparar `progreso_coordenadas.csv` vs `progreso_coordenadas_nuevas.csv` - ¿Son duplicados?
- [ ] Verificar si `datos_con_proyecciones.csv` es duplicado de `df_unificado_limpio_con_proyecciones.csv`
- [ ] Revisar archivos parciales del MinTIC (4, 5, 6, 7) - ¿Están incluidos en COMPLETO?
- [ ] Verificar duplicación entre `resultados/*.png` y `graficas/*.png`
- [ ] Verificar ubicación de `subdatasets-ubicacion/` (mencionado en prompt pero no encontrado)

### Fase 3: Decisión .py vs .ipynb
- [ ] Decidir qué mantener: `Coordenadas_api.py` o `.ipynb`
- [ ] Decidir qué mantener: `mapa.py` o `map.ipynb`
- [ ] Decidir qué mantener: `proyeccion_datos.py` o `.ipynb`
- [ ] Decidir qué mantener: `complete-internet-analysis.py` o `.ipynb`

### Fase 4: Renombrado
- [ ] Renombrar `complete-internet-analysis.py` → `internet_analysis.py` (opcional)
- [ ] Renombrar o eliminar `3.ipynb`
- [ ] Renombrar o eliminar `simple.ipynb`

### Fase 5: Documentación
- [ ] Crear README.md con estructura del proyecto
- [ ] Documentar flujo de datos
- [ ] Documentar dependencias entre archivos

---

## 10. ESTIMACIÓN DE ESPACIO A LIBERAR

### Archivos de Código a Eliminar (estimado):
- Duplicados explícitos: ~5 archivos
- Versiones antiguas: ~10-15 archivos
- Archivos de prueba: ~3 archivos
- Archivos vacíos: ~1 archivo
- **Subtotal código:** ~20-25 archivos

### Archivos de Datos a Eliminar (estimado):
- Metadatos Windows: ~2 archivos
- Versiones antiguas de datasets unificados: ~6-8 archivos (después de determinar versión final)
- Archivos parciales del MinTIC (si están en COMPLETO): ~4 archivos
- Duplicados de coordenadas: ~2 archivos (si se confirman duplicados)
- **Subtotal datos:** ~14-16 archivos

### Archivos de Resultados:
- **NO ELIMINAR** sin verificar - Son resultados del análisis (~1123 JSON + ~2000 PNG)
- Considerar comprimir si ocupan mucho espacio

**Total estimado a eliminar:** ~34-41 archivos (código + datos)

**⚠️ NOTA:** Los archivos de datos pueden ser grandes. Verificar tamaño antes de eliminar.

---

## 11. NOTAS FINALES

### ⚠️ ADVERTENCIAS

1. **NO ELIMINAR** sin revisar:
   - Archivos en `modelos/` - Pueden tener diferencias importantes
   - Versiones numeradas - Revisar diferencias antes de eliminar
   - Archivos .ipynb - Pueden contener resultados/gráficas importantes

2. **BACKUP RECOMENDADO:**
   - Hacer backup completo antes de eliminar archivos
   - Usar control de versiones (git) para poder revertir

3. **ARCHIVOS DE DATOS:**
   - No eliminar archivos en `Limpieza/data/` sin verificar dependencias
   - Los subdatasets pueden ser regenerados, pero verificar primero
   - **CRÍTICO:** Antes de eliminar versiones de `df_unificado_limpio*.csv`, verificar cuál es la versión que se usa actualmente en el código
   - Los archivos de resultados (`resultados/*.json`, `resultados/*.png`) son costosos de regenerar - considerar comprimir en lugar de eliminar
   - Los archivos del MinTIC (nombres largos) son la fuente original - **NO ELIMINAR**

### ✅ PRÓXIMOS PASOS SUGERIDOS

1. Revisar este documento con el equipo/usuario
2. Ejecutar Fase 1 (eliminación segura)
3. Revisar archivos de Fase 2 antes de eliminar
4. Documentar decisiones tomadas
5. Actualizar README con estructura final

---

**Documento generado el:** 2025-01-27  
**Última actualización:** 2025-01-27

