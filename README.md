# DATATHON MICAI 2026

## Reto 2 \| Predicción de resistencias del cemento

**Empresa participante: GCC**

Este reto busca predecir las resistencias mecánicas del cemento a partir de variables de calidad del proceso y del producto: métricas físicas, fases minerales y composición química. La solución debe transformar datos crudos en un pipeline reproducible dentro de Databricks, entrenar modelos con seguimiento en MLflow y entregar una experiencia analítica clara para negocio.

---

## 1\. Descripción del problema

En la industria cementera, la calidad del producto y su comportamiento mecánico dependen de múltiples factores físico-químicos. Poder estimar con anticipación la resistencia a 1, 7 y 28 días permite tomar mejores decisiones operativas, reducir merma, optimizar el uso de aditivos y reaccionar más rápido ante desviaciones de calidad.

El reto consiste en construir una solución de datos y machine learning que, a partir de mediciones de calidad del cemento, prediga las resistencias mecánicas esperadas y explique qué variables fueron más relevantes para llegar a cada predicción.

---

## 2\. Datos entregados

Cada equipo recibirá un dataset tabular en CSV con observaciones históricas de calidad y resistencias del cemento, así como la rúbrica oficial de evaluación en Excel.

| Recurso | Archivo en este proyecto | Descripción |
| :---- | :---- | :---- |
| Dataset principal | `datos/datathon_data_gcc.csv` | Dataset crudo con 791 registros y 16 columnas: variables predictoras de calidad \+ 3 variables objetivo de resistencia |
| Rúbrica oficial | `evaluacion/Rubrica_Datathon_GCC.xlsx` | Formato oficial de evaluación con criterios, pesos y escala de calificación |

### Estructura de la carpeta

```
gcc/
├── README.md
├── datos/
│   └── datathon_data_gcc.csv
└── evaluacion/
    └── Rubrica_Datathon_GCC.xlsx
```

### Esquema del dataset

**Variable categórica**

* `ProductType` — tipo de producto

**Variables predictoras**

* Métricas físicas: `Physical_Metric_A`, `Physical_Metric_B`  
* Fases minerales: `Mineral_Phase_A` a `Mineral_Phase_H`  
* Composición química: `Chemical_Composition_A`, `Chemical_Composition_B`

**Variables objetivo**

* `Strength1` — resistencia a 1 día  
* `Strength7` — resistencia a 7 días  
* `Strength28` — resistencia a 28 días

### Observaciones iniciales del dataset

* El archivo trae **791 registros de datos** más el encabezado.  
* Hay **5 tipos de producto** visibles: `Type 1`, `Type 2`, `Type 3`, `Type 4` y `Type 5`.  
* Existen valores nulos en varias columnas, por lo que el tratamiento de calidad del dato es parte explícita del reto.  
* `Strength1`, `Strength7` y `Strength28` son targets y **no deben usarse como features**.

---

## 3\. Qué deben construir

La solución debe incluir, como mínimo, los siguientes componentes:

### 3.1 Pipeline de limpieza con arquitectura medallón

Construir un pipeline de datos con arquitectura medallón (Bronze → Silver → Gold) en Databricks y Unity Catalog:

| Capa | Descripción |
| :---- | :---- |
| Bronze | Datos crudos ingestados desde el CSV tal cual, sin transformaciones |
| Silver | Datos limpios: manejo de nulos, validación de tipos, eliminación de duplicados, normalización de `ProductType`, codificación de variables categóricas si aplica |
| Gold | Datos listos para ML: feature engineering, selección de variables predictoras y separación de las 3 resistencias como targets |

**Criterios de limpieza esperados (Silver):**

* Detección y tratamiento de valores nulos  
* Detección y tratamiento de outliers  
* Verificación de consistencia de tipos  
* Eliminación de registros duplicados  
* Análisis de multicolinealidad entre variables predictoras

**Criterios de preparación esperados (Gold):**

* Separación clara entre features (`X`) y targets (`Strength1`, `Strength7`, `Strength28`)  
* Escalado o normalización si el modelo lo requiere  
* División train/test (recomendado 80/20)

### 3.2 Análisis exploratorio de datos (EDA)

El EDA debe incluir:

* Estadísticas descriptivas de todas las variables  
* Distribuciones de cada variable (histogramas, boxplots)  
* Detección de valores nulos y su proporción por columna  
* Detección de outliers (método IQR o Z-score)  
* Matriz de correlación entre variables predictoras y las 3 resistencias objetivo  
* Análisis por `ProductType`  
* Scatter plots de las variables más correlacionadas vs cada resistencia  
* Análisis de multicolinealidad (VIF) para identificar variables redundantes

### 3.3 Entrenamiento con MLflow

Entrenar **3 algoritmos diferentes** para predecir **cada una de las 3 resistencias** (`Strength1`, `Strength7`, `Strength28`).

El entrenamiento debe cumplir con:

* Tracking con MLflow para registrar experimentos, métricas, parámetros y artefactos  
* Registro de métricas por modelo y target:  
  * `R²`  
  * `MAE`  
  * `MAPE`  
  * Gráfico de valores reales vs. predichos  
  * Gráfico de residuales  
  * Importancia de características, si es posible  
* Registro de parámetros mínimos: `model_type`, `target`, `n_features`, `train_size`, `test_size`  
* Modelos registrados en MLflow Model Registry  
* Tabla comparativa de R², MAE y MAPE para los 3 modelos × 3 targets

### 3.4 Dashboard

Crear un dashboard interactivo que muestre:

* Resumen de resistencias: distribución de `Strength1`, `Strength7` y `Strength28`  
* Comparación de modelos: R², MAE y MAPE entre los 3 modelos para cada resistencia  
* Variables de calidad más importantes para el mejor modelo de cada resistencia  
* Relación calidad-resistencia: scatter plots de variables relevantes vs cada resistencia  
* Predicciones vs. reales: gráfico de dispersión del mejor modelo para cada resistencia  
* Filtro por `ProductType`

El dashboard debe permitir responder:

> "Dado un conjunto de datos de calidad (físicas, mineralogía, química), ¿qué resistencia se obtiene y qué variables de calidad fueron más determinantes para llegar a ese resultado?"

---

## 4\. Arquitectura sugerida del proyecto

```
datathon/
├── default/
│   └── datathon_data_gcc           # Tabla origen (datos crudos)
├── bronze/
│   └── datathon_data_bronze        # Ingesta sin transformaciones
├── silver/
│   └── datathon_data_silver        # Datos limpios y validados
└── gold/
    └── datathon_data_gold          # Features + targets listos para ML
```

### Assets sugeridos en Databricks

| Asset | Tipo | Descripción |
| :---- | :---- | :---- |
| Notebook de ingesta | Notebook | Pipeline Bronze → Silver → Gold |
| Notebook de EDA | Notebook | Análisis exploratorio completo |
| Notebook de ML | Notebook | Entrenamiento de 3 modelos × 3 targets con MLflow |
| Dashboard | AI/BI Dashboard | Visualización de resistencias y variables de calidad |
| Experimento MLflow | MLflow Experiment | Tracking de métricas y parámetros |

---

## 5\. Rúbrica oficial de evaluación

La rúbrica oficial se entrega también en el archivo `evaluacion/Rubrica_Datathon_GCC.xlsx`. La evaluación usa una **escala base de 1.0 a 5.0 puntos** y pondera 6 componentes.

| Criterio | Peso | Insuficiente (1) | Aceptable (3) | Excelente (5) |
| :---- | :---- | :---- | :---- | :---- |
| 1\. Entendimiento del problema y valor de negocio | 15% | Ven el proyecto solo como un ejercicio matemático, sin ligarlo a la física o química del cemento | Comprenden la meta de predecir la resistencia, pero no articulan bien el impacto de las variables ni el ahorro real | Vinculan la precisión del modelo con impactos de negocio claros y un modelo financiero sólido |
| 2\. Pipeline de limpieza con arquitectura medallón | 20% | No hay estructura organizada ni trazabilidad | Hay limpieza básica, pero la separación Bronze/Silver/Gold es difusa o manual | Pipeline automatizado en Unity Catalog con capas claras, reproducibles y gobernadas |
| 3\. Análisis EDA y entrenamiento con logging en MLflow | 20% | Falta EDA y no hay tracking serio de experimentos | EDA básico y MLflow incompleto | EDA avanzado \+ registro impecable en MLflow de parámetros, métricas, firmas y ejemplos |
| 4\. Modelo multi input / output: calidad → resistencias | 20% | El modelo no procesa correctamente las entradas o tiene errores inviables para operar | El modelo funciona, pero queda por debajo del benchmark mínimo en horizontes críticos | El modelo alcanza o supera el benchmark de precisión para R1, R7 y R28 |
| 5\. Dashboard interactivo | 15% | Tablero inexistente, estático o confuso | Dashboard funcional pero rígido | Dashboard interactivo de alta calidad con simulación y alertas visuales dinámicas |
| 6\. Participación grupal y sesión de Q\&A | 10% | El trabajo está centralizado en una sola persona | El equipo expone en conjunto, pero el dominio técnico está concentrado | Todos los integrantes defienden la solución con dominio técnico integral |

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Cómo se calcula

* Cada criterio se califica en una escala de **1 a 5**.  
* El puntaje ponderado se calcula como `peso × calificación`.  
* La suma ponderada define la **calificación base obligatoria**, con máximo de **5.0 puntos**.

---

## 6\. Consideraciones importantes

* `Strength1`, `Strength7` y `Strength28` son targets y no deben usarse como features.  
* `ProductType` debe tratarse como variable categórica.  
* Documenten todas las decisiones de limpieza y feature engineering.  
* Garanticen reproducibilidad fijando `random_state` o equivalente en todos los modelos.  
* La solución debe correr de punta a punta sin intervención manual del equipo durante la evaluación.

---

## 7\. Nota para todos los retos: tips y restricciones

### Tip 1\. Suban los datos a un volumen dentro de un esquema y un catálogo

En Databricks, los archivos del reto se gobiernan con Unity Catalog. Recomendamos esta estructura:

1. Crear (o usar) un **catálogo**, por ejemplo `datathon`.  
2. Crear un **esquema** para el reto, por ejemplo `datathon.gcc`.  
3. Crear un **volumen** dentro del esquema, por ejemplo `datathon.gcc.raw`.  
4. Subir `datos/datathon_data_gcc.csv` al volumen desde **Catalog \> (catálogo) \> (esquema) \> (volumen) \> Upload to this volume**.  
5. Ingerir el CSV desde el volumen para crear las tablas Bronze, Silver y Gold del reto.

Tener los datos en un volumen y en tablas de Unity Catalog facilita su trazabilidad, reutilización y gobierno en notebooks, dashboards, modelos y experimentos.

### Tip 2\. Usen Genie Code

Genie Code es el asistente de IA integrado en Databricks. Les puede ayudar a:

* Explorar el dataset y entender sus nulos, outliers y correlaciones.  
* Generar y depurar código SQL y Python para limpieza, EDA y entrenamiento.  
* Diseñar el pipeline medallón y documentar la solución.  
* Registrar correctamente experimentos en MLflow.  
* Preparar el dashboard y mejorar la reproducibilidad del proyecto.

Sean específicos en sus instrucciones (tablas, rutas del volumen, resultado esperado) y validen siempre el resultado.

### Restricción: límites de consumo de Databricks Free Edition

Databricks Free Edition tiene **límites de consumo diario** de cómputo. Esta restricción simula, de forma intencional, el uso de *budgets* y límites de consumo que existen en un entorno empresarial real: diseñen su solución pensando en eficiencia de costos.

Recomendaciones:

* Cada integrante del equipo puede usar su propia cuenta de Databricks Free Edition.  
* Si una cuenta alcanza su límite diario, **muevan el código y los objetos** (notebooks, archivos, datos del volumen, tablas, experimentos) a la Free Edition de otro integrante para continuar.  
* Mantener el código en un repositorio Git o exportar notebooks facilita este traslado.  
* Escriban su solución de forma que se pueda recrear desde cero, lo que además suma puntos en reproducibilidad.  
* Prefieran consultas acotadas y eviten dejar recursos encendidos sin uso.

¡Éxito a todos los equipos\!
