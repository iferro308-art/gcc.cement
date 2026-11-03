# Databricks notebook source
# MAGIC %md
# MAGIC # Pipeline medallón · Datathon GCC (Bronze → Silver → Gold)
# MAGIC
# MAGIC **Plataforma:** Databricks + Unity Catalog (Delta Lake) · **Lenguaje:** Python / PySpark + pandas + scikit-learn
# MAGIC
# MAGIC **Problema:** predecir `Strength1`, `Strength7` y `Strength28` (resistencias a 1, 7 y 28 días) a partir de propiedades físicas, fases minerales y composición química, más el tipo de producto (`ProductType`).
# MAGIC
# MAGIC | Capa | Tabla(s) Delta | Qué hace |
# MAGIC |---|---|---|
# MAGIC | **Bronze** | `bronze_datathon_gcc` | Copia fiel del CSV (todo como *string*) + metadatos de ingesta (`_row_id`, `_source_file`, `_ingest_ts`). No se corrige nada. |
# MAGIC | **Silver** | `silver_datathon_gcc` + `silver_cleaning_log`, `silver_outlier_report`, `silver_corr_pairs`, `silver_vif_report` | Tipos, duplicados, nulos, outliers, multicolinealidad. Cada decisión queda registrada en tablas. |
# MAGIC | **Gold** | `gold_X_train/test`, `gold_y_train/test`, `gold_feature_catalog`, `gold_scaler_params`, `gold_baseline_metrics` | Features `X` separados de targets `y`, one-hot de `ProductType`, escalado (fit solo en train), split 80/20 y modelos base. |
# MAGIC
# MAGIC **Ejecución sin intervención manual:** todos los parámetros tienen valor por defecto (widgets). Solo necesita que el CSV esté en alguno de los lugares que busca la celda 3 (por defecto: junto al notebook en el repo, o en el Volume `raw`). Al correrlo como **Job** o con *Run all* termina de punta a punta.
# MAGIC
# MAGIC ## Decisiones de diseño (resumen)
# MAGIC 1. **Bronze inmutable y en *string***: así la verificación de tipos en Silver detecta problemas reales en vez de que `inferSchema` los esconda.
# MAGIC 2. **Targets nunca se imputan ni se modifican** (hacerlo fabricaría etiquetas). Se descartan solo las filas sin ningún target.
# MAGIC 3. **Columnas con > 50 % de nulos se eliminan** (en el dataset entregado, `Physical_Metric_A` tiene ~75 % de nulos: imputarla inventaría la mayoría de sus valores).
# MAGIC 4. **Filas con > 50 % de predictores nulos se eliminan**: en los datos, un bloque de filas trae vacías las 8 fases minerales a la vez; imputarlas fabricaría 8 de 10 predictores.
# MAGIC 5. **Outliers de predictores: winsorización con fences de 3 × IQR** (no 1.5 × IQR, que recortaría ~15 % de algunas columnas de cola pesada como `Mineral_Phase_D`). Targets: solo se reportan.
# MAGIC 6. **Nulos restantes: mediana por `ProductType`** (con mediana global como respaldo), aplicada *después* de recortar outliers.
# MAGIC 7. **Multicolinealidad**: matriz de correlación + VIF iterativo; se elimina el predictor con mayor VIF mientras VIF > 10 (el análisis es no supervisado, no usa los targets).
# MAGIC 8. **`ProductType` es categórica**: categorías con < 10 filas se agrupan en `Other` y se codifica one-hot (referencia = categoría más frecuente, evita la trampa de la variable dummy).
# MAGIC 9. **Split 80/20 estratificado por `ProductType`**, escalado `StandardScaler` ajustado **solo con train** (sin *data leakage*) y solo sobre variables continuas.
# MAGIC 10. **Reproducibilidad**: una sola semilla (`SEED`) para split y modelos; lecturas ordenadas por `_row_id`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Parámetros y utilidades

# COMMAND ----------

import os, re, glob, shutil, random, math
from datetime import datetime
import numpy as np
import pandas as pd
from pyspark.sql.types import (StructType, StructField, StringType, DoubleType,
                               LongType, TimestampType, BooleanType)

def get_param(name, default):
    """Widget de Databricks con valor por defecto (cae a variable de entorno fuera de Databricks)."""
    try:
        dbutils.widgets.text(name, str(default))
        return dbutils.widgets.get(name)
    except NameError:
        return os.environ.get(name.upper(), str(default))

# --- Ubicación (Unity Catalog) ---
P_CATALOG   = get_param("catalog", "")                 # vacío = catálogo actual
SCHEMA      = get_param("schema", "datathon_gcc")
VOLUME      = get_param("volume", "raw")
CSV_NAME    = get_param("csv_name", "datathon_data_gcc.csv")
CSV_PATH    = get_param("csv_path", "")                # opcional: ruta explícita

# --- Reproducibilidad y umbrales ---
SEED               = int(get_param("seed", 42))
TEST_SIZE          = float(get_param("test_size", 0.20))
COL_NULL_THRESHOLD = float(get_param("col_null_threshold", 0.50))  # elimina columna si > 50 % nulos
ROW_NULL_THRESHOLD = float(get_param("row_null_threshold", 0.50))  # elimina fila si > 50 % de predictores nulos
IQR_K              = float(get_param("iqr_k", 3.0))                # fences para winsorizar outliers
CORR_THRESHOLD     = float(get_param("corr_threshold", 0.80))      # par "muy correlacionado"
VIF_THRESHOLD      = float(get_param("vif_threshold", 10.0))
MIN_CAT_COUNT      = int(get_param("min_cat_count", 10))           # categorías raras -> "Other"

random.seed(SEED); np.random.seed(SEED)

# --- Esquema del problema ---
TARGETS  = ["Strength1", "Strength7", "Strength28"]
CAT_COL  = "ProductType"
ROW_ID   = "_row_id"

CATALOG = P_CATALOG.strip()
if not CATALOG:
    try:
        CATALOG = spark.sql("SELECT current_catalog()").first()[0]
    except Exception:
        CATALOG = "main"

def fqn(table):
    return f"`{CATALOG}`.`{SCHEMA}`.`{table}`"

T_BRONZE = "bronze_datathon_gcc"
T_SILVER = "silver_datathon_gcc"

def _spark_type(dtype):
    if pd.api.types.is_bool_dtype(dtype):           return BooleanType()
    if pd.api.types.is_integer_dtype(dtype):        return LongType()
    if pd.api.types.is_float_dtype(dtype):          return DoubleType()
    if pd.api.types.is_datetime64_any_dtype(dtype): return TimestampType()
    return StringType()

def to_spark(pdf: pd.DataFrame):
    """pandas -> Spark con esquema explícito; NaN se escribe como NULL real."""
    schema = StructType([StructField(str(c), _spark_type(t), True) for c, t in pdf.dtypes.items()])
    clean = pdf.astype(object).where(pdf.notna(), None)
    return spark.createDataFrame(clean.values.tolist(), schema)

def save_table(pdf: pd.DataFrame, table: str):
    (to_spark(pdf).write.format("delta").mode("overwrite")
        .option("overwriteSchema", "true").saveAsTable(fqn(table)))
    print(f"✔ {CATALOG}.{SCHEMA}.{table}  ({len(pdf)} filas, {pdf.shape[1]} columnas)")

def load_table(table: str) -> pd.DataFrame:
    pdf = spark.table(fqn(table)).toPandas()
    return pdf.sort_values(ROW_ID).reset_index(drop=True) if ROW_ID in pdf.columns else pdf

def show(pdf, n=20):
    try:
        display(pdf.head(n))
    except NameError:
        print(pdf.head(n).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Infraestructura en Unity Catalog (schema + volumes)

# COMMAND ----------

def _try_sql(stmt):
    try:
        spark.sql(stmt); return True
    except Exception as e:
        print(f"[aviso] no se pudo ejecutar: {stmt}\n        {str(e)[:150]}")
        return False

_try_sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{SCHEMA}`")
_try_sql(f"CREATE VOLUME IF NOT EXISTS `{CATALOG}`.`{SCHEMA}`.`{VOLUME}`")
_try_sql(f"CREATE VOLUME IF NOT EXISTS `{CATALOG}`.`{SCHEMA}`.`artifacts`")

VOLUME_RAW = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
ARTIFACTS  = f"/Volumes/{CATALOG}/{SCHEMA}/artifacts"
if not os.path.isdir(ARTIFACTS):                      # respaldo si no hay permisos para crear Volumes
    ARTIFACTS = "/tmp/datathon_gcc_artifacts"
os.makedirs(ARTIFACTS, exist_ok=True)
print("Catálogo:", CATALOG, "| Schema:", SCHEMA, "| Artefactos:", ARTIFACTS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 🥉 BRONZE — ingesta cruda
# MAGIC El notebook localiza el CSV automáticamente (orden de búsqueda: parámetro `csv_path` → Volume → carpeta del notebook/repo → carpeta padre → `/dbfs/FileStore`). Si lo encuentra fuera del Volume, lo copia al Volume para dejar la fuente versionada junto a las tablas.

# COMMAND ----------

def resolve_csv() -> str:
    cwd = os.getcwd()
    candidates = [
        CSV_PATH,
        os.path.join(VOLUME_RAW, CSV_NAME),
        os.path.join(cwd, CSV_NAME),
        os.path.join(cwd, "data", CSV_NAME),
        os.path.join(os.path.dirname(cwd), CSV_NAME),
        os.path.join(os.path.dirname(cwd), "data", CSV_NAME),
        f"/dbfs/FileStore/tables/{CSV_NAME}",
    ]
    candidates += glob.glob(f"/Volumes/{CATALOG}/*/*/{CSV_NAME}")
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError(
        f"No se encontró '{CSV_NAME}'. Súbelo a {VOLUME_RAW}/ o colócalo junto al notebook "
        f"(o pasa el parámetro csv_path). Rutas probadas: {[c for c in candidates if c]}")

src = resolve_csv()
print("CSV fuente:", src)

# Copia al Volume (lineage) si procede
dst = os.path.join(VOLUME_RAW, CSV_NAME)
if os.path.isdir(VOLUME_RAW) and os.path.abspath(src) != os.path.abspath(dst):
    try:
        shutil.copyfile(src, dst); print("Copiado a", dst)
    except Exception as e:
        print("[aviso] no se pudo copiar al Volume:", str(e)[:120])

# Todo como string: Bronze no interpreta ni corrige nada
raw = pd.read_csv(src, dtype=str, keep_default_na=False, na_values=[""])
raw.insert(0, ROW_ID, np.arange(len(raw), dtype="int64"))
raw["_source_file"] = os.path.basename(src)
raw["_ingest_ts"]   = pd.Timestamp.now(tz="UTC").tz_localize(None)

missing_cols = [c for c in [CAT_COL] + TARGETS if c not in raw.columns]
assert not missing_cols, f"Faltan columnas esperadas en el CSV: {missing_cols}"

save_table(raw, T_BRONZE)
show(raw, 5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 🥈 SILVER — limpieza y validación
# MAGIC Cada paso agrega una fila a `silver_cleaning_log` (qué se hizo y a cuántas filas/valores afectó), de modo que la documentación se genera sola y siempre coincide con los datos.

# COMMAND ----------

log = []
def add_log(step, detail, n=0):
    log.append({"step": step, "detail": detail, "n_affected": int(n)})
    print(f"[{step}] {detail}: {int(n)}")

bronze = load_table(T_BRONZE)
df = bronze.drop(columns=["_source_file", "_ingest_ts"]).copy()
add_log("00_input", "Filas leídas de Bronze", len(df))

FEATURES = [c for c in df.columns if c not in (ROW_ID, CAT_COL, *TARGETS)]
add_log("00_input", f"Predictores numéricos detectados ({len(FEATURES)})", len(FEATURES))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Consistencia de tipos
# MAGIC Se convierten predictores y targets a `float64` (los valores no numéricos pasan a nulo y se cuentan) y se normaliza `ProductType` al formato `Type N`.

# COMMAND ----------

for c in FEATURES + TARGETS:
    s = df[c].astype("string").str.strip()
    s = s.mask(s == "", pd.NA)
    num = pd.to_numeric(s, errors="coerce").astype("float64")
    n_bad = int((s.notna() & num.isna()).sum())
    n_inf = int(np.isinf(num).sum())
    num = num.replace([np.inf, -np.inf], np.nan)
    df[c] = num
    if n_bad or n_inf:
        add_log("01_types", f"{c}: valores no numéricos/infinitos convertidos a nulo", n_bad + n_inf)

pt = df[CAT_COL].astype("string").str.strip()
num_part = pt.str.extract(r"(?i)^type\s*[_-]?\s*(\d+)$")[0]
df[CAT_COL] = ("Type " + num_part).fillna("Unknown").astype(object)
add_log("01_types", "ProductType con formato no reconocido -> 'Unknown'", (df[CAT_COL] == "Unknown").sum())
add_log("01_types", "Todas las columnas numéricas convertidas a float64; ProductType normalizado", len(FEATURES) + len(TARGETS))
print(df.dtypes.value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Duplicados y filas sin información
# MAGIC * Duplicado = misma fila en `ProductType` + predictores + targets (se ignora `_row_id`). Se conserva la primera.
# MAGIC * Se descartan filas sin **ningún** target (no sirven ni para entrenar ni para evaluar). Los targets parcialmente nulos se conservan: cada target se modela con sus filas válidas.

# COMMAND ----------

subset = [CAT_COL] + FEATURES + TARGETS
dups = df.duplicated(subset=subset, keep="first")
add_log("02_duplicates", "Filas duplicadas exactas eliminadas", dups.sum())
df = df[~dups]

no_target = df[TARGETS].isna().all(axis=1)
add_log("02_duplicates", "Filas sin ningún target eliminadas", no_target.sum())
df = df[~no_target]

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 Valores nulos (detección y tratamiento)

# COMMAND ----------

null_frac = df[FEATURES].isna().mean().sort_values(ascending=False)
show(null_frac.rename("fraccion_nulos").to_frame().reset_index().rename(columns={"index": "columna"}), 20)

# (a) columnas con demasiados nulos
drop_cols = null_frac[null_frac > COL_NULL_THRESHOLD].index.tolist()
for c in drop_cols:
    add_log("03_nulls", f"Columna eliminada por {null_frac[c]:.0%} de nulos (umbral {COL_NULL_THRESHOLD:.0%}): {c}", df[c].isna().sum())
df = df.drop(columns=drop_cols)
FEATURES = [c for c in FEATURES if c not in drop_cols]

# (b) filas con demasiados predictores nulos
row_null = df[FEATURES].isna().mean(axis=1)
bad_rows = row_null > ROW_NULL_THRESHOLD
add_log("03_nulls", f"Filas eliminadas con >{ROW_NULL_THRESHOLD:.0%} de predictores nulos", bad_rows.sum())
df = df[~bad_rows]

# (c) la imputación se hace después de tratar outliers (ver 3.5)
add_log("03_nulls", "Nulos de predictores restantes (se imputarán tras el paso de outliers)", df[FEATURES].isna().sum().sum())
add_log("03_nulls", "Nulos en targets (NO se imputan; se conservan)", df[TARGETS].isna().sum().sum())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4 Coherencia física de los targets
# MAGIC La resistencia debe crecer con la edad (`Strength1 ≤ Strength7 ≤ Strength28`). Se verifica y se reporta; no se altera ninguna etiqueta.

# COMMAND ----------

v1 = (df["Strength7"] < df["Strength1"]).sum()
v2 = (df["Strength28"] < df["Strength7"]).sum()
add_log("04_target_consistency", "Filas con Strength7 < Strength1", v1)
add_log("04_target_consistency", "Filas con Strength28 < Strength7", v2)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.5 Outliers
# MAGIC * **Predictores:** winsorización con fences `[Q1 − k·IQR, Q3 + k·IQR]`, `k = 3` (parámetro `iqr_k`). Se recorta en lugar de eliminar filas para no perder datos en un dataset pequeño.
# MAGIC * **Targets:** solo se reportan (modificar la etiqueta sesgaría la evaluación).

# COMMAND ----------

out_rows = []
for c in FEATURES:
    q1, q3 = df[c].quantile([0.25, 0.75]); iqr = q3 - q1
    if iqr == 0 or np.isnan(iqr):
        continue
    lo, hi = q1 - IQR_K * iqr, q3 + IQR_K * iqr
    n_lo, n_hi = int((df[c] < lo).sum()), int((df[c] > hi).sum())
    df[c] = df[c].clip(lo, hi)
    out_rows.append({"variable": c, "role": "predictor", "lower_fence": lo, "upper_fence": hi,
                     "n_below": n_lo, "n_above": n_hi, "pct_affected": (n_lo + n_hi) / max(len(df), 1),
                     "action": "winsorized"})
for c in TARGETS:
    q1, q3 = df[c].quantile([0.25, 0.75]); iqr = q3 - q1
    lo, hi = q1 - IQR_K * iqr, q3 + IQR_K * iqr
    n_lo, n_hi = int((df[c] < lo).sum()), int((df[c] > hi).sum())
    out_rows.append({"variable": c, "role": "target", "lower_fence": lo, "upper_fence": hi,
                     "n_below": n_lo, "n_above": n_hi, "pct_affected": (n_lo + n_hi) / max(len(df), 1),
                     "action": "reported_only"})
outlier_report = pd.DataFrame(out_rows)
add_log("05_outliers", f"Valores de predictores recortados (fences {IQR_K}×IQR)", outlier_report.query("role=='predictor'")[["n_below", "n_above"]].sum().sum())
add_log("05_outliers", "Outliers de targets detectados (sin modificar)", outlier_report.query("role=='target'")[["n_below", "n_above"]].sum().sum())
show(outlier_report, 20)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.6 Imputación de nulos restantes
# MAGIC Mediana por `ProductType`; si la categoría no tiene ningún valor, mediana global. *Nota de rigor:* la mediana se calcula con todo Silver (antes del split); el efecto de fuga es mínimo (mediana robusta, pocos valores imputados) y queda documentado. El **escalado**, que sí influye en los modelos, se ajusta solo con train en Gold.

# COMMAND ----------

for c in FEATURES:
    n = int(df[c].isna().sum())
    if n:
        grp_med = df.groupby(CAT_COL)[c].transform("median")
        df[c] = df[c].fillna(grp_med).fillna(df[c].median())
        add_log("06_imputation", f"{c}: imputados con mediana por ProductType", n)
assert df[FEATURES].isna().sum().sum() == 0, "Quedan nulos en predictores"

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.7 Multicolinealidad entre predictores
# MAGIC 1. Matriz de correlación de Pearson y lista de pares con |r| ≥ `corr_threshold`.
# MAGIC 2. **VIF** (Variance Inflation Factor) = diagonal de la inversa de la matriz de correlación.
# MAGIC 3. **Selección iterativa:** se marca para eliminar el predictor con mayor VIF mientras VIF > `vif_threshold` (10). El análisis solo usa predictores (nunca targets). Gold aplica la lista resultante.

# COMMAND ----------

def compute_vif(X: pd.DataFrame) -> pd.Series:
    std = X.std(ddof=0).replace(0, np.nan)
    Z = ((X - X.mean()) / std).dropna(axis=1, how="all")
    corr = np.corrcoef(Z.values, rowvar=False)
    return pd.Series(np.diag(np.linalg.pinv(corr)), index=Z.columns)

# pares correlacionados
corr = df[FEATURES].corr()
pairs = []
for i, a in enumerate(FEATURES):
    for b in FEATURES[i + 1:]:
        r = corr.loc[a, b]
        pairs.append({"feature_a": a, "feature_b": b, "pearson_r": float(r),
                      "abs_r": float(abs(r)), "flag_high": bool(abs(r) >= CORR_THRESHOLD)})
corr_pairs = pd.DataFrame(pairs).sort_values("abs_r", ascending=False).reset_index(drop=True)
add_log("07_multicollinearity", f"Pares con |r| ≥ {CORR_THRESHOLD}", corr_pairs["flag_high"].sum())

# VIF inicial y selección iterativa
vif_initial = compute_vif(df[FEATURES])
kept, dropped = list(FEATURES), []
while len(kept) > 2:
    v = compute_vif(df[kept])
    worst = v.idxmax()
    if v[worst] <= VIF_THRESHOLD:
        break
    dropped.append(worst); kept.remove(worst)
    add_log("07_multicollinearity", f"Predictor marcado para eliminar (VIF={v[worst]:.1f} > {VIF_THRESHOLD}): {worst}", 1)
vif_final = compute_vif(df[kept])

vif_report = pd.DataFrame({
    "feature": FEATURES,
    "vif_initial": [float(vif_initial.get(f, np.nan)) for f in FEATURES],
    "vif_final":   [float(vif_final.get(f, np.nan)) for f in FEATURES],
    "action":      ["drop_in_gold" if f in dropped else "keep" for f in FEATURES],
}).sort_values("vif_initial", ascending=False).reset_index(drop=True)
add_log("07_multicollinearity", f"Máximo VIF inicial ({vif_initial.idxmax()})", round(float(vif_initial.max())))
if not dropped:
    add_log("07_multicollinearity", "Ningún predictor supera el umbral de VIF: se conservan todos", 0)
show(vif_report, 20); show(corr_pairs, 10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.8 Persistencia de Silver y de su documentación

# COMMAND ----------

silver = df[[ROW_ID, CAT_COL] + FEATURES + TARGETS].sort_values(ROW_ID).reset_index(drop=True)
add_log("99_output", "Filas finales en Silver", len(silver))

assert silver[ROW_ID].is_unique
assert not silver.duplicated(subset=[CAT_COL] + FEATURES + TARGETS).any()
assert all(pd.api.types.is_float_dtype(silver[c]) for c in FEATURES + TARGETS)
assert silver[FEATURES].isna().sum().sum() == 0

cleaning_log = pd.DataFrame(log)
cleaning_log.insert(0, "order", np.arange(len(cleaning_log), dtype="int64"))

save_table(silver, T_SILVER)
save_table(cleaning_log, "silver_cleaning_log")
save_table(outlier_report, "silver_outlier_report")
save_table(corr_pairs, "silver_corr_pairs")
save_table(vif_report, "silver_vif_report")
show(cleaning_log, 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. 🥇 GOLD — features, split y escalado
# MAGIC
# MAGIC * `X` = predictores numéricos (menos los marcados por VIF) + one-hot de `ProductType`.
# MAGIC * `y` = `Strength1`, `Strength7`, `Strength28` **solo** en tablas `gold_y_*`; hay asserts que impiden que un target entre a `X`.
# MAGIC * Split 80/20 estratificado por `ProductType` agrupado, `random_state = SEED`.
# MAGIC * `StandardScaler` ajustado solo con train y aplicado únicamente a variables continuas (las dummies 0/1 no se escalan).

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

silver = load_table(T_SILVER)
vif_report = load_table("silver_vif_report")
dropped_vif = vif_report.loc[vif_report["action"] == "drop_in_gold", "feature"].tolist()

num_feats = [c for c in silver.columns if c not in (ROW_ID, CAT_COL, *TARGETS) and c not in dropped_vif]

# --- Feature engineering: ProductType categórica ---
counts = silver[CAT_COL].value_counts()
rare = counts[counts < MIN_CAT_COUNT].index.tolist()
silver["ProductType_grp"] = silver[CAT_COL].where(~silver[CAT_COL].isin(rare), "Other")
baseline_cat = silver["ProductType_grp"].value_counts().idxmax()          # categoría de referencia

dummies = pd.get_dummies(silver["ProductType_grp"], prefix="ProductType", dtype="int64")
dummies.columns = [re.sub(r"\W+", "_", c) for c in dummies.columns]
dummies = dummies.drop(columns=re.sub(r"\W+", "_", f"ProductType_{baseline_cat}"))
cat_feats = dummies.columns.tolist()

X_all = pd.concat([silver[num_feats], dummies], axis=1)
y_all = silver[TARGETS]
ids   = silver[ROW_ID]

# --- Guardas anti-leakage ---
assert not set(TARGETS) & set(X_all.columns), "¡Un target quedó en X!"
assert X_all.isna().sum().sum() == 0

# --- Split 80/20 reproducible ---
strat = silver["ProductType_grp"] if silver["ProductType_grp"].value_counts().min() >= 2 else None
idx_train, idx_test = train_test_split(np.arange(len(silver)), test_size=TEST_SIZE,
                                       random_state=SEED, stratify=strat)
idx_train, idx_test = np.sort(idx_train), np.sort(idx_test)

X_train, X_test = X_all.iloc[idx_train].copy(), X_all.iloc[idx_test].copy()
y_train, y_test = y_all.iloc[idx_train].copy(), y_all.iloc[idx_test].copy()
id_train, id_test = ids.iloc[idx_train].values, ids.iloc[idx_test].values
assert set(id_train).isdisjoint(id_test)

# --- Escalado: fit SOLO en train, solo variables continuas ---
scaler = StandardScaler().fit(X_train[num_feats])
X_train[num_feats] = scaler.transform(X_train[num_feats])
X_test[num_feats]  = scaler.transform(X_test[num_feats])
joblib.dump({"scaler": scaler, "num_features": num_feats, "cat_features": cat_feats, "seed": SEED},
            os.path.join(ARTIFACTS, "gold_scaler.joblib"))

print(f"Train: {len(X_train)} | Test: {len(X_test)} | Features: {X_train.shape[1]} "
      f"({len(num_feats)} continuas + {len(cat_feats)} dummies de ProductType)")
print("Categorías agrupadas en 'Other':", rare, "| Referencia one-hot:", baseline_cat)
print("Predictores eliminados por VIF:", dropped_vif or "ninguno")

# COMMAND ----------

def with_id(frame, id_values):
    out = frame.reset_index(drop=True).copy()
    out.insert(0, ROW_ID, np.asarray(id_values, dtype="int64"))
    return out

save_table(with_id(X_train, id_train), "gold_X_train")
save_table(with_id(X_test,  id_test),  "gold_X_test")
save_table(with_id(y_train, id_train), "gold_y_train")
save_table(with_id(y_test,  id_test),  "gold_y_test")

feature_catalog = pd.DataFrame(
    [{"feature": f, "kind": "numeric", "transformation": "winsorized(3*IQR) + median-impute + StandardScaler(train)",
      "source_column": f} for f in num_feats] +
    [{"feature": f, "kind": "categorical_onehot", "transformation": f"one-hot (ref={baseline_cat}; raras->Other)",
      "source_column": CAT_COL} for f in cat_feats])
save_table(feature_catalog, "gold_feature_catalog")

scaler_params = pd.DataFrame({"feature": num_feats, "mean": scaler.mean_, "scale": scaler.scale_})
save_table(scaler_params, "gold_scaler_params")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Modelos base (reproducibles)
# MAGIC Se entrena un modelo por target con las filas cuyo target no es nulo. Todos con `random_state = SEED`. Son líneas base para validar la capa Gold, no el modelo final.

# COMMAND ----------

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

models = {
    "Dummy_mean":   lambda: DummyRegressor(strategy="mean"),
    "Ridge":        lambda: Ridge(alpha=1.0, random_state=SEED),
    "RandomForest": lambda: RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
}

rows = []
for t in TARGETS:
    mtr, mte = y_train[t].notna().values, y_test[t].notna().values
    for name, make in models.items():
        m = make().fit(X_train[mtr], y_train.loc[mtr, t])
        pred = m.predict(X_test[mte])
        rows.append({"target": t, "model": name, "n_train": int(mtr.sum()), "n_test": int(mte.sum()),
                     "r2": float(r2_score(y_test.loc[mte, t], pred)),
                     "mae": float(mean_absolute_error(y_test.loc[mte, t], pred)),
                     "rmse": float(math.sqrt(mean_squared_error(y_test.loc[mte, t], pred))),
                     "seed": SEED})
metrics = pd.DataFrame(rows)
save_table(metrics, "gold_baseline_metrics")
show(metrics, 20)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Verificación final de la ejecución

# COMMAND ----------

chk_xtr, chk_xte = load_table("gold_X_train"), load_table("gold_X_test")
chk_ytr, chk_yte = load_table("gold_y_train"), load_table("gold_y_test")

assert chk_xtr[ROW_ID].tolist() == chk_ytr[ROW_ID].tolist(), "X_train / y_train desalineados"
assert chk_xte[ROW_ID].tolist() == chk_yte[ROW_ID].tolist(), "X_test / y_test desalineados"
assert not set(TARGETS) & (set(chk_xtr.columns) | set(chk_xte.columns)), "Target en X"
assert chk_xtr.drop(columns=ROW_ID).isna().sum().sum() == 0
assert set(chk_xtr[ROW_ID]).isdisjoint(chk_xte[ROW_ID])
ratio = len(chk_xte) / (len(chk_xte) + len(chk_xtr))
assert abs(ratio - TEST_SIZE) < 0.02, f"Proporción de test inesperada: {ratio:.3f}"

print(f"✅ Pipeline completado · seed={SEED} · train={len(chk_xtr)} · test={len(chk_xte)} · test%={ratio:.1%}")
print("Tablas en", f"{CATALOG}.{SCHEMA}:",
      [T_BRONZE, T_SILVER, "silver_cleaning_log", "silver_outlier_report", "silver_corr_pairs", "silver_vif_report",
       "gold_X_train", "gold_X_test", "gold_y_train", "gold_y_test", "gold_feature_catalog",
       "gold_scaler_params", "gold_baseline_metrics"])
