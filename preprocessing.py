"""
=============================================================================
MODULE 2 - DATA UNDERSTANDING & PREPROCESSING
Project : Spatial-Temporal Mining of Urban Heatwave Precursors and
          Regional Vulnerability Clustering
Course  : Data Mining (6th Semester)
File    : preprocessing.py
=============================================================================
Responsibilities
----------------
1. Load all CSV files from the datasets/ folder.
2. Merge them on a common datetime index.
3. Identify and handle missing values via forward-fill + column-mean imputation.
4. Remove noise (IQR-based outlier capping).
5. Compute descriptive statistics (mean, median, variance).
6. Normalise continuous features using Min-Max scaling.
7. Discretise temperature into Heatwave risk categories.
8. Save the cleaned dataset and print a detailed summary report.
=============================================================================
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

DATASET_DIR   = os.path.join(os.path.dirname(__file__), "datasets")
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CSV         = os.path.join(OUTPUT_DIR, "preprocessed_data.csv")
OUTPUT_DISC_CSV    = os.path.join(OUTPUT_DIR, "discretized_data.csv")

# Temperature thresholds (Kelvin – the dataset stores values in Kelvin)
# These map to approximate Celsius equivalents:
#   < 294 K  (~21°C)  -> Normal
#   294–300 K (~21–27°C) -> Heat Watch
#   300–306 K (~27–33°C) -> Heat Warning
#   ≥ 306 K  (>33°C)  -> Extreme Heat
TEMP_BINS   = [-np.inf, 294, 300, 306, np.inf]
TEMP_LABELS = ["Normal", "Heat_Watch", "Heat_Warning", "Extreme_Heat"]

# Continuous features to normalise
CONT_FEATURES = ["temperature", "humidity", "pressure", "wind_speed"]


# ------------------------------------------------------------------------------
# HELPER : pretty section header
# ------------------------------------------------------------------------------

def _header(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ------------------------------------------------------------------------------
# STEP 1 : LOAD DATA
# ------------------------------------------------------------------------------

def load_data() -> dict:
    """
    Load each CSV file into a Pandas DataFrame.

    Returns
    -------
    dict
        Keys: 'temperature', 'humidity', 'pressure',
              'wind_speed', 'weather_description', 'city_attributes'
    """
    _header("STEP 1 – LOADING RAW CSV FILES")
    files = {
        "temperature"         : "temperature.csv",
        "humidity"            : "humidity.csv",
        "pressure"            : "pressure.csv",
        "wind_speed"          : "wind_speed.csv",
        "weather_description" : "weather_description.csv",
        "city_attributes"     : "city_attributes.csv",
    }

    dfs = {}
    for key, fname in files.items():
        path = os.path.join(DATASET_DIR, fname)
        df   = pd.read_csv(path)
        dfs[key] = df
        print(f"  [OK] Loaded {fname:35s} -> shape {df.shape}")

    return dfs


# ------------------------------------------------------------------------------
# STEP 2 : MERGE ON DATETIME
# ------------------------------------------------------------------------------

def merge_dataframes(dfs: dict) -> pd.DataFrame:
    """
    Reshape each wide-format time-series CSV (rows = timestamps,
    columns = cities) into a long format, then merge all four
    numerical sources into a single tidy DataFrame.

    Parameters
    ----------
    dfs : dict   Output of load_data()

    Returns
    -------
    pd.DataFrame
        Columns: datetime, city, temperature, humidity, pressure, wind_speed
    """
    _header("STEP 2 – MERGING DATA ON DATETIME")

    numerical_keys = ["temperature", "humidity", "pressure", "wind_speed"]
    merged = None

    for key in numerical_keys:
        df = dfs[key].copy()

        # The first column is the datetime string
        datetime_col = df.columns[0]
        df.rename(columns={datetime_col: "datetime"}, inplace=True)
        df["datetime"] = pd.to_datetime(df["datetime"])

        # Melt wide -> long
        df_long = df.melt(id_vars="datetime", var_name="city", value_name=key)

        if merged is None:
            merged = df_long
        else:
            merged = pd.merge(merged, df_long, on=["datetime", "city"], how="outer")

    print(f"  [OK] Merged DataFrame shape : {merged.shape}")
    print(f"  [OK] Date range             : {merged['datetime'].min()}  ->  {merged['datetime'].max()}")
    print(f"  [OK] Unique cities          : {merged['city'].nunique()}")
    return merged


# ------------------------------------------------------------------------------
# STEP 3 : HANDLE MISSING VALUES & NOISE
# ------------------------------------------------------------------------------

def handle_missing_and_noise(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Report missing-value counts before imputation.
    2. Forward-fill within each city group (temporal continuity).
    3. Fill any remaining NaNs with the column mean.
    4. Cap outliers using 1.5×IQR (Winsorization) per numeric column.

    Parameters
    ----------
    df : pd.DataFrame   Output of merge_dataframes()

    Returns
    -------
    pd.DataFrame   Cleaned DataFrame
    """
    _header("STEP 3 – MISSING VALUE IMPUTATION & NOISE REMOVAL")

    # -- Missing values -------------------------------------------------------
    missing_before = df[CONT_FEATURES].isnull().sum()
    print("\n  Missing values BEFORE imputation:")
    for col, cnt in missing_before.items():
        pct = 100 * cnt / len(df)
        print(f"    {col:15s}: {cnt:7,d}  ({pct:.2f}%)")

    # Forward-fill per city (preserves temporal patterns)
    df.sort_values(["city", "datetime"], inplace=True)
    df[CONT_FEATURES] = (
        df.groupby("city")[CONT_FEATURES]
          .transform(lambda s: s.ffill())
    )

    # Remaining NaNs -> column mean
    for col in CONT_FEATURES:
        col_mean = df[col].mean()
        df[col].fillna(col_mean, inplace=True)

    missing_after = df[CONT_FEATURES].isnull().sum()
    print("\n  Missing values AFTER imputation:")
    for col, cnt in missing_after.items():
        print(f"    {col:15s}: {cnt:7,d}")

    # -- Outlier capping (IQR) ------------------------------------------------
    print("\n  Outlier capping (Winsorization at 1.5×IQR):")
    for col in CONT_FEATURES:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        print(f"    {col:15s}: {n_outliers:7,d} values capped  "
              f"[{lower:.3f}, {upper:.3f}]")

    return df


# ------------------------------------------------------------------------------
# STEP 4 : DESCRIPTIVE STATISTICS
# ------------------------------------------------------------------------------

def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean, median, and variance for each continuous feature.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame   Statistics table
    """
    _header("STEP 4 – BASIC DESCRIPTIVE STATISTICS")

    stats = pd.DataFrame({
        "Mean"     : df[CONT_FEATURES].mean(),
        "Median"   : df[CONT_FEATURES].median(),
        "Variance" : df[CONT_FEATURES].var(),
        "Std Dev"  : df[CONT_FEATURES].std(),
        "Min"      : df[CONT_FEATURES].min(),
        "Max"      : df[CONT_FEATURES].max(),
    }).round(4)

    print(stats.to_string())
    return stats


# ------------------------------------------------------------------------------
# STEP 5 : NORMALISATION
# ------------------------------------------------------------------------------

def normalise_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Min-Max scaling to all continuous features.
    New columns are named  <feature>_norm  so original values are retained.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame   DataFrame with added *_norm columns
    """
    _header("STEP 5 – MIN-MAX NORMALISATION")

    scaler   = MinMaxScaler()
    norm_arr = scaler.fit_transform(df[CONT_FEATURES])
    norm_cols = [f"{c}_norm" for c in CONT_FEATURES]

    df[norm_cols] = norm_arr

    print("  Normalised columns added (Min-Max -> [0, 1]):")
    for col in norm_cols:
        print(f"    {col:25s}  min={df[col].min():.4f}  max={df[col].max():.4f}")

    return df


# ------------------------------------------------------------------------------
# STEP 6 : DISCRETISATION
# ------------------------------------------------------------------------------

def discretise_temperature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin the raw temperature column into ordinal risk categories using
    pre-defined Kelvin thresholds.

    Adds column  'heat_risk'  to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    _header("STEP 6 – TEMPERATURE DISCRETISATION (HEAT RISK LEVELS)")

    df["heat_risk"] = pd.cut(
        df["temperature"],
        bins=TEMP_BINS,
        labels=TEMP_LABELS,
        right=True
    )

    dist = df["heat_risk"].value_counts().sort_index()
    print("  Heat-risk category distribution:")
    for label, count in dist.items():
        pct = 100 * count / len(df)
        bar = "#" * int(pct // 2)
        print(f"    {label:15s}: {count:9,d}  ({pct:5.1f}%)  {bar}")

    return df


# ------------------------------------------------------------------------------
# STEP 7 : DISCRETISE OTHER FEATURES (for Pattern Mining)
# ------------------------------------------------------------------------------

def discretise_features_for_mining(df: pd.DataFrame) -> pd.DataFrame:
    """
    Discretise humidity, pressure, and wind_speed into Low/Medium/High
    tertile bins so they can be used directly by the Apriori / FP-Growth
    algorithms in Module 3.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame   A reduced DataFrame with only the discretised columns
                   (city, datetime, heat_risk, hum_level, pres_level, wind_level)
    """
    _header("STEP 7 – DISCRETISING ALL FEATURES FOR PATTERN MINING")

    disc = df[["datetime", "city", "heat_risk"]].copy()

    feature_map = {
        "humidity"   : "hum_level",
        "pressure"   : "pres_level",
        "wind_speed" : "wind_level",
    }

    for src, tgt in feature_map.items():
        disc[tgt] = pd.qcut(
            df[src],
            q=3,
            labels=["Low", "Medium", "High"],
            duplicates="drop"
        )
        counts = disc[tgt].value_counts().sort_index()
        print(f"\n  {tgt} distribution:")
        for lvl, cnt in counts.items():
            print(f"    {lvl:6s}: {cnt:,}")

    return disc


# ------------------------------------------------------------------------------
# STEP 8 : SAVE OUTPUTS
# ------------------------------------------------------------------------------

def save_outputs(df_clean: pd.DataFrame, df_disc: pd.DataFrame) -> None:
    """
    Persist the two processed DataFrames to the outputs/ folder.

    Parameters
    ----------
    df_clean : pd.DataFrame   Full preprocessed dataset (with _norm columns)
    df_disc  : pd.DataFrame   Discretised dataset for pattern mining
    """
    _header("STEP 8 – SAVING OUTPUT FILES")

    df_clean.to_csv(OUTPUT_CSV, index=False)
    print(f"  [OK] Cleaned & normalised data  -> {OUTPUT_CSV}")
    print(f"      Shape : {df_clean.shape}")

    df_disc.to_csv(OUTPUT_DISC_CSV, index=False)
    print(f"  [OK] Discretised data           -> {OUTPUT_DISC_CSV}")
    print(f"      Shape : {df_disc.shape}")


# ------------------------------------------------------------------------------
# SUMMARY REPORT
# ------------------------------------------------------------------------------

def print_summary_report(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    stats: pd.DataFrame,
    elapsed: float
) -> None:
    """Print a concise end-of-pipeline summary."""

    _header("PREPROCESSING SUMMARY REPORT")

    sep = "=" * 60
    print(f"""
  {sep}
  PIPELINE SUMMARY
  {sep}
  Raw records loaded                 : {len(df_raw):>15,}
  Records after merge & clean        : {len(df_clean):>15,}
  Total columns in cleaned dataset   : {len(df_clean.columns):>15}
  Continuous features normalised     : {4:>15}
  Temperature bins created           : {len(TEMP_LABELS):>15}
  Total pipeline time                : {elapsed:>12.2f}  s
  {sep}

  Temperature Discretisation Thresholds (Kelvin):
    Normal       : T  < 294 K  (approx < 21 C)
    Heat Watch   : 294 <= T < 300 K  (approx 21-27 C)
    Heat Warning : 300 <= T < 306 K  (approx 27-33 C)
    Extreme Heat : T >= 306 K  (approx > 33 C)

  Output Files:
    1. {OUTPUT_CSV}
    2. {OUTPUT_DISC_CSV}
  {sep}
""")


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

def main():
    t_start = time.time()

    # 1. Load
    dfs = load_data()

    # 2. Merge
    df = merge_dataframes(dfs)
    df_raw_snapshot = df.copy()          # keep raw shape for report

    # 3. Clean
    df = handle_missing_and_noise(df)

    # 4. Statistics
    stats = compute_statistics(df)

    # 5. Normalise
    df = normalise_features(df)

    # 6. Discretise temperature -> heat_risk
    df = discretise_temperature(df)

    # 7. Discretise all features -> pattern-mining ready CSV
    df_disc = discretise_features_for_mining(df)

    # 8. Save
    save_outputs(df, df_disc)

    elapsed = time.time() - t_start

    # 9. Summary report
    print_summary_report(df_raw_snapshot, df, stats, elapsed)


if __name__ == "__main__":
    main()
