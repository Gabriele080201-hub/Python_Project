import pandas as pd 
import numpy as np 

def drop_uninformative_columns(df, cols=None):
    """
    Drops non-informative columns and returns:
    - cleaned DataFrame
    - feature columns (graph nodes)
    """

    if cols is None:
        cols = [
            'setting1','setting2', 'setting3',
            'sensor1', 'sensor5', 'sensor6',
            'sensor10', 'sensor16', 'sensor18', 'sensor19'
        ]

    df_clean = df.copy()
    to_drop = [c for c in cols if c in df_clean.columns]
    df_clean.drop(columns=to_drop, inplace=True)

    # feature columns = everything except id, cycle, RUL
    non_feature_cols = ["id", "cycle", "RUL"]
    feature_cols = [c for c in df_clean.columns if c not in non_feature_cols]

    return df_clean, feature_cols

def compute_rul(df, id_col="id", cycle_col="cycle"):
    """
    Calcola la RUL per ogni riga:
        RUL = max_cycle(id) - cycle
    """
    df_rul = df.copy()
    max_cycles = df_rul.groupby(id_col)[cycle_col].max()
    df_rul["RUL"] = df_rul[id_col].map(max_cycles) - df_rul[cycle_col]
    return df_rul


def rul_cap(df, max_rul=125):
    """
    Applica un limite superiore alla RUL (C-MAPSS richiede capping).
    """
    if "RUL" not in df.columns:
        raise ValueError("RUL must be computed before capping.")

    df_capped = df.copy()
    df_capped["RUL"] = df_capped["RUL"].clip(upper=max_rul)
    return df_capped

