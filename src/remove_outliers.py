import pandas as pd
from sklearn.ensemble import IsolationForest

def remove_outliers(df, feature_cols, contamination=0.02):
    """
    
    """
    iso = IsolationForest(contamination=contamination, random_state = 42)
    outlier_predictions = iso.fit_predict(df[feature_cols])
    df_clean = df[outlier_predictions ==1].reset_index(drop=True)
    dropped_count = len(df)-len(df_clean)
    print(f"Outlier rejection complete: Dropped {dropped_count} anamolous runs.")
    return df_clean
