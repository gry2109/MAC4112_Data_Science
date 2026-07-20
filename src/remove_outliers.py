import pandas as pd
from sklearn.ensemble import IsolationForest

def remove_outliers(df, feature_cols, contamination=0.02):
    """
    Detect and remove anomalous machining runs using an Isolation Forest.

    This function applies the Isolation Forest algorithm to the selected
    numerical feature columns in order to identify observations that differ
    significantly from the majority of the dataset. Runs classified as
    anomalies are removed prior to dimensionality reduction and machine
    learning to reduce the influence of potentially erroneous or atypical
    measurements.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the extracted and standardised features.

    feature_cols : list[str]
        List of numerical feature columns used for outlier detection.

    contamination : float, optional
        Estimated proportion of observations expected to be anomalous.
        The default value of 0.02 assumes approximately 2% of the data
        contains outliers.

    Returns
    -------
    pandas.DataFrame
        A copy of the dataset with observations identified as outliers
        removed and the index reset.
    """
    iso = IsolationForest(contamination=contamination, random_state = 42)
    outlier_predictions = iso.fit_predict(df[feature_cols])
    df_clean = df[outlier_predictions ==1].reset_index(drop=True)
    dropped_count = len(df)-len(df_clean)
    print(f"Outlier rejection complete: Dropped {dropped_count} anamolous runs.")
    return df_clean
