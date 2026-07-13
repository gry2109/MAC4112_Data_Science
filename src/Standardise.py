from sklearn.preprocessing import StandardScaler
import pandas as pd

def feature_scaling(df, feature_cols):
    """
    Scales the features in the DataFrame using StandardScaler.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing features to be scaled.
    feature_cols (list): List of column names to be scaled.
    
    Returns:
    pd.DataFrame: A new DataFrame with scaled features.
    """
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    return df_scaled