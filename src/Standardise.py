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
    
    ################### This must be rmeoved when code is moved to pipline to be ran by main.py. main.py will handle creation of CSV files 
    # 4. Save the cleaned data to a new CSV file
    output_path = 'results/master_features_standardised.csv'
    df_cleaned.to_csv(output_path, index=False)
    print(f"Standardised data saved to {output_path}")   


    return df_scaled