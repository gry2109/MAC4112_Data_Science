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


if __name__ == "__main__":
    print("Running standalone feature scaling test...")
    
    input_path = "results/master_features_cleaned.csv"
    output_path = "results/master_features_standardised.csv"
    
    try:
        # 1. Load your raw extracted/cleaned features 
        df_raw = pd.read_csv(input_path)
        
        # 2. Isolate just the numeric columns for scaling
        metadata_cols = ['Run_Number', 'Target_Condition']
        features_to_scale = [col for col in df_raw.columns if col not in metadata_cols]
        
        # 3. Pass it to the function and CAPTURE the returned result
        df_scaled_output = feature_scaling(df=df_raw, feature_cols=features_to_scale)
        
        # 4. Save the cleanly scaled data to a new CSV file for local debugging
        df_scaled_output.to_csv(output_path, index=False)
        print(f"Standalone standardisation successful! Saved to {output_path}") 
        
    except FileNotFoundError:
        print(f"Error: Could not find '{input_path}'. Make sure your cleaning script ran first!")