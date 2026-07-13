import os
import pandas as pd
from sklearn.impute import SimpleImputer

def clean_data_only(df):
     """
     Reads the csv file master_features from the results directory, cleans the data by 
     imputing missing values with the mean of each column, and saves the cleaned data 
     to a new CSV file.
     """
     print("Starting data cleaning process...")
     # 1 load data for the cleaning process. This comes from the features definition executed in main.py
     
     # sepreate nuerical data from column headings as this will crash the imputer 
     data_identity = df[['Run_Number', 'Target_Condition']]
     data_features = df.drop(columns=['Run_Number', 'Target_Condition'])

     #3 Impute missign values with the mean of the column
     imputer = SimpleImputer(strategy='mean')
     features_array = imputer.fit_transform(data_features)
     features_cleaned = pd.DataFrame(features_array, columns=data_features.columns)
     df_cleaned = pd.concat([data_identity, features_cleaned], axis=1)

     return df_cleaned

if __name__ == "__main__":
    RESULTS_DIR = r'C:\Users\map25ger\Documents\PhD\Modules\MAC4112\Assignment\GIT_Code\MAC4112_Data_Science\results'
    input_csv = os.path.join(RESULTS_DIR, 'master_features.csv')
    output_csv = os.path.join(RESULTS_DIR, 'master_features_cleaned.csv')
    
    print(f"Running standalone debugging test...")
    
    try:
        # 1. Load the raw data yourself for this test
        df_raw = pd.read_csv(input_csv)
        
        # 2. Pass it to your pure function and catch the result
        df_clean_output = clean_data_only(df_raw)
        
        # 3. Save the result to a CSV for local debugging
        df_clean_output.to_csv(output_csv, index=False)
        print(f"Debug clean successful! Saved to {output_csv}")
        
    except FileNotFoundError:
        print(f"Error: Could not find {input_csv} to run standalone test.")