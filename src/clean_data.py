import os
import pandas as pd
from sklearn.impute import SimpleImputer

def clean_data_only(results_dir):
    print("Starting data cleaning process...")
    # 1 load data for the cleaning process. This comes from the features definition executed in main.py
    data_path = os.path.join(results_dir, 'master_features.csv')
    df = pd.read_csv(data_path)

    # sepreate nuerical data from column headings as this will crash the imputer 
    data_identity = df[['Run_Number', 'Target_Condition']]
    data_features = df.drop(columns=['Run_Number', 'Target_Condition'])

    #3 Impute missign values with the mean of the column
    imputer = SimpleImputer(strategy='mean')
    features_array = imputer.fit_transform(data_features)
    features_cleaned = pd.DataFrame(features_array, columns=data_features.columns)
    df_cleaned = pd.concat([data_identity, features_cleaned], axis=1)


    ################### This must be rmeoved when code is moved to pipline to be ran by main.py. main.py will handle creation of CSV files 
    # 4. Save the cleaned data to a new CSV file
    output_path = 'results/master_features_cleaned.csv'
    df_cleaned.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    RESULTS_DIR = r'C:\Users\map25ger\Documents\PhD\Modules\MAC4112\Assignment\GIT_Code\MAC4112_Data_Science\results'
    clean_data_only(RESULTS_DIR)
