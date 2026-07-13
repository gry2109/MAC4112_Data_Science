import os
import pandas as pd
from src.features import feature_extraction
from src.clean_data import clean_data_only



def main():
    """
    Main function to orchestrate the feature extraction process.
    improve description of what this does. 
    """
    ### Feature extraction process ###
    print("Starting feature extraction process...")
    # 1. Process the files to conduct the aalysis on
    df_baseline = feature_extraction('data/Segmented_Linear_Baseline.mat', 'Baseline')      # this is what should be a command line interface to choose whtehr you want linear, machining etc.
    df_heavy = feature_extraction('data/Segmented_Linear_Heavy.mat', 'Heavy')
    df_override = feature_extraction('data/Segmented_Linear_Override.mat', 'Override')

    # 2. Combine them into one master dataset
    print("Combining datasets...")
    master_dataset = pd.concat([df_baseline, df_heavy, df_override], ignore_index=True)

    # 3. Save the final table as a CSV so we can easily look at it
    output_path = 'results/master_features.csv'
    master_dataset.to_csv(output_path, index=False)
    print(f"Success! Master dataset saved to {output_path}")
    print(f"Total rows: {len(master_dataset)} | Total columns: {len(master_dataset.columns)}")

    ### Cleaning process ###
    print("Starting data cleaning process...")

    df_cleaned = clean_data_only(master_dataset)
    df_cleaned.to_csv('results/master_features_cleaned.csv', index=False)
    print(f"Data cleaning complete! Cleaned dataset saved to 'results/master_features_cleaned.csv'")










if __name__ == "__main__":
    main()