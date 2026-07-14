import os
import pandas as pd
from src.features import feature_extraction
from src.clean_data import clean_data_only
from src.Standardise import feature_scaling
from src.PCA import perform_pca
from src.PCA import plot_pca



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
    # 1. Call function and pass the master dataset to it, then output to a CSV file 
    df_cleaned = clean_data_only(master_dataset)
    df_cleaned.to_csv('results/master_features_cleaned.csv', index=False)
    print(f"Data cleaning complete! Cleaned dataset saved to 'results/master_features_cleaned.csv'")

    ### Standardise ###
    print("Starting feature standardisation process...")
    feature_cols_cleaned = [col for col in df_cleaned.columns if col not in ['Run_Number', 'Target_Condition']]
    df_standardised = feature_scaling(df_cleaned, feature_cols_cleaned)
    df_standardised.to_csv('results/master_features_standardised.csv', index=False)

    ### PCA ###
    print("Starting PCA process...")
    feature_cols_standardised = [col for col in df_standardised.columns if col not in ['Run_Number', 'Target_Condition']]
    df_pca, variance_ratio = perform_pca(df_standardised, feature_cols_standardised, n_components=3)
    df_pca.to_csv('results/master_pca_features.csv', index=False)
    
    plot_pca(df_pca, variance_ratio)












##### Tomorrow, add the standardised and PCA function calls here to complete the set
##### May need to review what features I am extracting from the dataset from within the features function
##### Different signals need to be extracted, do same feautre extraction as the paper -- DO AT THE END BECAUSE ITS SLOW TO RUN.








if __name__ == "__main__":
    main()