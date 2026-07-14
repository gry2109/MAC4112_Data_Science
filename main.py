import os
import pandas as pd
import argparse
from src.features import feature_extraction
from src.clean_data import clean_data_only
from src.Standardise import feature_scaling
from src.PCA import perform_pca
from src.PCA import plot_pca
from src.remove_outliers import remove_outliers
from src.classifier import train_diagnostic_classifier


def main(selected_DS):
    """
    Main function to orchestrate the feature extraction process.
    improve description of what this does. 
    """
    DATA_DIRECTORY = 'data/'
    
    ### Feature extraction process ###
    
    # 1. Process the files to conduct the aalysis on
    print("\nStarting feature extraction process...")
    DS_dict = {}
    for filename in selected_DS:
        full_file_path = os.path.join(DATA_DIRECTORY, filename)
        condition_label = filename.replace('Segmented_', '').replace('_', ' ').replace('.mat', '').title()
        print(f"Processing: {filename}, LAbelled as: '{condition_label}'")
        df_feat = feature_extraction(full_file_path, condition_label)
        DS_dict[condition_label] = df_feat
    
    # 2. Combine the datasets 
    print("\nCombining selected datasets...")
    master_dataset = pd.concat(DS_dict.values(), ignore_index=True)

    # 3. Save the final table as a CSV so we can easily look at it
    output_path = 'results/master_features.csv'
    master_dataset.to_csv(output_path, index=False)
    print(f"Success! Master dataset saved to {output_path}")
    print(f"Total rows: {len(master_dataset)} | Total columns: {len(master_dataset.columns)}")

    ### Cleaning process ###
    print("\nStarting data cleaning process...")
    # 1. Call function and pass the master dataset to it, then output to a CSV file 
    df_cleaned = clean_data_only(master_dataset)
    df_cleaned.to_csv('results/master_features_cleaned.csv', index=False)
    print(f"Data cleaning complete! Cleaned dataset saved to 'results/master_features_cleaned.csv'")

    ### Standardise ###
    print("\nStarting feature standardisation process...")
    feature_cols_cleaned = [col for col in df_cleaned.columns if col not in ['Run_Number', 'Target_Condition']]
    df_standardised = feature_scaling(df_cleaned, feature_cols_cleaned)
    df_standardised.to_csv('results/master_features_standardised.csv', index=False)

    ### Filter anomalous runs ###
    print("\nStarting outlier removal...")
    feature_cols_standardised = [col for col in df_standardised.columns if col not in ['Run_Number', 'Target_Condition']]
    df_no_outliers = remove_outliers(df_standardised, feature_cols_standardised)
    df_no_outliers.to_csv('results/master_features_no_outliers.csv', index=False)

    ### PCA ###
    print("\nStarting PCA process...")
    feature_cols_no_outliers = [col for col in df_no_outliers.columns if col not in ['Run_Number', 'Target_Condition']]
    df_pca, variance_ratio = perform_pca(df_no_outliers, feature_cols_no_outliers, n_components=3)
    df_pca.to_csv('results/master_pca_features.csv', index=False)
    plot_pca(df_pca, variance_ratio)

    ### ML section ###
    print("\nInitiating Machine Learning Diagnostics...")
    train_diagnostic_classifier('results/master_pca_features.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CNC Machine Diagnostic and Fault Classification Pipeline"
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="One or more .mat datasets to analyse (e.g., data/Segmented_Linear_Baseline.mat data/Segmented_Machining_baseline.mat. To run the files, type e.g.: python main.py --datasets Segmented_Linear_Baseline.mat Segmented_Machining_Baseline.mat"
    )
    args = parser.parse_args()
    main(args.datasets)