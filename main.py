import os
import time
import sys
import pandas as pd
import argparse
from src.features import feature_extraction, get_feature_cols
from src.clean_data import clean_data_only
from src.Standardise import feature_scaling
from src.PCA import perform_pca, plot_pca
from src.remove_outliers import remove_outliers
from src.classifier import train_diagnostic_classifier


def main(selected_DS):
    """
    Executes the complete CNC/process health assessment pipeline.

    This function coordinates an end-to-end workflow, beginning with feature extraction from the selected MATLAB 
    datasets and progresses throguh data cleaning, standardisation, outlier removal, Principle Component Analysis 
    (PCA), and Machine Learning (ML) fault classification. Intermediate datasets are generated at each tage and 
    saved as CSV files to supportthe traceability and inspection of the previous process.

    Parameters
    ----------------
    selected_DS : lsit[str]
        List of .mat dataset filenames selected for analysis. Each dataset is processed indiviually before being 
        combined into a master CSV file

    Returns
    ----------------
    None
        This function does not return a value. Processed datasets, figures and classification results are written to
        the projects 'results/' directory, and progress information is displayed within the terminal

    Notes
    ----------------
    The processing timeline follows these steps:
        1. Feature extraction
        2. Date cleaning
        3. Feature standardisation
        4. Outlier removal
        5. PCA
        6. ML model training and evaluation
    Runtime statistics are reported on completion providing feedback on pipeline performance.
    """
    DATA_DIRECTORY = 'data/'
    start_time = time.time()

    ### Feature extraction process ###
    
    # 1. Process the files to conduct the analysis on
    print("\nStarting feature extraction process...")
    DS_dict = {}
    for filename in selected_DS:
        full_file_path = os.path.join(DATA_DIRECTORY, filename)
        condition_label = filename.replace('Segmented_', '').replace('_', ' ').replace('.mat', '').title()
        print(f"Processing: {filename}, Labelled as: '{condition_label}'")
        df_feat = feature_extraction(full_file_path, condition_label)
        DS_dict[condition_label] = df_feat
    
    # 2. Combine the datasets 
    print("\nCombining selected datasets...")
    master_dataset = pd.concat(DS_dict.values(), ignore_index=True)

    # 3. Save the final table as a CSV 
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

    ### Feature Standardisation ###
    print("\nStarting feature standardisation process...")
    df_standardised = feature_scaling(df_cleaned, get_feature_cols(df_cleaned))
    df_standardised.to_csv('results/master_features_standardised.csv', index=False)

    ### Filter anomalous results ###
    print("\nStarting outlier removal...")
    df_no_outliers = remove_outliers(df_standardised, get_feature_cols(df_standardised))
    df_no_outliers.to_csv('results/master_features_no_outliers.csv', index=False)

    ### Principle Component Analysis ###
    print("\nStarting PCA process...")
    df_pca, variance_ratio = perform_pca(df_no_outliers, get_feature_cols(df_no_outliers), n_components=3)
    df_pca.to_csv('results/master_pca_features.csv', index=False)
    plot_pca(df_pca, variance_ratio)

    ### ML classification ###
    print("\nInitiating Machine Learning Diagnostics...")
    train_diagnostic_classifier('results/master_pca_features.csv')

    ### Timing ###
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("\n=================== EXECUTION METRICS ===================")
    if elapsed_time > 60:
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        print(f"Total Pipeline Runtime: {minutes}m {seconds:.2f}s")
    else:
        print(f"Total Pipeline Runtime: {elapsed_time:.2f} seconds")
    print("=========================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNC Machine Diagnostic and Fault Classification Pipeline")

    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="One or more .mat datasets to analyse (e.g., data/Segmented_Linear_Baseline.mat data/Segmented_Machining_baseline.mat. To run the files, type e.g.: python main.py --datasets Segmented_Linear_Baseline.mat Segmented_Machining_Baseline.mat"
    )
    args = parser.parse_args()
    

    # Validation logic
    DATA_DIRECTORY = 'data/'
    resolved_files = []
    # Directory check
    if not os.path.exists(DATA_DIRECTORY):
        parser.error(f"Error: The database folder '{DATA_DIRECTORY}' does not exist")
    available_files = [f for f in os.listdir(DATA_DIRECTORY) if f.endswith('.mat')]

    try:
        for item in args.datasets:
            item_lower = item.lower()

            if item_lower == 'all':
                print(f"Adding all (len{available_files} .mat files...)")
                resolved_files.extend(available_files)
                continue
            if not item_lower.endswith('.mat'):
                keyword_matches = [f for f in available_files if item_lower in f.lower()]
                if keyword_matches:
                    print(f"Keyword '{item}' matched: {keyword_matches}")
                    resolved_files.extend(keyword_matches)
                    continue
                else:
                    raise ValueError(f"Keyword '{item}' did not match any files in '{DATA_DIRECTORY}'.")
            filename = os.path.basename(item)
            full_path = os.path.join(DATA_DIRECTORY, filename)
            
            if not os.path.exists(full_path):
                raise ValueError(f"File '{filename}' not found in database folder '{DATA_DIRECTORY}'.")
            
            resolved_files.append(filename)
            
        # Deduplicate the resolved files list
        final_unique_files = list(set(resolved_files))
        if len(final_unique_files) < 2:
             raise ValueError(
                 f"The pipeline resolved to only {len(final_unique_files)} dataset(s): {final_unique_files}.\n"
                 f"You must select at least 2 datasets for comparison."
             )
        
        # Pass the verified list of files directly to main()
        main(final_unique_files)
        
    except ValueError as err:
        # Exit cleanly and print the error message and available datasets
        available_DS = [f for f in os.listdir(DATA_DIRECTORY) if f.endswith('.mat')]
        print(f"\nError: {str(err)}")
        print("========== AVAILABLE DATASETS ==========")
        for index, filename in enumerate(available_DS, 1):
            print(f"[{index}] {filename}")
        print("==========================================")
        sys.exit(1)

