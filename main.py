import os
import pandas as pd
from src.features import feature_extraction
from src.clean_data import clean_data_only
from src.Standardise import feature_scaling
from src.PCA import perform_pca
from src.PCA import plot_pca
from src.remove_outliers import remove_outliers



def main():
    """
    Main function to orchestrate the feature extraction process.
    improve description of what this does. 
    """
    ### Command Line Interface (CLI) for user input ###
    print("\n===== CNC Diagnostic Pipline Options=====")
    print("\nThis diagnostic tool performs feature extraction, cleaning, standardisation and a PCA analysis on the inputted datasets.")
    
    DATA_DIRECTORY = 'data/'
    if not os.path.exists(DATA_DIRECTORY):
        print(f"Data directory '{DATA_DIRECTORY}' does not exist. Please ensure the data files are in the correct location.")
        return
    available_DS = [f for f in os.listdir(DATA_DIRECTORY) if f.endswith('.mat')]
    if not available_DS:
        print(f"Error: No '.mat' files found within the '{DATA_DIRECTORY}' folder.")
        return
    print("\n========== AVAILABLE DATASETS ==========")
    for index, filename in enumerate(available_DS, 1):
        print(f"[{index}] {filename}")
    print("==========================================")

    selected_DS = [] 

    print("\nPlease input the datasets you wish to run.")
    print("Select either the dataset name or number from the list above one at a time. Press enter when done. Leave blank and press enter when all datasets added.")
    while True:
        user_input = input("\nEnter filename/number and press enter to finish.  ")
        # make sure user entered appropriate number of datasets 
        if user_input == "":
            if len(selected_DS) <2:
                print("\nYOu must select more thna one dataset for comparison")
                continue
            break
         # check user inputed dataset mathcin list 
        if user_input.isdigit() and 1 <= int(user_input) <= len(available_DS):
            filename = available_DS[int(user_input)-1]
        else:
            filename = user_input if user_input.endswith('.mat') else f"{user_input}"
        # Verfiy name is in file
        full_file_path = os.path.join(DATA_DIRECTORY, filename)
        if os.path.exists(full_file_path):
            if filename not in selected_DS:
                selected_DS.append(filename)
                print(f"Added: {filename}")
            else:
                print("File already added to the list.")
        else:
            print(f"Error: '{filename}' not found in '{DATA_DIRECTORY}/'. Please check spelling.")

    ### Feature extraction process ###
    
    # 1. Process the files to conduct the aalysis on
    print("Starting feature extraction process...")
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

    ### Filter anomalous runs ###
    print("Starting outlier removal...")
    feature_cols_standardised = [col for col in df_standardised.columns if col not in ['Run_Number', 'Target_Condition']]
    df_no_outliers = remove_outliers(df_standardised, feature_cols_standardised)
    df_no_outliers.to_csv('results/master_features_no_outliers.csv', index=False)

    ### PCA ###
    print("Starting PCA process...")
    feature_cols_no_outliers = [col for col in df_no_outliers.columns if col not in ['Run_Number', 'Target_Condition']]
    df_pca, variance_ratio = perform_pca(df_standardised, feature_cols_no_outliers, n_components=3)
    df_pca.to_csv('results/master_pca_features.csv', index=False)
    
    plot_pca(df_pca, variance_ratio)













##### May need to review what features I am extracting from the dataset from within the features function
##### Different signals need to be extracted, do same feautre extraction as the paper -- DO AT THE END BECAUSE ITS SLOW TO RUN.








if __name__ == "__main__":
    main()