import os
import pandas as pd
from src.features import feature_extraction

#So far chekcing this like whta was done in the explore notebook, now trying in main block of code for two datasets
# used these two as they are the smallest and largest datasets, so we can see if the code works for both small and large datasets
def main():
    print("Starting feature extraction process...")
    # 1. Process a Healthy Baseline file
    df_baseline = feature_extraction('data/Segmented_Linear_Baseline.mat', 'Baseline')
    
    # 2. Process a Faulty file (Let's use Misalignment as an example)
    df_misaligned = feature_extraction('data/Segmented_Machining_Misalignment.mat', 'Misalignment')
    
    # 3. Combine them into one master dataset
    print("Combining datasets...")
    master_dataset = pd.concat([df_baseline, df_misaligned], ignore_index=True)
    
    # 4. Save the final table as a CSV so we can easily look at it
    output_path = 'results/master_features.csv'
    master_dataset.to_csv(output_path, index=False)
    
    print(f" Success! Master dataset saved to {output_path}")
    print(f"Total rows: {len(master_dataset)} | Total columns: {len(master_dataset.columns)}")

if __name__ == "__main__":
    main()