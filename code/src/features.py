
import mat73
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis


def feature_extraction(file_path, condition_label):
    """
    Reads a .mat file, extracts 5 statistical features for key sensors, 
    and returns a Pandas DataFrame. These are the mean, square root, 
    standard deviation, skew and kurtosis.
    """
    print(f"Extracting features from {file_path} for condition: {condition_label}")

    # Load the .mat file
    mat = mat73.loadmat(file_path)
    struct_name = list(mat.keys())[0]  # Get the first key in the dictionary
    dataset = mat[struct_name]

    # Define list of sensors we want to extract features from
    sensors = ['PlateHFAccZ', 'SpindleAccX', 'SpindleAccY', 'SpindleAccZ', 'Power']
    extracted_features = []
    num_runs = len(dataset['PlateHFAccZ'])
    
    # Extraction loop for each sensor
    for i in range(num_runs):
        run_features = {
            'Run_Number': i + 1,
            'Target_Condition': condition_label
        }
        
        for sensor in sensor_list:
            wave = dataset[sensor][i]
            run_features[f'{sensor}_Mean'] = np.mean(wave)
            run_features[f'{sensor}_RMS'] = np.sqrt(np.mean(wave**2))
            run_features[f'{sensor}_Max_Peak'] = np.max(np.abs(wave))
            run_features[f'{sensor}_Skewness'] = skew(wave)
            run_features[f'{sensor}_Kurtosis'] = kurtosis(wave)
            
        extracted_features.append(run_features)
        
    return pd.DataFrame(extracted_features)

