
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
    sensor_list = {'PlateHFAccZ': ['PlateHFAccZ'],
                    'SpindleAccX': ['SpindleAccX', 'SpindleX'],
                    'SpindleAccY': ['SpindleAccY', 'SpindleY'],
                    'SpindleAccZ': ['SpindleAccZ', 'SpindleZ'],
                    'Power': ['Power'],
                    'Load': ['SpindleLoad'],}
    extracted_features = []
    
    
    
    # Don't all have same heading names for the datasets, need to check keys in dataset
    available_sensors = list(dataset.keys())
    first_sensor = available_sensors[0]  # Get the first sensor to determine the number of runs
    num_runs = len(dataset[first_sensor])  # Assuming all sensors have the same number of runs
    
    # Extraction loop for each sensor
    for i in range(num_runs):
        run_features = {
            'Run_Number': i + 1,
            'Target_Condition': condition_label
        }
       # 1. Unpack BOTH the standard name and the list of aliases
        for standard_name, aliases in sensor_list.items():
            resolved_key = None # This will hold the actual name we find in the file
            
            # 2. Check if any of our known alternative names exist in this specific file
            for alias in aliases:
                if alias in available_sensors:
                    resolved_key = alias
                    break 
            
            # 3. If found mathc, extract features
            if resolved_key is not None:
                wave = dataset[resolved_key][i]
                
                # NEW SAFETY CHECK: Ensure the wave is not None and has actual data points
                if wave is not None and np.size(wave) > 1:
                    run_features[f'{standard_name}_Mean'] = np.mean(wave)
                    run_features[f'{standard_name}_RMS'] = np.sqrt(np.mean(wave**2))
                    run_features[f'{standard_name}_Max_Peak'] = np.max(np.abs(wave))
                    run_features[f'{standard_name}_Skewness'] = skew(wave)
                    run_features[f'{standard_name}_Kurtosis'] = kurtosis(wave)
                else:
                    # The sensor exists, but THIS specific run is totally empty
                    run_features[f'{standard_name}_Mean'] = np.nan
                    run_features[f'{standard_name}_RMS'] = np.nan
                    run_features[f'{standard_name}_Max_Peak'] = np.nan
                    run_features[f'{standard_name}_Skewness'] = np.nan
                    run_features[f'{standard_name}_Kurtosis'] = np.nan
            else:
                # 4. The sensor is entirely missing from the file
                run_features[f'{standard_name}_Mean'] = np.nan
                run_features[f'{standard_name}_RMS'] = np.nan
                run_features[f'{standard_name}_Max_Peak'] = np.nan
                run_features[f'{standard_name}_Skewness'] = np.nan
                run_features[f'{standard_name}_Kurtosis'] = np.nan
        extracted_features.append(run_features)
        
    return pd.DataFrame(extracted_features)




