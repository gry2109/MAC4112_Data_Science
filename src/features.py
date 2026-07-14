
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
                    'PlateLFAccX': ['PlateLFAccX'],
                    'PlateLFAccY': ['PlateLFAccY'],
                    'PlateLFAccZ': ['PlateLFAccZ'],
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

            # 3. If found match, extract features
            if resolved_key is not None:
                # This must be INDENTED under the if statement
                sensor_data = dataset[resolved_key][i]
                
                if sensor_data is not None and np.size(sensor_data) > 1:
                    sensor_data = np.array(sensor_data)
                    
                    if standard_name == 'Power':
                        sensor_data = np.hstack(sensor_data).flatten()
                        sensor_data = ((sensor_data * 2500) / 10) * 3
                    
                    mean_val = np.mean(sensor_data)
                    rms_val = np.sqrt(np.mean(sensor_data**2))
                    max_peak_val = np.max(np.abs(sensor_data))
                    abs_mean_val = np.mean(np.abs(sensor_data))
                    mean_sqrt_val = np.mean(np.sqrt(np.abs(sensor_data)))
                    
                    #Standard features
                    run_features[f'{standard_name}_Mean'] = mean_val
                    run_features[f'{standard_name}_RMS'] = rms_val
                    run_features[f'{standard_name}_Max_Peak'] = max_peak_val
                    run_features[f'{standard_name}_Skewness'] = skew(sensor_data)
                    run_features[f'{standard_name}_Kurtosis'] = kurtosis(sensor_data)
                    
                    # Dimensionless features
                    # Protect against division-by-zero for flatlining/dead signals
                    if rms_val > 0 and abs_mean_val > 0 and mean_sqrt_val > 0:
                        run_features[f'{standard_name}_CrestFactor'] = max_peak_val / rms_val
                        run_features[f'{standard_name}_ShapeFactor'] = rms_val / abs_mean_val
                        run_features[f'{standard_name}_ImpulseFactor'] = max_peak_val / abs_mean_val
                        run_features[f'{standard_name}_MarginFactor'] = max_peak_val / (mean_sqrt_val**2)
                    else:
                        run_features[f'{standard_name}_CrestFactor'] = 0.0
                        run_features[f'{standard_name}_ShapeFactor'] = 0.0
                        run_features[f'{standard_name}_ImpulseFactor'] = 0.0
                        run_features[f'{standard_name}_MarginFactor'] = 0.0

                    # Energy calculation (Total signal power)
                    run_features[f'{standard_name}_Energy'] = np.mean(sensor_data**2)
                    
                    
                else:
                    # The sensor exists, but THIS specific run is totally empty
                    run_features[f'{standard_name}_Mean'] = np.nan
                    run_features[f'{standard_name}_RMS'] = np.nan
                    run_features[f'{standard_name}_Max_Peak'] = np.nan
                    run_features[f'{standard_name}_Skewness'] = np.nan
                    run_features[f'{standard_name}_Kurtosis'] = np.nan
                    run_features[f'{standard_name}_CrestFactor'] = np.nan
                    run_features[f'{standard_name}_ShapeFactor'] = np.nan
                    run_features[f'{standard_name}_ImpulseFactor'] = np.nan
                    run_features[f'{standard_name}_MarginFactor'] = np.nan
                    run_features[f'{standard_name}_Energy'] = np.nan
            else:
                # 4. The sensor is entirely missing from the file
                run_features[f'{standard_name}_Mean'] = np.nan
                run_features[f'{standard_name}_RMS'] = np.nan
                run_features[f'{standard_name}_Max_Peak'] = np.nan
                run_features[f'{standard_name}_Skewness'] = np.nan
                run_features[f'{standard_name}_Kurtosis'] = np.nan
                run_features[f'{standard_name}_CrestFactor'] = np.nan
                run_features[f'{standard_name}_ShapeFactor'] = np.nan
                run_features[f'{standard_name}_ImpulseFactor'] = np.nan
                run_features[f'{standard_name}_MarginFactor'] = np.nan
                run_features[f'{standard_name}_Energy'] = np.nan
        extracted_features.append(run_features)
        
    return pd.DataFrame(extracted_features)




