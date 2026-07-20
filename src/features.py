
import mat73
import pywt
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import stft


def get_feature_cols(df, exclude=['Run_Number', 'Target_Condition']):
    """
    Identifies the feature columns availible for analysis.

    This helper function returns the names f all feature columns in a dataframe while excluding metadata and target 
    label columns that should not be included in preprocessing or machine learning tasks.

    Parameters
    ----------------
    df : pandas.DataFrame
        Input dataset containing both feature variables and metadata.
    
    exclude: lsit[str], optional
        List of column names to exclude from the returned feature lsit. By default the run identifier and target 
        conditions are excluded.

    Returns
    ----------------
    list[str]
        A lsit containing the names of the feature columns available for statistical analysis and ML
    """
    return [col for col in df.columns if col not in exclude]






def feature_extraction(file_path, condition_label):
    """
    Extract statistical and time-frequency features from the dataset.

    This function loads a .mat dataset containing sensor signals collected from machining experiemnts and generates 
    a set of engineered features for each machining run. The extracted features include the mean, root mean square,
    maximum peak, skewness, kurtosis, crest factor, shape factor, impulse factor, margin factor, energy, signal 
    kurtosis and continuois wave transform (CWT).

    The resulting feature set is labelled with the corresponding machining condition and returned as a structured
    pandas dataframe for preprocessing.
    
    Paramaters
    ----------------
    file_path : str
        Path to the .mat dataset containing the sensor signals.

    condition_label : str
        Label describing the operating condition represented by the dataset. This label is assigned to every 
        extracted run.

    Returns
    ----------------
    pandas.DataFrame
        A dataframe where each row represents an individual run and each column contaisn the extracted feature or
        associated metadata.

    Notes
    ----------------
    Missing or unavailable sensor data are handled by assigning NaN values to the corresponding feature set, ensuring
    a consistent feature structure across all datasets.
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
       # 1. Unpack the standard name and the list of aliases
        for standard_name, aliases in sensor_list.items():
            resolved_key = None # This will hold the actual name we find in the file
            
            # 2. Check if any of our known alternative names exist in this specific file
            for alias in aliases:
                if alias in available_sensors:
                    resolved_key = alias
                    break 

            # 3. If found match, extract features
            if resolved_key is not None:
                # This must be indented under the if statement
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
                try:
                        # Compute Short-Time Fourier Transform (STFT)
                        frequencies, times, Zxx = stft(sensor_data, fs=10000, nperseg=256)
                        magnitudes = np.abs(Zxx)  # Magnitude spectrum over time
                        
                        # Calculate Kurtosis of the magnitudes across the time axis (axis 1)
                        freq_kurtosis = kurtosis(magnitudes, axis=1, bias=False)
                        freq_kurtosis = np.nan_to_num(freq_kurtosis)  # Clean any dead/flat frequency bands
                        
                        run_features[f'{standard_name}_SK_Mean'] = np.mean(freq_kurtosis)
                        run_features[f'{standard_name}_SK_Max'] = np.max(freq_kurtosis)
                        run_features[f'{standard_name}_SK_Std'] = np.std(freq_kurtosis)
                except Exception:
                        run_features[f'{standard_name}_SK_Mean'] = 0.0
                        run_features[f'{standard_name}_SK_Max'] = 0.0
                        run_features[f'{standard_name}_SK_Std'] = 0.0
                    
                    # D. TIME-FREQUENCY: CONTINUOUS WAVELET TRANSFORM ENERGY BANDS 
                try:
                        scales = np.arange(1, 33)  # 32 scales
                        coefficients, frequencies = pywt.cwt(sensor_data, scales, 'morl')
                        cwt_energy = np.abs(coefficients)
                        
                        # Split the 32 scale coefficients into 4 equal bands (8 scales each)
                        bands = np.array_split(cwt_energy, 4, axis=0)
                        
                        run_features[f'{standard_name}_CWT_Band1'] = np.mean(bands[0])
                        run_features[f'{standard_name}_CWT_Band2'] = np.mean(bands[1])
                        run_features[f'{standard_name}_CWT_Band3'] = np.mean(bands[2])
                        run_features[f'{standard_name}_CWT_Band4'] = np.mean(bands[3])
                except Exception:
                        run_features[f'{standard_name}_CWT_Band1'] = 0.0
                        run_features[f'{standard_name}_CWT_Band2'] = 0.0
                        run_features[f'{standard_name}_CWT_Band3'] = 0.0
                        run_features[f'{standard_name}_CWT_Band4'] = 0.0
                    
                else:
                    # Sensor exists, but run is empty -> Fill all 17 features with NaN
                    for suffix in ['Mean', 'RMS', 'Max_Peak', 'Skewness', 'Kurtosis', 
                                   'CrestFactor', 'ShapeFactor', 'ImpulseFactor', 'MarginFactor', 'Energy',
                                   'SK_Mean', 'SK_Max', 'SK_Std', 
                                   'CWT_Band1', 'CWT_Band2', 'CWT_Band3', 'CWT_Band4']:
                        run_features[f'{standard_name}_{suffix}'] = np.nan
        else:
                # Sensor is completely missing from this mat file -> Fill all 17 features with NaN
            for suffix in ['Mean', 'RMS', 'Max_Peak', 'Skewness', 'Kurtosis', 
                               'CrestFactor', 'ShapeFactor', 'ImpulseFactor', 'MarginFactor', 'Energy',
                               'SK_Mean', 'SK_Max', 'SK_Std', 
                               'CWT_Band1', 'CWT_Band2', 'CWT_Band3', 'CWT_Band4']:
                    run_features[f'{standard_name}_{suffix}'] = np.nan
        extracted_features.append(run_features)
        
    return pd.DataFrame(extracted_features)
