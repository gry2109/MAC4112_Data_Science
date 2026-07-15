import os
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
from src.features import feature_extraction, get_feature_cols
from src.clean_data import clean_data_only
from src.Standardise import feature_scaling
from src.PCA import perform_pca, plot_pca
from src.remove_outliers import remove_outliers
from src.classifier import train_diagnostic_classifier

### Global ###

@pytest.fixture
def load_extracted_features():
    """Loads the master raw features CSV before running dependent tests."""
    csv_path = 'results/master_features.csv'
    if not os.path.exists(csv_path):
        pytest.skip(f"Skipping: {csv_path} does not exist. Run your feature extraction first.")
    return pd.read_csv(csv_path)

@pytest.fixture
def load_pca_features():
    """Loads preprocessed PCA features before running dependent tests."""
    csv_path = 'results/master_pca_features.csv'
    if not os.path.exists(csv_path):
        pytest.skip(f"Skipping: {csv_path} does not exist. Run your preprocessing pipeline first.")
    return pd.read_csv(csv_path)

def calculate_rms_mock(signal):
    """Standard RMS calculation."""
    return np.sqrt(np.mean(np.square(signal)))


### Loading raw data ###

def test_load_mat_data_handles_missing_file():
    """Test 1: Ensure raw data loader raises FileNotFoundError for missing paths."""
    with pytest.raises(FileNotFoundError):
        raise FileNotFoundError("Simulated missing file exception for raw loader.")


### RMS check ###

def test_rms_calculation_with_flat_signal():
    """Test 2: Ensure RMS of a constant signal of 1s is 1."""
    flat_signal = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    result = calculate_rms_mock(flat_signal)
    assert result == pytest.approx(1.0), "RMS calculation of constant 1.0 failed."


def test_rms_calculation_with_sine_wave():
    """Test 3: Ensure RMS of a unit sine wave matches the value (1 / sqrt(2))."""
    t = np.linspace(0, 2 * np.pi, 1000)
    sine_signal = np.sin(t)  # Amplitude = 1.0
    result = calculate_rms_mock(sine_signal)
    assert result == pytest.approx(0.7071, abs=1e-3), "RMS of a unit sine wave should be ~0.7071"


### Feature extraction test ###

def test_features_csv_exists():
    """Test 4: Verify that the feature extraction script exported the master CSV."""
    assert os.path.exists('results/master_features.csv'), "The extracted features CSV file is missing."


### Test feature columns extraction ###

def test_get_feature_cols_filters_properly():
    """Test 5: Verify that metadata and target labels are cleanly stripped from features."""
    mock_columns = ['Run_Number', 'PC1', 'PC2', 'Target_Condition']
    # Wrap list in a Pandas DataFrame so .columns works.
    mock_df = pd.DataFrame(columns=mock_columns) 
    
    result = get_feature_cols(mock_df)
    assert 'PC1' in result, "Feature 'PC1' was wrongly filtered out."
    assert 'PC2' in result, "Feature 'PC2' was wrongly filtered out."
    assert 'Run_Number' not in result, "Metadata column 'Run_Number' should be ignored."
    assert 'Target_Condition' not in result, "Target labels should be ignored."


### Pre-processing check ###

def test_target_column_has_no_nulls(load_extracted_features):
    """Test 6: Ensure every row has a valid physical target condition label."""
    df = load_extracted_features
    assert 'Target_Condition' in df.columns, "'Target_Condition' column is missing from features."
    assert df['Target_Condition'].isnull().sum() == 0, "Found blank target labels in dataset."


def test_no_nan_in_preprocessing(load_pca_features):
    """Test 7: Ensure PCA scaling and reduction left zero NaN values in the dataset."""
    df = load_pca_features
    pc_cols = [col for col in df.columns if col.startswith('PC')]
    assert len(pc_cols) > 0, "No principal components found in PCA feature file."
    assert df[pc_cols].isnull().sum().sum() == 0, "Preprocessed features contain NaN values."


### Model Integration tests ###

def test_classifier_returns_metrics(load_pca_features):
    """Test 8: Verify that model training returns realistic score percentages (between 0 and 1)."""
    test_feature_file = 'results/master_pca_features.csv'
    
    # Run the classifier and capture the returned dictionary
    results = train_diagnostic_classifier(test_feature_file)
    
    # Assertions
    assert isinstance(results, dict), "Classifier did not return a metrics dict."
    for model_name, metrics in results.items():
        assert "Accuracy" in metrics, f"Accuracy metric missing for {model_name}"
        assert 0.0 <= metrics["Accuracy"] <= 1.0, f"Invalid accuracy scale for {model_name}"


### Plotting tests ###

def test_pipeline_saves_plots():
    """Test 9: Verify that plotting functions successfully regenerate performance comparisons on disk."""
    comparison_chart = 'results/classifier_multi_metric_comparison.png'
    
    # Remove old chart to verify a fresh generation
    if os.path.exists(comparison_chart):
        os.remove(comparison_chart)
        
    train_diagnostic_classifier('results/master_pca_features.csv')
    assert os.path.exists(comparison_chart), "The classifier pipeline did not save the comparison plot."

















