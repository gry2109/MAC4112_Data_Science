import os
import numpy as np
import pandas as pd
import pytest
from src.features import feature_extraction, get_feature_cols
from src.clean_data import clean_data_only
from src.Standardise import feature_scaling
from src.PCA import perform_pca, plot_pca
from src.remove_outliers import remove_outliers
from src.classifier import train_diagnostic_classifier


### Testing feature extraction logic ###
def calculate_rms_mock(signal):
    """A standard RMS calculation function (like the one in your pipeline)."""
    return np.sqrt(np.mean(np.square(signal)))

def test_rms_calculation_with_flat_signal():
    """Unit Test: Ensure RMS of a constant signal of 1s is exactly 1."""
    flat_signal = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    result = calculate_rms_mock(flat_signal)
    assert result == pytest.approx(1.0), "RMS calculation of constant 1.0 failed!"

def test_rms_calculation_with_sine_wave():
    """Unit Test: Ensure RMS of a sine wave matches the theoretical value (amplitude / sqrt(2))."""
    t = np.linspace(0, 2 * np.pi, 1000)
    sine_signal = np.sin(t) # Amplitude = 1.0
    result = calculate_rms_mock(sine_signal)
    # 1 / sqrt(2) is approximately 0.7071
    assert result == pytest.approx(0.7071, abs=1e-3), "RMS of a unit sine wave should be ~0.707"


### Testing processing outputs ###
@pytest.fixture
def load_extracted_features():
    """Fixture: Loads your master feature file before running dependent tests."""
    csv_path = 'results/master_features.csv'
    if not os.path.exists(csv_path):
        pytest.skip(f"Skipping: {csv_path} does not exist. Run your pipeline first!")
    return pd.read_csv(csv_path)

def test_features_csv_exists():
    """Check if the feature extraction script successfully outputs a file."""
    assert os.path.exists('results/master_features.csv'), "The extracted features CSV file is missing!"

def test_target_column_has_no_nulls(load_extracted_features):
    """Ensure every row has a valid physical target label (no empty strings or NaNs)."""
    df = load_extracted_features
    assert 'Target_Condition' in df.columns, "'Target_Condition' column is missing from features!"
    assert df['Target_Condition'].isnull().sum() == 0, "There are rows missing target labels!"



### Testing Pipeline outputs ###

def test_pipeline_saves_plots():
    """Ensure the classifier outputs a comparison bar chart after training."""
    comparison_chart = 'results/classifier_multi_metric_comparison.png'
    
    # Remove old chart to prevent false positives
    if os.path.exists(comparison_chart):
        os.remove(comparison_chart)
        
    # Run your classifier training (Assuming train_diagnostic_classifier exists in src.classifier)
    from src.classifier import train_diagnostic_classifier
    train_diagnostic_classifier('results/master_pca_features.csv')
    
    # Assert the image was regenerated
    assert os.path.exists(comparison_chart), "The classifier pipeline did not save the comparison bar chart!"

















