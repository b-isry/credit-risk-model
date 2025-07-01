import pytest
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_processing import load_data 

# Test Case 1: Test successful data loading
def test_load_data_success():
    """
    Tests that the load_data function correctly reads a CSV file
    and returns a pandas DataFrame.
    """
    # Create a dummy CSV file for testing
    dummy_file_path = 'tests/fixtures/dummy.csv'
    
    # Run the function
    df = load_data(dummy_file_path)
    
    # Assertions
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == ['col1', 'col2']
    assert len(df) == 2

# Test Case 2: Test handling of a non-existent file
def test_load_data_file_not_found():
    """
    Tests that the load_data function raises a FileNotFoundError
    when the file path is invalid.
    """
    invalid_path = 'path/to/non_existent_file.csv'
    
    # Use pytest.raises to check if the expected exception is raised
    with pytest.raises(FileNotFoundError):
        load_data(invalid_path)