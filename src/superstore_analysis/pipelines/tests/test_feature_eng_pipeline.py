import pandas as pd
import pytest
from superstore_analysis.pipelines.feature_eng_pipeline import FeatureEng
from superstore_analysis.datasets import DataLoader

NEW_FEATURES = [
    'Month_Order', 'Day_Order',
   	'Days_Shipping', 'Max_Sales_Month',
   	'Min_Sales_Month', 'Mean_Sales_Month'
]
@pytest.fixture(scope='module')
def processed_data():
	raw_data = DataLoader().from_gdrive(file_id='13YrRW9ufAJ_WDGn7BYFJ6XUh5G1W3h2T')
	transformed_data = FeatureEng().fit_transform(raw_data)
	return raw_data, transformed_data

def test_transformed_data_type(processed_data):
	_, transformed_data = processed_data
	assert isinstance(transformed_data, pd.DataFrame), "Output data should be dataframe format"

def test_transformed_data_shape(processed_data):
	raw_data, transformed_data = processed_data
	assert transformed_data.shape[0] == raw_data.shape[0], "Row count mismatch after transformation"


@pytest.mark.parametrize("expected_col", NEW_FEATURES)
def test_column_existence(processed_data, expected_col):	
	_, transformed_data = processed_data
	assert expected_col in transformed_data.columns, f"Missing expected columns: {expected_col}"

@pytest.mark.parametrize("expected_col", NEW_FEATURES)
def test_no_nulls_in_new_features(processed_data, expected_col):
	_, transformed_data = processed_data
	assert transformed_data[expected_col].isnull().sum() == 0, f"New feature `{expected_col}` has a missing value"





