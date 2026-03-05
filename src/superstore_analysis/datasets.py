import pathlib
import pandas as pd

script_dir = pathlib.Path(__file__).parent.resolve()

class DataLoader:
	def __init__(self):
		pass

	def from_gdrive(self, file_id: str = '13YrRW9ufAJ_WDGn7BYFJ6XUh5G1W3h2T') -> pd.DataFrame:
		url = f'https://drive.google.com/uc?id={file_id}&export=download'
		df = pd.read_csv(url, parse_dates=['Ship_Date', 'Order_Date'])
		return df
	
	def from_local(self, file_path=None):
		if file_path is None:
			file_path = script_dir.parents[0] / 'data/Clustered.csv'
		data_csv = pd.read_csv(file_path, index_col=0, parse_dates=['Ship_Date', 'Order_Date'])
		return data_csv