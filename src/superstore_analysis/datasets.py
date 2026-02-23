
import pandas as pd
class DataLoader:
	def __init__(self):
		pass

	def from_gdrive(self, file_id: str = '13YrRW9ufAJ_WDGn7BYFJ6XUh5G1W3h2T') -> pd.DataFrame:
		url = f'https://drive.google.com/uc?id={file_id}&export=download'
		df = pd.read_csv(url, parse_dates=['Ship_Date', 'Order_Date'])
		return df