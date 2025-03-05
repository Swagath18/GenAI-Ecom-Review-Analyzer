import pandas as pd

class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self):
        """Load Walmart review data from CSV"""
        df = pd.read_csv(self.file_path, encoding="utf-8")  # Assuming tab-separated
        #print(df.head(2))
        return df
