import pandas as pd

class DataCleaner:
    def __init__(self):
        pass

    def fill_missing_values(self, df):
        return df.fillna(method="ffill").fillna(method="bfill")

    def drop_missing_values(self, df):
        return df.dropna()

    def remove_duplicates(self, df):
        return df.drop_duplicates()

    def clean_strings(self, df, column):
        df[column] = df[column].str.strip().str.lower()
        return df

    def convert_to_datetime(self, df, column):
        df[column] = pd.to_datetime(df[column])
        return df

    def rename_columns(self, df, columns_dict):
        return df.rename(columns=columns_dict)

    def clean_data(self, df):
        df = self.fill_missing_values(df)
        df = self.drop_missing_values(df)
        df = self.remove_duplicates(df)
        df = self.clean_strings(df, "name")
        df = self.convert_to_datetime(df, "date")
        df = self.rename_columns(df, {"name": "full_name"})
        return df
