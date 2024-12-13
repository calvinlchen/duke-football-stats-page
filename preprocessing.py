import pandas as pd

class Preprocessing:

    @staticmethod
    def add_day_of_year(df):
        df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
        df['DayOfYear'] = df['Datetime'].dt.dayofyear
        return df
    