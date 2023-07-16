import pandas as pd
from fbprophet import Prophet

class PredictionModel:
    def __init__(self):
        self.model = Prophet()

    def drop_columns(self, df, columns):
        return df.drop(columns, axis=1)

    def perform_instance_prediction(self, data):

        # Rename the date column as 'ds' to match Prophet's requirements
        df = df.rename(columns={'date': 'ds'})

        # Group the data by date and sum the target column
        df_gp = df.groupby('ds')['sales'].sum()

        # Extract the exogenous variable (oil_price in this case)
        exog_data = df_gp['oil_price']

        # Prepare the input data for prediction
        df = df_gp.copy()
        df['ds'] = pd.to_datetime(df.index)
        df = df.rename(columns={'sales': 'y'})

        # Include the exogenous variable in the input data
        if 'oil_price' in df.columns:
            df['oil_price'] = exog_data.values.astype(float)
        else:
            raise ValueError("Column 'oil_price' not found in the input data.")

        # Make predictions
        self.model.fit(df)
        predictions = self.model.predict(df)
        
        return predictions