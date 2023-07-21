import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import os 
import re
from store_sales.logger import logging
import yaml

class LabelEncoderTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_encoded = X.copy()
        for column in X_encoded.columns:
            X_encoded[column] = X_encoded[column].astype('category').cat.codes
        return X_encoded


def label_encode_categorical_columns(data: pd.DataFrame, categorical_columns, target_column='sales'):


    # Create the pipeline with the LabelEncoderTransformer
    pipeline = Pipeline([
        ('label_encoder', LabelEncoderTransformer())
    ])

    # Apply label encoding to categorical columns in the input DataFrame
    df = data.copy()
    # Apply label encoding to categorical columns
    df_encoded = pipeline.fit_transform(df[categorical_columns])

    # Combine encoded categorical columns with other columns
    df_combined = pd.concat([df_encoded, df.drop(categorical_columns, axis=1)], axis=1)

    return df_combined


class BatchPrediction:
    def __init__(self, model_file,data,
                 exog_columns, target_column,
                drop_columns,
                label_encode,
                group_column,
                sum_column,
                mean_column):
        # Load the trained SARIMAX model
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
            
            # Exog data 
            self.exog_columns=exog_columns
            
            # Target Columns 
            self.target_column=target_column
            
            # Drop columns 
            self.drop_columns_list=drop_columns
            
            # Label encode 
            self.label_encode_columns=label_encode
            
            
            # group 
            self.group_column=group_column
            self.sum_column=sum_column
            self.mean_column=mean_column
    def get_model_name_from_yaml(self,file_path):
        """
        Extracts the model name from a YAML file.

        Args:
            file_path (str): The path to the YAML file.

        Returns:
            str: The name of the model.
        """
        try:
            # Read the YAML file
            with open(file_path, 'r') as file:
                yaml_data = yaml.safe_load(file)

            # Get the model name from the YAML data
            model_name = yaml_data['model_name']

            return model_name

        except FileNotFoundError:
            print(f"Error: The file '{file_path}' does not exist.")
            return None


    def drop_columns(self,df,drop_columns):
        # List of columns to drop
        columns_to_drop = drop_columns

        # Drop the columns from the DataFrame
        df = df.drop(columns=columns_to_drop)
        
        
       
        drop_columns = ['year', 'month', 'week', 'quarter', 'day_of_week']

        # Check if the columns exist
        existing_columns = [col for col in drop_columns if col in df.columns]

        # Drop the existing columns
        df.drop(existing_columns, axis=1, inplace=True)
    
        return df

    def group_data(self,df, group_columns, sum_columns, mean_columns):
        """
        Groups the data in the DataFrame based on the specified group_columns,
        calculates the sum of sum_columns within each group, and calculates
        the mean of mean_columns within each group.
        
        Args:
            df (pandas.DataFrame): The input DataFrame.
            group_columns (list): A list of column names to group the data by.
            sum_columns (list): A list of column names to calculate the sum within each group.
            mean_columns (list): A list of column names to calculate the mean within each group.
            
        Returns:
            pandas.DataFrame: The modified DataFrame with group-wise sums and means.
        """
        # Group the data and calculate the sum of sum_columns within each group
        df_gp = df.groupby(group_columns)[sum_columns].sum()

        # Calculate the mean of mean_columns within each group
        df_gp[mean_columns] = df.groupby(group_columns)[mean_columns].mean()

        return df_gp
    def Sarima_predict(self, data):
        
        # Accessing necessary Data 
        exog_columns=['oil_price']
        target_column=self.target_column
        # LAbel encode columns 
        label_encode_columns=self.label_encode_columns
        
        
        df=data.copy()
        
        # Setting Date column as index 
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Dropping unncessry columns 
        df=self.drop_columns(df)
        #df.to_csv("After_drop.csv")
        '''
        # Assuming you have a DataFrame called 'df' with a column named 'holiday_type'
        df['holiday_type'] = df['holiday_type'].astype('category')
        
        # Perform label encoding on categorical columns
        df = label_encode_categorical_columns(df,categorical_columns=label_encode_columns,target_column='sales')
        #df.to_csv("label_encode.csv")
  
        df_gp=self.group_data(df,
                              sum_columns=self.sum_column,
                              group_columns=self.group_column,
                              mean_columns=self.mean_column)
        df_gp.to_csv('grouped.csv')
        '''
        # Extract the time series data and exogenous variables
        df_gp=df.groupby('date')['oil_price']
        exog_data = df_gp[exog_columns]
        
        #df_gp[target_column].to_csv('targest.csv')
        #exog_data.to_csv('exog_data.csv')
        # Make predictions
        predictions = self.model.get_prediction(exog=exog_data)
        predicted_values = predictions.predicted_mean
        
        predicted_values.to_csv('predicted.csv')

        # Get the last few 100 values
        last_few_values = df_gp.iloc[-100:]

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(last_few_values.index, last_few_values[target_column], label='Actual')
        plt.plot(last_few_values.index, predicted_values[-100:], label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Time Series Prediction')
        plt.legend()

        # Create the batch prediction folder if it doesn't exist
        if not os.path.exists('batch_prediction'):
            os.makedirs('batch_prediction')

        # Save the plot in the batch prediction folder
        plot_file_path = os.path.join('batch_prediction', 'plot.png')
        plt.savefig(plot_file_path)
        plt.close()

        # Return the path to the plot file
        return plot_file_path
       
    
    def Prophet_predict(self,data):
        logging.info("Prophet prediction started")
        # Accessing necessary Data 
        """
        exog_columns=['oil_price']
        target_column=self.target_column
        
        drop_columns=self.drop_columns_list
        logging.info("Columns dropped")
        df = data.copy()
        #df.to_csv("dataframe.csv")

        logging.info("df copyed")
        # Setting Date column as index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Dropping unnecessary columns
        df = self.drop_columns(df,drop_columns)
        logging.info("unnecessary columns dropped")

        # Renaming Date column
        df = df.rename(columns={'date': 'ds'})
        logging.info("rename done")
       # df.to_csv("prophet_data.csv")
        '''
        # datatype --> category
        df = label_encode_categorical_columns(df,categorical_columns=self.label_encode_columns,target_column='sales')
        # Group data
        df_gp = self.group_data(df,
                                sum_columns=self.sum_column,
                                group_columns=self.group_column,
                                mean_columns=self.mean_column)
      #  df_gp.to_csv('grouped.csv')
    
        # Extract the time series data and exogenous variables
        time_series_data = df_gp[target_column]
        exog_data = df_gp[exog_columns]
        
       # exog_data.to_csv('exog_prophet.csv')

          '''
        
        
        df_gp=df.groupby('date')['oil_price', 'sales'].mean()

        logging.info("df_gp series created")
        #df_gp =pd.DataFrame(df_gp)
        logging.info("df_gp dataframe created")
        exog_data = df_gp[['oil_price']]
        #df_gp.to_csv("grouped_data.csv")
        #df_gp.csv("Grouped_data")
        
        # Prepare the input data for prediction
        df_copy = df_gp.copy()
        df_copy['ds'] = pd.to_datetime(df_copy.index)
        df_copy = df_copy.rename(columns={'sales': 'y'})

        # Include exogenous variables
        if exog_columns is not None:
            for column in exog_columns:
                if column in df_copy.columns:
                    df_copy[column] = exog_data[column].values.astype(float)
                else:
                    raise ValueError(f"Column '{column}' not found in the input data.")
        #df.to_csv("exog_column.csv ")
        new_df = df.copy()

        # ... (previous code remains the same) ...

        # Make predictions
        logging.info("Predictions started")
        predictions = self.model.predict(df_copy)
        logging.info("Prediction completed")
        logging.info(f"{predictions}")
        logging.info(f"{predictions.columns}")

        logging.info(f"{type(predictions)}")

        # Get the last few 100 values
        last_few_values = new_df.iloc[-100:]

        # Create a new DataFrame for predictions with 'ds' and 'yhat' columns
        last_few_predictions = pd.DataFrame({
            'ds': pd.to_datetime(predictions['ds']),  # Convert 'ds' to datetime format
            'yhat': predictions['yhat']
        })

        # Set the 'ds' column as the index for both DataFrames
        last_few_predictions.set_index('ds', inplace=True)

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(last_few_values.index, last_few_values[target_column], label='Actual', color='blue')
        plt.plot(last_few_predictions.index, last_few_predictions['yhat'], label='Predicted', color='orange')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Time Series Prediction')
        plt.legend()
        """
       # Accessing necessary Data 
        exog_columns = ['oil_price']
        target_column = self.target_column
        drop_columns = self.drop_columns_list

        df = data.copy()

        # Setting Date column as index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Dropping unnecessary columns
        df = self.drop_columns(df, drop_columns)

        # Renaming Date column
        df = df.rename(columns={'date': 'ds'})
        # df.to_csv("prophet_data.csv")

        # Group data and include the 'target_column' and 'exog_columns'
        df_gp = df.resample('D').mean()

        # Include exogenous variables
        if exog_columns is not None:
            for column in exog_columns:
                if column in df_gp.columns:
                    df_gp[column] = df_gp[column].values.astype(float)
                else:
                    raise ValueError(f"Column '{column}' not found in the input data.")

        # Prepare the input data for prediction
        df_gp['ds'] = df_gp.index  # Creating the 'ds' column using the resampled index
        df_gp = df_gp.rename(columns={"sales": 'y'})  # Renaming the target column to 'y'

        # Make predictions
        predictions = self.model.predict(df_gp)

        # Get the last few 100 values
        last_few_values = df_gp.iloc[-100:]

        # Get the corresponding predictions for the last 100 values
        last_few_predictions = predictions[predictions['ds'].isin(last_few_values.index)]

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(last_few_values.index, last_few_values['y'], label='Actual')
        plt.plot(last_few_predictions['ds'], last_few_predictions['yhat'], label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Time Series Prediction')
        plt.legend()





            
        # Create the batch prediction folder if it doesn't exist
        if not os.path.exists('batch_prediction'):
            os.makedirs('batch_prediction')

        # Save the plot in the batch prediction folder
        plot_file_path = os.path.join('batch_prediction', 'plot.png')
        
        plt.savefig(plot_file_path)
        plt.show()
        plt.close()
        
        # Round the values in the 'yhat' column to two decimal places
        last_few_predictions['yhat'] = last_few_predictions['yhat'].round(2)

        # Convert numpy array to DataFrame
        prediction_df = pd.DataFrame({'prediction': last_few_predictions['yhat'].values})

        # Save DataFrame as CSV
        prediction_csv = 'prediction.csv'
        prediction_path = os.path.join('batch_prediction', prediction_csv)
        prediction_df.to_csv(prediction_path, index=False)

        # Return the path to the plot file
        return plot_file_path
    def prediction(self,data):
        
        exog_columns=['oil_price']
        target_column=self.target_column
        model_file = self.model
        name = self.get_model_name_from_yaml(file_path=r"C:\Users\Sumeet Maheshwari\Desktop\end to end project\sales_store\Store_sales_forcasting_Timeseries\saved_models\model.yaml")
        
        if name:
            logging.info(f"Model Name :{name}")
            print(f"The model name is: {name}")

            # Check if the model is "sarima" or "prophet"
            if 'sarima' in name.lower():
                # Call Sarima_predict() method
                plot_file_path=self.Sarima_predict(data)
            elif 'prophet' in name.lower():
                # Call Prophet_predict() method
                plot_file_path=self.Prophet_predict(data)
            else:
                print("Unsupported model. Cannot perform prediction.")
                
        return plot_file_path
    
    
    