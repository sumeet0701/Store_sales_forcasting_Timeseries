from store_sales.exception import ApplicationException
from store_sales.logger import logging
from store_sales.utils.utils import read_yaml_file
from store_sales.utils.utils import save_array_to_directory
from store_sales.utils.utils import save_data
from store_sales.utils.utils import save_object
from store_sales.entity.config_entity import DataIngestionConfig
from store_sales.entity.config_entity import DataValidationConfig
from store_sales.entity.config_entity import DataTransformationConfig
from store_sales.entity.artifact_entity import DataIngestionArtifact
from store_sales.entity.artifact_entity import DataValidationArtifact
from store_sales.entity.artifact_entity import DataTransformationArtifact
from store_sales.constant import *
from store_sales.constant.training_pipeline.data_ingestion import *
from store_sales.constant.training_pipeline.data_validation import *
from store_sales.constant.training_pipeline.data_transformation import *
from store_sales.constant.training_pipeline.model_trainer_time import *
from store_sales.constant.training_pipeline.Model_trainer import *
from store_sales.constant.training_pipeline import *


import numpy as np
import pandas as pd
import sys 
import re

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder

class Feature_Engineering(BaseEstimator, TransformerMixin):
    
    def __init__(self,
                 numerical_columns,
                 categorical_columns,
                 target_columns,
                 drop_columns,
                 date_column,
                 all_column,
                 time_series_data_path,
                 time_drop_columns ):
        """
        This class applies necessary Feature Engneering 
        """
        logging.info(f"\n{'*'*20} Feature Engneering Started {'*'*20}\n\n")
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.target_columns = target_columns
        self.date_column=date_column
        self.columns_to_drop = drop_columns
        self.time_drop_columns=time_drop_columns
        self.col=all_column
        self.time_series_data_path=time_series_data_path
        logging.info(f" Numerical Columns , Categorical Columns , Target Column initialised in Feature engineering Pipeline ")
        
    def drop_columns(self,df: pd.DataFrame):
        try:
            fe_drop = ['year', 'month', 'week', 'quarter', 'day_of_week']
            columns_to_drop = [column for column in fe_drop if column in df.columns]
            columns_not_found = [column for column in fe_drop if column not in df.columns]
            if len(columns_not_found) > 0:
                logging.info(f"Columns not found: {columns_not_found}")
                return df
            logging.info(f"Dropping columns: {columns_to_drop}")
            df.drop(columns=columns_to_drop, axis=1, inplace=True)
            logging.info(f"Columns after dropping: {df.columns}")
            return df
        except Exception as e:
            raise ApplicationException(e, sys) from e
    
    def date_datatype(self,df:pd.DataFrame):
        # Convert 'date' column to datetime
        df['date'] = pd.to_datetime(df['date'])
        return df    
    
    def convert_columns_to_category(self,df, columns):
        for col in columns:
            df[col] = df[col].astype('category')
            logging.info(f"Column '{col}' converted to 'category' data type.")
                    # Dropping columns 
        return df
    
    def convert_columns_to_category_Check(self,df, columns):
        for col in columns:
            df[col] = df[col].astype('category')
            logging.info(f"Column '{col}' converted to 'category' data type.")
        return df
    
    def remove_special_chars_and_integers_from_unique_values(self,df, column_name):
        # Remove special characters and integers from unique values
        df[column_name] = df[column_name].apply(lambda x: re.sub(r'[^a-zA-Z\s:]', '', x))
        df[column_name] = df[column_name].apply(lambda x: re.sub(r'\d+', '', x))
        
        return df
    
    def replace_low_percentages(self,df:pd.DataFrame, column_name, threshold):
        # Calculate unique value percentages
        value_counts = df[column_name].value_counts()
        unique_value_percentages = (value_counts / len(df)) * 100

        # Identify unique values with percentage less than the threshold
        low_percentage_values = unique_value_percentages[unique_value_percentages < threshold].index

        # Replace low percentage values with 'Others'
        df[column_name].replace(low_percentage_values, 'Others', inplace=True)
        
        return df
    
    def check_duplicate_values(self,df):
        initial_shape = df.shape
        # Remove duplicates and get the modified DataFrame
        df_no_duplicates = df.drop_duplicates()
        modified_shape = df_no_duplicates.shape
        # Compute the count of duplicate values
        duplicated_count = initial_shape[0] - modified_shape[0]

        logging.info(f"Shape before removing duplicates: {initial_shape}")
        logging.info(f"Shape after removing duplicates: {modified_shape}")
        logging.info(f"Count of duplicate values: {duplicated_count}")
        return df_no_duplicates
    
    def renaming_oil_price(self,df: pd.DataFrame):
        df = df.rename(columns={"dcoilwtico": "oil_price"})
        logging.info(" Oil Price column renamed ")
        return df
    
    def missing_values_info(self,df: pd.DataFrame):
        if df.isnull().sum().sum() != 0:
            na_df = (df.isnull().sum() / len(df)) * 100
            na_df = na_df.drop(na_df[na_df == 0].index).sort_values(ascending=False)
            logging.info("Missing ratio information:")
            for column, missing_ratio in na_df.iteritems():
                logging.info(f"{column}: {missing_ratio:.2f}%")
        else:
            logging.info("No missing values found in the dataframe.")
        return df    
    
    def drop_null_unwanted_columns(self,df:pd.DataFrame):
        if 'id' in df.columns:
            logging.info("Dropping 'id' column...")
            df.drop(columns=['id'], inplace=True)
        else:
            logging.info("'id' column not found. Skipping dropping operation.")

        logging.info("Dropping rows with null values...")
        df.dropna(inplace=True)

        logging.info("Resetting DataFrame index...")
        df.reset_index(drop=True, inplace=True)

        logging.info("Columns dropped, null values removed, and index reset.")

        return df
    
    def handling_missing_values(self,df):

        logging.info("Checking 'oil_price' column for missing values...")
        missing_values = df['oil_price'].isna().sum()
        logging.info(f"Number of missing values: {missing_values}")

        logging.info("interpolate missing values in 'oil_price' column...")
        df['oil_price'].interpolate(method='linear', inplace=True)
                # Verify if missing values have been filled
        missing_values_after = df['oil_price'].isna().sum()
        logging.info(f"Number of missing values after filling: {missing_values_after}")

        columns_missing = ['holiday_type']
        logging.info(f"{df['holiday_type'].mode()}")

        for column in columns_missing:
            logging.info(f"Filling missing values in '{column}' column with mode...")
            if not df[column].empty:
                mode_value = df[column].mode().iloc[0]
                df[column].fillna(mode_value, inplace=True)
        logging.info("Missing values handled.")
        
        return df
    

    def remove_outliers_IQR(self,data, cols):
        for col in cols:
            logging.info(f"Removing outliers in column '{col}' using IQR method...")

            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            lower_limit = q1 - 1.5 * iqr
            upper_limit = q3 + 1.5 * iqr

            outliers_removed = data[(data[col] > upper_limit) | (data[col] < lower_limit)]
            num_outliers_removed = outliers_removed.shape[0]

            logging.info(f"Number of outliers removed in column '{col}': {num_outliers_removed}")

            data[col] = np.where(data[col] > upper_limit, upper_limit,
                                np.where(data[col] < lower_limit, lower_limit, data[col]))

            logging.info(f"Column '{col}' modified: {num_outliers_removed} outliers modified")

        return data
    
    
    def convert_numerical_to_categorical(self,df, columns):
        converted_df = df.copy()  # Create a copy of the original DataFrame
        
        for column in columns:
            converted_df[column] = converted_df[column].astype(str).astype('category')
        
        return converted_df
    
    def map_categorical_values(self,df:pd.DataFrame):
        logging.info("Mapping unique values of categorical columns:")
        for column in df.select_dtypes(include=['category']):
            unique_values = df[column].unique()
            mapping = {value: i for i, value in enumerate(unique_values)}
            df[column] = df[column].map(mapping)
            logging.info(f"Column: '{column}', Unique Values: {unique_values}")
            
        return df
    
    def run_data_modification(self,df:pd.DataFrame):
          
        # Dropping Irrelevant Columns
        df= self.drop_columns(df)
        
        # Change Datatype of the column 
        df= self.date_datatype(df)
        
       # df=self.convert_numerical_to_categorical(df,self.categorical_columns)
        
        # Set categorical Columns to category 
      #  df=self.convert_columns_to_category(df,self.categorical_columns)
        
        drop_columns=self.time_drop_columns
        logging.info(f"Dropping columns for time series analysis {drop_columns}")
        df.drop(drop_columns,axis=1,inplace=True)
        
        
        # Removing special character from "Description"
        #df=self.remove_special_chars_and_integers_from_unique_values(df,'description')
        
        # Replace low percenatages unique values 
        #df=self.replace_low_percentages(df,'locale_name',0.5)
       # df=self.replace_low_percentages(df,'description',0.5)
        
        # renaming Oil_Price
        df=self.renaming_oil_price(df)
        
        # Drop Duplicated values 
        df=self.check_duplicate_values(df)
        
        # Missing Values info 
        df= self.missing_values_info(df)
        
        
        # dropping null values 
        df=self.drop_null_unwanted_columns(df)
        
        # handling missing Values 
       # df=self.handling_missing_values(df)
        
        # Outlier column
        outliers_mod_columns=['oil_price']
        
        df=self.remove_outliers_IQR(df,outliers_mod_columns)
        
        # Exported data
        # df.to_csv('removed_outliers.csv') 
        return df

    def data_wrangling(self,df:pd.DataFrame):
        try:

            # Data Modification 
            data_modified=self.run_data_modification(df)

            logging.info(f"Columns after modification {data_modified.columns}")
            
            logging.info(" Data Modification Done")
            
            # Groupping the data as necessary for time series prediction 
            logging.info("Column Data Types:")
            for column in data_modified.columns:
                logging.info(f"Column: '{column}': {data_modified[column].dtype}")

            return data_modified
    
        
        except Exception as e:
            raise ApplicationException(e,sys) from e

        
    def fit(self,X,y=None):
            return self
    
    
    def transform(self,X:pd.DataFrame,y=None):
        try:    
            data_modified = self.data_wrangling(X)
            col = self.col

            # Reindex the DataFrame columns according to the specified column sequence
            data_modified = data_modified.reindex(columns=col)

            #data_modified.to_csv("data_modified.csv", index=False)
            
            logging.info("Data Wrangling Done")
                
            return data_modified
        except Exception as e:
            raise ApplicationException(e,sys) from e


class DataTransformation:
    
    
    def __init__(self, data_transformation_config: DataTransformationConfig,
                    data_ingestion_artifact: DataIngestionArtifact,
                    data_validation_artifact: DataValidationArtifact):
        try:
            logging.info(f"\n{'*'*20} Data Transformation log started {'*'*20}\n\n")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact

            # Schema File path 
            self.schema_file_path = self.data_validation_artifact.schema_file_path
            
            # Reading data in Schema 
            self.schema = read_yaml_file(file_path=self.schema_file_path)
            
            # Time series transaformed csv path 
            self.time_series_data_path=self.data_transformation_config.time_series_data_file_path
            
            # Column data accessed from Schema.yaml
            self.target_column_name = self.schema[TARGET_COLUMN_KEY]
            self.numerical_column_without_target=self.schema[NUMERICAL_COLUMN_WITHOUT_TAR]
            self.categorical_columns = self.schema[CATEGORICAL_COLUMN_KEY]
            self.date_column=self.schema[DATE_COLUMN]
            
            self.drop_columns=self.schema[DROP_COLUMN_KEY]
            
            # Columns dropped for time analysis 
  
            self.time_config=read_yaml_file(file_path=TIME_CONFIG_FILE_PATH)
            self.time_drop_columns=self.time_config[DROP_COLUMNS]
            self.time_map_encoding=self.time_config[ENCODE_COLUMNS]
      
            
            
            # groupping the data 
            # Grouping columns 
            self.group_column=self.time_config[GROUP_COLUMN]
            self.sum_column = self.time_config[SUM_COLUMN]
            self.mean_column=self.time_config[MEAN_COLUMN]
            
            
            self.col=self.numerical_column_without_target+self.categorical_columns+self.date_column+self.target_column_name

        except Exception as e:
            raise ApplicationException(e,sys) from e


    def get_feature_engineering_object(self):
        try:
            
            feature_engineering = Pipeline(steps = [("fe",Feature_Engineering(numerical_columns=self.numerical_column_without_target,
                                                                            categorical_columns=self.categorical_columns,
                                                                            target_columns=self.target_column_name,
                                                                            date_column=self.date_column,
                                                                            all_column=self.col,
                                                                            drop_columns=self.drop_columns,
                                                                            time_series_data_path=self.time_series_data_path,
                                                                            time_drop_columns=self.time_drop_columns,
                                                                           ))])
            return feature_engineering
        except Exception as e:
            raise ApplicationException(e,sys) from e

        
    def initiate_data_transformation(self):
        try:
            
            logging.info(f"Obtaining file from file path ")
            validated_file_path = self.data_validation_artifact.validated_file_path

            logging.info(f"Loading Data as pandas dataframe.")
            file_name=self.schema[FILE_NAME]
            validated_file_path=os.path.join(validated_file_path,file_name)
            file_data = pd.read_csv(validated_file_path)
            
            logging.info(f" Data columns {file_data.columns}")
            
            # Schema.yaml ---> Extracting target column name
            target_column_name = self.target_column_name
            numerical_columns_without_target = self.numerical_column_without_target
            categorical_columns = self.categorical_columns
            date_column=self.date_column
                        
            # Log column information
            logging.info("Numerical columns: {}".format(numerical_columns_without_target))
            logging.info("Categorical columns: {}".format(categorical_columns))
            logging.info("Target Column: {}".format(target_column_name))
            logging.info(f"Date column :{date_column}")
            
            
            col = self.col
            # All columns 
            logging.info("All columns: {}".format(col))
            
            
            # Reorder the columns in the DataFrame
            file_data = file_data.reindex(columns=col)

            # Feature Engineering 
            logging.info(f"Obtaining feature engineering object.")
            fe_obj = self.get_feature_engineering_object()
            
            logging.info(f"Applying feature engineering object on training dataframe and testing dataframe")
            logging.info(">>>" * 20 + " Training data " + "<<<" * 20)
            logging.info(f"Feature Enineering - Train Data ")
            feature_eng_df = fe_obj.fit_transform(file_data)
            #feature_eng_arr.to_csv('feature_eng_df.csv')
            
            
            logging.info(f"Columns for Feature Engineering : {col}")

            logging.info(f"Feature Engineering - Train Completed")
            
            #feature_eng_df.to_csv('feature_eng_df.csv')
            
            logging.info("Saving Feature Engineering Object")
            feature_engineering_object_file_path = self.data_transformation_config.feature_engineering_object_file_path
            save_object(file_path = feature_engineering_object_file_path,obj = fe_obj)
            save_object(file_path=os.path.join(ROOT_DIR,PIKLE_FOLDER_NAME_KEY,
                                 os.path.basename(feature_engineering_object_file_path)),obj=fe_obj)
            
            
            # Saving data for time series training before map encoding
            time_series_data_path=os.path.join(self.time_series_data_path,'time_model_file_name.csv')
            
            save_data(file_path = time_series_data_path, data = feature_eng_df)
            
            data_transformation_artifact = DataTransformationArtifact(
                                                                        message="Data transformation successfull.",
                                                                        time_series_data_file_path=time_series_data_path,
                                                                        feature_engineering_object_file_path = feature_engineering_object_file_path)
            
            logging.info(f"Data Transformation Artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def __del__(self):
        logging.info(f"\n{'*'*20} Data Transformation log completed {'*'*20}\n\n")