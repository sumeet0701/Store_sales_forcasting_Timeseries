import os  
import sys 
from store_sales.config.configuration import *
from store_sales.entity.config_entity import DataIngestionConfig,DataValidationConfig
from store_sales.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from store_sales.config.configuration import Configuration
from store_sales.exception import ApplicationException
from store_sales.logger import logging
from store_sales.utils.utils import read_yaml_file
from store_sales.entity.raw_data_validation import IngestedDataValidation
import shutil
from store_sales.constant import *
import pandas as pd
import json

class DataValidation:
    def __init__(self,data_validation_config:DataValidationConfig,
                 data_ingestion_artifact:DataIngestionArtifact) -> None:
        try:
            logging.info(f"{'>>' * 30}Data Validation log started.{'<<' * 30} \n\n") 
            
            # Creating_instance           
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            
            # Schema_file_path
            self.schema_path = self.data_validation_config.schema_file_path
            
            
            # creating instance for row_data_validation
            self.train_data = IngestedDataValidation(
                                validate_path=self.data_ingestion_artifact.Ingestion_file_path, schema_path=self.schema_path)

            
            # Data_ingestion_artifact--->Unvalidated train and test file path
            self.ingested_file_path = self.data_ingestion_artifact.Ingestion_file_path
 
            
            # Data_validation_config --> file paths to save validated_data
            self.validated_file_path = self.data_validation_config.file_path
            
        
        except Exception as e:
            raise ApplicationException(e,sys) from e



    def isFolderPathAvailable(self) -> bool:
        try:

             # True means avaliable false means not avaliable
             
            isfolder_available = False
            datafile_path=self.ingested_file_path

            if os.path.exists(datafile_path):
                    isfolder_available = True
            return isfolder_available
        except Exception as e:
            raise ApplicationException(e, sys) from e     
      


        
    def is_Validation_successfull(self):
        try:
            validation_status = True
            logging.info("Validation Process Started")
            if self.isFolderPathAvailable() == True:
                train_filename = os.path.basename(
                    self.data_ingestion_artifact.Ingestion_file_path)

                is_train_filename_validated = self.train_data.validate_filename(
                    file_name=train_filename)

                is_train_column_name_same = self.train_data.check_column_names()

                is_train_missing_values_whole_column = self.train_data.missing_values_whole_column()

                self.train_data.replace_null_values_with_null()
                 

                logging.info(
                    f"Train_set status|is Train filename validated?: {is_train_filename_validated}|is train column name validated?: {is_train_column_name_same}|whole missing columns?{is_train_missing_values_whole_column}")

                if is_train_filename_validated  & is_train_column_name_same & is_train_missing_values_whole_column:
                    ## Exporting Train.csv file 
                    # Create the directory if it doesn't exist
                    os.makedirs(self.validated_file_path, exist_ok=True)

                    # Copy the CSV file to the validated train path
                    shutil.copy(self.ingested_file_path, self.validated_file_path)
                    # Log the export of the validated train dataset
                    logging.info(f"Exported validated  dataset to file: [{self.validated_file_path}]")
                                     
                    return validation_status,self.validated_file_path
                else:
                    validation_status = False
                    logging.info("Check yout Data! Validation Failed")
                    raise ValueError(
                        "Check your data! Validation failed")
                

            return validation_status,"NONE"
        except Exception as e:
            raise ApplicationException(e, sys) from e      



    def initiate_data_validation(self):
        try:
            is_validated, validated_file_path = self.is_Validation_successfull()
            
            if is_validated is True:
                message='Validated'
            else:
                message='Not Validated'


            data_validation_artifact = DataValidationArtifact(
                schema_file_path=self.schema_path,
                message=message,
                validated_file_path = validated_file_path
                
            )
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise ApplicationException(e, sys) from e


    def __del__(self):
        logging.info(f"{'>>' * 30}Data Validation log completed.{'<<' * 30}")
