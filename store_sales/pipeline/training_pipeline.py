from collections import namedtuple
from datetime import datetime
import uuid
from store_sales.config.configuration  import Configuration
from store_sales.logger import logging
from store_sales.exception import ApplicationException
from threading import Thread
from typing import List
from store_sales.utils.utils import read_yaml_file
from multiprocessing import Process
from store_sales.entity.artifact_entity import *
from store_sales.components.data_ingestion import DataIngestion
from store_sales.components.data_validation import DataValidation
from store_sales.components.data_transformation import DataTransformation
from store_sales.components.model_eval_and_pusher import Model_eval_and_Pusher
from store_sales.components.Model_Trainer_Time_Series import ModelTrainer_time
from store_sales.constant import *



import os, sys
from collections import namedtuple
from datetime import datetime
import pandas as pd



class Pipeline():

    def __init__(self, config: Configuration = Configuration()) -> None:
        try:
            self.config = config
        except Exception as e:
            raise ApplicationException(e, sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise ApplicationException(e, sys) from e

    def start_data_validation(self, data_ingestion_artifact:DataIngestionArtifact)-> DataValidationArtifact:
        try:
            data_validation = DataValidation(data_validation_config=self.config.get_data_validation_config(),
                                             data_ingestion_artifact=data_ingestion_artifact)
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    def start_data_transformation(self,data_ingestion_artifact: DataIngestionArtifact,
                                       data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            data_transformation = DataTransformation(
                data_transformation_config = self.config.get_data_transformation_config(),
                data_ingestion_artifact = data_ingestion_artifact,
                data_validation_artifact = data_validation_artifact)

            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise ApplicationException(e,sys) from e
        
        
        
    def start_time_model_training(self,data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            model_trainer = ModelTrainer_time(model_trainer_config=self.config.get_model_trainer_time_series_config(),
                                        data_transformation_artifact=data_transformation_artifact)   

            return model_trainer.initiate_model_training()
        except Exception as e:
            raise ApplicationException(e,sys) from e 
        
    def start_model_evaluation_pusher(self,model_trainer_artifact:ModelTrainerArtifact,
                                ):
        try:
            model_eval = Model_eval_and_Pusher(model_trainer_artifact)
            model_eval_artifact = model_eval.initiate_model_evaluation_and_pusher()
            return model_eval_artifact
        except  Exception as e:
            raise  ApplicationException(e,sys)
        
         
    
    def run_pipeline(self):
        try:
             #data ingestion

            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact,
                                                    data_validation_artifact=data_validation_artifact)
            
            
            time_model_trainer_artifact = self.start_time_model_training(data_transformation_artifact=data_transformation_artifact)
            model_pusher_artifact = self.start_model_evaluation_pusher(time_model_trainer_artifact)  

        except Exception as e:
            raise ApplicationException(e, sys) from e