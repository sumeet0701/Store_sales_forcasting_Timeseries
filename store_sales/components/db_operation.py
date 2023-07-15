from store_sales.exception import ApplicationException
import pandas as pd
import pymongo
import os,sys
from store_sales.constant import *
from store_sales.exception import ApplicationException
from store_sales.logger import logging
from store_sales.constant import *
from store_sales.utils.utils import *
import yaml
import yaml
import pymongo
import certifi
import os
# Path to the CA file
ca_file_path = certifi.where()


class MongoDB:
    def __init__(self):
        try:
            # Get the absolute path to the env.yaml file
            env_file_path = os.path.join(ROOT_DIR, 'env.yaml')
            
            # Load environment variables from env.yaml
            with open(env_file_path) as file:
                env_vars = yaml.safe_load(file)
            username = env_vars.get('USER_NAME')
            password = env_vars.get('PASS_WORD')

            # Use the escaped username and password in the MongoDB connection string
            mongo_db_url = f"mongodb+srv://{username}:{password}@rentalbike.5fi8zs7.mongodb.net/"
            
            self.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca_file_path)

            logging.info("Connection with DB created successfully!!!")     
            
            
            # Read onfig data 
            config = read_yaml_file(file_path=CONFIG_FILE_PATH)
            database_key=config[DATABASE_KEY]
            database_name = database_key[DATABASE_NAME_KEY]
            collection_name = database_key[DATABASE_COLLECTION_NAME_KEY]
            
            # Mongo 
            self.db = self.client[database_name]
            self.collection_name = collection_name
            
            

        except Exception as e:
            raise ApplicationException(e,sys) from e


    def create_and_check_collection(self,coll_name:str = None)->None:
        try:
            
            if coll_name is None:
                # Checking whether the main collection already exist or not, if does then delete it
                if self.collection_name in self.db.list_collection_names():
                    self.db.drop_collection(self.collection_name)

                # Creating new main collection
                self.collection = self.db[self.collection_name]
                
            if coll_name == "Training" or coll_name == "Test":
                # Checking whether the training/test collection already exist or not, if does then delete it
                if coll_name in self.db.list_collection_names():
                    self.db.drop_collection(coll_name)

                self.collection = self.db[coll_name]
                
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def insertall(self,data_dict:dict)-> None:
        try:
            logging.info(f"Inserting data into database:[{DATABASE_NAME_KEY}] in collection: [{self.collection_name}]")
            self.collection.insert_many(data_dict)
            logging.info("Insertion into DB is successful!!! ")
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def fetch_df(self,coll_name:str = None )->pd.DataFrame:
        try:
            if coll_name is None:
                self.collection = self.db[self.collection_name]
                dataframe = pd.DataFrame(self.collection.find())

            if coll_name == "Training" or coll_name == "Test":
                self.collection = self.db[coll_name]
                dataframe = pd.DataFrame(self.collection.find())

            logging.info(f"Data Fetched from collection: [{coll_name}] successfully!!!")

            return dataframe

        except Exception as e:
            raise ApplicationException(e,sys) from e