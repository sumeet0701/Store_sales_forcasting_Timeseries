import sys 
import pandas as pd
import yaml
import numpy as np
import dill  
import os
import sys 
from store_sales.exception import ApplicationException
from store_sales.logger import logging
import pickle
import joblib
import shutil




def read_yaml_file(file_path:str)->dict:
    """
    Reads a YAML file and returns the contents as dictionary.
    Params:
    ---------------
    file_path (str) : file path for the yaml file
    """
    try:
        with open(file_path,"rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise ApplicationException(e,sys) from e
    

def save_array_to_directory(array: np.array, directory_path: str, file_name: str, extension: str = '.npy'):
    try:
        # Create the directory if it doesn't exist
        os.makedirs(directory_path, exist_ok=True)

        # Add the extension to the file name
        file_name_with_extension = file_name + extension

        # Generate the file path
        file_path = os.path.join(directory_path, file_name_with_extension)

        # Save the array to the file path
        np.save(file_path, array)
    except Exception as e:
        ApplicationException(e,sys)

    
def save_object(file_path:str,obj):
    try:
        logging.info(f" file path{file_path}")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise ApplicationException(e,sys) from e
    
    
def load_pickle_object(file_path: str):
    """
    Load a pickled object from a file.
    
    file_path: str
        Path to the file containing the pickled object.
    return: object
        The unpickled object.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise ApplicationException(e, sys) from e
    
    
def save_object(file_path:str,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise ApplicationException(e,sys) from e

def save_pickle_object(file_path, model):
    """
    Save a model object as a pickled file.
    
    file_path: str
        Path to the file where the model will be saved.
    model: object
        The model object to be pickled and saved.
    """
    try:
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
    except Exception as e:
        raise ApplicationException(e, sys) from e
    
    
def load_numpy_array_data(file_path: str, file_name: str) -> np.ndarray:
    """
    Load numpy array data from a file.
    file_path: str, path of the file directory
    file_name: str, name of the file (without the extension)
    return: The loaded numpy array data
    """
    try:
        file_with_path = os.path.join(file_path, file_name + '.npy')
        return np.load(file_with_path, allow_pickle=True)
    except Exception as e:
        raise ApplicationException(e, sys) from e
    
    

def load_object(file_path: str):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise ApplicationException(e, sys) from e
    
    
def save_data(file_path:str, data:pd.DataFrame):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        data.to_csv(file_path,index = None)
    except Exception as e:
        raise ApplicationException(e,sys) from e
    
    

def save_image(image_path, image):
    """
    Save an image to a specified file path.
    
    Args:
        image_path (str): The file path to save the image.
        image (PIL.Image.Image): The image object to save.
    """
    try:
        image.save(image_path)
        print(f"Image saved successfully at {image_path}")
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        
def copy_image(source_path, destination_path):
    # Extract the file name from the source path
    file_name = os.path.basename(source_path)

    # Construct the destination path with the file name
    destination_file_path = os.path.join(destination_path, file_name)

    # Copy the image file to the destination directory
    shutil.copyfile(source_path, destination_file_path)

    return destination_file_path