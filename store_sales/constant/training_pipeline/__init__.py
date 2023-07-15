# Training pipeline related variable
TRAINING_PIPELINE_CONFIG_KEY = "training_pipeline_config"
TRAINING_PIPELINE_ARTIFACT_DIR_KEY = "artifact_dir"
TRAINING_PIPELINE_NAME_KEY = "pipeline"

SCHEMA_CONFIG_KEY='schema_config'
SCHEMA_DIR_KEY ='schema_dir'
SCHEMA_FILE_NAME='schema_file'


TARGET_COLUMN_KEY='target_column'
NUMERICAL_COLUMN_KEY='numerical_columns'
NUMERICAL_COLUMN_WITHOUT_TAR='numerical_columns_without_target'
CATEGORICAL_COLUMN_KEY='categorical_columns'
DROP_COLUMN_KEY='drop_columns'
DATE_COLUMN='date_columns'




from store_sales.constant.training_pipeline.data_ingestion import *
from store_sales.constant.training_pipeline.data_validation import *
from store_sales.constant.training_pipeline.data_transformation import *
from store_sales.constant.training_pipeline.Model_trainer import *
from store_sales.constant.training_pipeline.model_trainer_time import *