from collections import namedtuple


TrainingPipelineConfig = namedtuple("TrainingPipelineConfig", ["artifact_dir"])

DataIngestionConfig = namedtuple("DataIngestionConfig",[
    "dataset_download_url",
    "tgz_download_dir",
    "raw_data_dir",
    "ingested_data_dir"])


DataValidationConfig = namedtuple("DataValidationConfig", [
    "schema_file_path",
    "file_path"])



DataTransformationConfig = namedtuple("DataTransformationConfig",[
    "time_series_data_file_path",
    "feature_engineering_object_file_path"])




ModelTrainerTIMEConfig = namedtuple("ModelTrainerConfig",[
    "trained_model_file_path",
    "time_Series_grouped_data",
    "model_report",
    "prediction_image",
    "best_model_png",
    "saved_model_file_path",
    "saved_report_file_path",
    "saved_model_plot"])
