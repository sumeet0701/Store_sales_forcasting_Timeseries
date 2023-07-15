from collections import namedtuple

DataIngestionArtifact = namedtuple("DataIngestionArtifact",[
    "Ingestion_file_path",
    "is_ingested",
    "message"])



DataValidationArtifact = namedtuple("DataValidationArtifact",[
    "schema_file_path",
    "message",
    "validated_file_path"])

DataTransformationArtifact = namedtuple("DataTransformationArtifact",[
    "message",
    "time_series_data_file_path",
    "feature_engineering_object_file_path"])

ModelTrainerArtifact = namedtuple("ModelTrainerArtifact",["is_trained",
                                                        "message",
                                                        "trained_model_object_file_path"])

ModelTrainerTIMEArtifact = namedtuple("ModelTrainerArtifact",[
    "message",
    "trained_model_object_file_path",
    "model_report",
    "prediction_image",
    "mse_score",
    "saved_report_file_path",
    "saved_model_file_path",
    "best_model_name",
    "saved_model_plot"])

ModelPusherArtifact=namedtuple("ModelPusherArtifact",["message"])