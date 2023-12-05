from stockify.exception import StockifyExpection
from stockify.logger import logging
from stockify.config.configuration import Configuartion
import os, sys
from stockify.entity.artifact_entity import DataIngestionArtifact , DataValidationArtifact ,\
DataTransformationArtifact ,ModelEvaluationArtifact,ModelPusherArtifact , ModelTrainerArtifact
from stockify.entity.config_entity import  DataIngestionConfig
from stockify.components.data_ingestion import DataIngestion 
from stockify.components.data_validation  import DataValidation
from stockify.components.data_transformation  import DataTransformation
from stockify.components.model_trainer import ModelTrainer
class Pipepline:
    def __init__(self,config:Configuartion = Configuartion()):
        try:
            self.config=config
        except Exception as e:
            raise StockifyExpection(e,sys) from e
        
    def start_data_ingestion(self) -> DataIngestionArtifact:
            try:
                data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
                return data_ingestion.initiate_data_ingestion()
            except Exception as e:
                raise StockifyExpection(e, sys) from e
  

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            data_validation = DataValidation(data_validation_config=self.config.get_data_validation_config(),
                                             data_ingestion_artifact=data_ingestion_artifact
                                             )
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise StockifyExpection(e, sys) from e
        
    def start_data_transformation(self,
                                  data_ingestion_artifact: DataIngestionArtifact,
                                  data_validation_artifact: DataValidationArtifact
                                  ) -> DataTransformationArtifact:
        try:
            data_transformation = DataTransformation(
                data_transformation_config=self.config.get_data_transformation_config(),
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact
            )   
            return data_transformation.get_data_transformer()
        except Exception as e:
            raise StockifyExpection(e, sys) 
               
    def start_model_trainer(self,
                            data_transformation_artifact : DataTransformationArtifact,
                            data_ingestion_artifact: DataIngestionArtifact) -> ModelTrainerArtifact:
        try:
            model_trainer = ModelTrainer( data_transformation_artifact=data_transformation_artifact,        
                                         data_ingestion_artifact=data_ingestion_artifact,)
            return model_trainer.stock_data(), model_trainer.LSTM_model(), model_trainer.FinBert(),model_trainer.ichimoku_recommendation()
        except Exception as e:
            raise StockifyExpection(e, sys) from e


    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact,
                                                              data_ingestion_artifact=data_ingestion_artifact,)
            
            
            # model_evaluation_artifact = self.start_model_evaluation(data_ingestion_artifact=data_ingestion_artifact,
            #                                                         data_validation_artifact=data_validation_artifact,
            #                                                         model_trainer_artifact=model_trainer_artifact)

            


        except Exception as e:
            raise StockifyExpection(e, sys) from e