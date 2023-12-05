from stockify.entity.config_entity import DataIngestionConfig
import sys,os
from stockify.exception import StockifyExpection
from stockify.logger import logging
from stockify.entity.artifact_entity import DataIngestionArtifact
import tarfile
import numpy as np
from six.moves import urllib
import pandas as pd
import numpy as np
from zipfile import ZipFile
import pickle

import warnings
warnings.filterwarnings('ignore')


class DataIngestion:

    def __init__(self,data_ingestion_config:DataIngestionConfig ):
        try:
            logging.info(f"{'>>'*20}Data Ingestion log started.{'<<'*20} ")
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise StockifyExpection(e,sys)
    

    def download_housing_data(self,) -> str:
        try:
            #extraction remote url to download dataset
            download_url = self.data_ingestion_config.dataset_download_url

            #folder location to download file
            tgz_download_dir = self.data_ingestion_config.tgz_download_dir
            
            os.makedirs(tgz_download_dir,exist_ok=True)

            Dataset_Name = os.path.basename(download_url)

            tgz_file_path = os.path.join(tgz_download_dir, Dataset_Name)

            logging.info(f"Downloading file from :[{download_url}] into :[{tgz_file_path}]")
            urllib.request.urlretrieve(download_url, tgz_file_path)
            logging.info(f"File :[{tgz_file_path}] has been downloaded successfully.")
            return tgz_file_path

        except Exception as e:
            raise StockifyExpection(e,sys) from e

    def extract_tgz_file(self,tgz_file_path:str):
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            
            if os.path.exists(raw_data_dir):
                os.remove(raw_data_dir)

            os.makedirs(raw_data_dir,exist_ok=True)

            logging.info(f"Extracting tgz file: [{tgz_file_path}] into dir: [{raw_data_dir}]")
            
            
            with ZipFile(tgz_file_path,'r') as housing_tgz_file_obj:
                housing_tgz_file_obj.extractall(path=raw_data_dir)
            logging.info(f"Extraction completed")

        except Exception as e:
            raise StockifyExpection(e,sys) from e
    
    def split_data_as_train_test(self) -> DataIngestionArtifact:
        try:


            list_of_data = []
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            csv_files = [os.path.join(raw_data_dir, file) for file in os.listdir(raw_data_dir) if file.endswith('data.csv')]
            news_data = [os.path.join(raw_data_dir, file) for file in os.listdir(raw_data_dir) if file.endswith('news.csv')]
            
            
            for i in range(0,len(csv_files)):

                file_name = os.path.basename(csv_files[i])
            
                housing_file_path = os.path.join(raw_data_dir,file_name)
                list_of_data.append(file_name)


                logging.info(f"Reading csv file: [{housing_file_path}]")
                

                logging.info(f"Splitting data into train and test")
    
                train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir,file_name)
                test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir,file_name)

                housing_data_frame = pd.read_csv(housing_file_path)
                
                close = housing_data_frame["Close"].values
                # pickle.dump(close, open(f'{file_name.split(".")[0]}_close.pkl', 'wb'))

                df = housing_data_frame[['Open', 'High', 'Low']].values

                traning = None
                testing = None

                traning = int(len(df)*0.70)
                testing = len(df) - traning
                traning,testing = df[0:traning],df[testing:len(df)]
                traning = pd.DataFrame(traning)
                testing = pd.DataFrame(testing)


                
                if traning is not None:
                    os.makedirs(self.data_ingestion_config.ingested_train_dir,exist_ok=True)
                    logging.info(f"Exporting training datset to file: [{train_file_path,}]")
                    traning.to_csv(train_file_path,index=False)



                if testing is not None:
                    os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok= True)
                    logging.info(f"Exporting test dataset to file: [{test_file_path}]")
                    testing.to_csv(test_file_path,index=False)
                train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir)
                test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir)


            data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
                                list_of_data=list_of_data,
                                test_file_path=test_file_path,
                                is_ingested=True,
                                message=f"Data ingestion completed successfully.",
                                news_data = news_data,
                                csv_files = csv_files
                                
                                )
            logging.info(f"Data Ingestion artifact:[{data_ingestion_artifact}]")
            return data_ingestion_artifact

        except Exception as e:
            raise StockifyExpection(e,sys) from e

    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        try:
            tgz_file_path =  self.download_housing_data()
            self.extract_tgz_file(tgz_file_path=tgz_file_path)
            return self.split_data_as_train_test()
        except Exception as e:
            raise StockifyExpection(e,sys) from e
    

    def __del__(self):
        logging.info(f"{'>>'*20}Data Ingestion log completed.{'<<'*20} \n\n")
