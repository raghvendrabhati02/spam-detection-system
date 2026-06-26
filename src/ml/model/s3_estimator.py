from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import CustomerException
from src.ml.model.estimator import CustomerSegmentationModel
import sys
from pandas import DataFrame
from src.logger import logging



class CustomerClusterEstimator:
    """
    This class is used to save and retrieve src model in s3 bucket and to do prediction
    """

    def __init__(self,bucket_name,model_path,):
        """
        :param bucket_name: Name of your model bucket
        :param model_path: Location of your model in bucket
        """
        self.bucket_name = bucket_name
        self.s3 = SimpleStorageService()
        self.model_path = model_path
        self.loaded_model:CustomerSegmentationModel=None


    def is_model_present(self,model_path):
        try:
            if not hasattr(self.s3, "is_active") or not self.s3.is_active:
                logging.info("S3 is not active/available. Returning False for model presence on S3.")
                return False
            return self.s3.s3_key_path_available(bucket_name=self.bucket_name, s3_key=model_path)
        except Exception as e:
            logging.warning(f"Error checking model presence on S3: {e}")
            return False

    def load_model(self) -> CustomerSegmentationModel:
        """
        Load the model from the model_path (S3 bucket with local file fallback).
        :return: CustomerSegmentationModel object
        """
        import os
        import glob
        import pickle
        try:
            if not hasattr(self.s3, "is_active") or not self.s3.is_active:
                raise Exception("S3 client is not active/configured.")
            logging.info("Attempting to load model from S3...")
            return self.s3.load_model(self.model_path, bucket_name=self.bucket_name)
        except Exception as e:
            logging.warning(f"Failed to load model from S3 due to: {e}. Checking local fallback files.")
            
            local_paths = [
                os.path.join("artifact", "model_trainer", "trained_model", "model.pkl"),
                os.path.join("src", "model.pkl"),
                "model.pkl"
            ]
            # Recursively search for any model.pkl in the artifact directory
            local_paths.extend(glob.glob("artifact/**/model.pkl", recursive=True))
            
            for path in local_paths:
                if os.path.exists(path):
                    logging.info(f"Found local model at: {path}. Loading...")
                    with open(path, "rb") as f:
                        return pickle.load(f)
            
            raise Exception("Model loading failed: S3 is unavailable and no local model.pkl was found.") from e

    def save_model(self,from_file,remove:bool=False)->None:
        """
        Save the model to the model_path
        :param from_file: Your local system model path
        :param remove: By default it is false that mean you will have your model locally available in your system folder
        :return:
        """
        
        try:
            if not hasattr(self.s3, "is_active") or not self.s3.is_active:
                logging.info("S3 client is not active. Bypassing model upload to S3 and keeping local model file.")
                return
            self.s3.upload_file(from_file,
                                to_filename=self.model_path,
                                bucket_name=self.bucket_name,
                                remove=remove
                                )
        except Exception as e:
            raise CustomerException(e, sys) from e


    def predict(self,dataframe:DataFrame):
        """
        :param dataframe:
        :return:
        """
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe)
        except Exception as e:
            raise CustomerException(e,sys)