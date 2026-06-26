import sys
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from src.constant.training_pipeline import TARGET_COLUMN
from src.entity.config_entity import PCAConfig
from src.exception import CustomerException
from src.logger import logging


class CustomerClusteringTransformer:
    """
    Stateful PCA and K-Means transformer to ensure consistent clustering definitions
    across training, validation, and real-time inference pipelines.
    """

    def __init__(self):
        self.pca_config = PCAConfig()
        self.pca = None
        self.kmeans = None

    def fit(self, preprocessed_data: DataFrame):
        """
        Fits PCA and K-Means on the preprocessed training dataset.
        """
        try:
            logging.info("Fitting PCA on the preprocessed training dataset...")
            # Unpack configuration dictionary for PCA
            pca_params = self.pca_config.__dict__
            self.pca = PCA(**pca_params)
            reduced_dataset = self.pca.fit_transform(preprocessed_data)
            logging.info(f"PCA fit completed. Reduced dataset shape: {reduced_dataset.shape}")

            logging.info("Fitting K-Means (n_clusters=3) on the PCA-reduced training dataset...")
            self.kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            self.kmeans.fit(reduced_dataset)
            logging.info("K-Means clustering fit completed successfully.")
            return self
        except Exception as e:
            raise CustomerException(e, sys) from e

    def transform(self, preprocessed_data: DataFrame) -> DataFrame:
        """
        Transforms preprocessed dataset using the fitted PCA and K-Means models.
        Appends the cluster assignment to the TARGET_COLUMN.
        """
        try:
            if self.pca is None or self.kmeans is None:
                raise Exception("CustomerClusteringTransformer is not fitted yet. Call fit() before transform().")

            df_copy = preprocessed_data.copy()
            reduced_dataset = self.pca.transform(df_copy)
            labels = self.kmeans.predict(reduced_dataset)
            df_copy[TARGET_COLUMN] = labels.astype(int)
            logging.info("Assigned cluster labels successfully using stateful PCA & K-Means.")
            return df_copy
        except Exception as e:
            raise CustomerException(e, sys) from e
