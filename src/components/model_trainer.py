import sys
from typing import List, Tuple
import os
from pandas import DataFrame
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from src.ml.model.estimator import CustomerSegmentationModel
from src.exception import CustomerException
from src.logger import logging
from src.utils.main_utils import MainUtils, load_numpy_array_data


class ModelTrainer:
    def __init__(self, 
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        self.utils = MainUtils()

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            
            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]
            
            best_model = None
            best_score = -1.0
            
            # Attempt to use neuro_mf ModelFactory, fallback to standard scikit-learn GridSearch if not available
            try:
                logging.info("Attempting to import and train using neuro_mf...")
                from neuro_mf import ModelFactory
                
                model_factory = ModelFactory(model_config_path=self.model_trainer_config.model_config_file_path)
                best_model_detail = model_factory.get_best_model(
                    X=x_train, 
                    y=y_train, 
                    base_accuracy=self.model_trainer_config.expected_accuracy
                )
                best_model = best_model_detail.best_model
                best_score = best_model_detail.best_score
                logging.info(f"neuro_mf ModelFactory completed. Selected model score: {best_score}")
            except Exception as factory_err:
                logging.warning(f"Could not use neuro_mf model search: {factory_err}. Falling back to standard scikit-learn search.")
                
                from sklearn.model_selection import GridSearchCV
                from sklearn.linear_model import LogisticRegression
                from sklearn.ensemble import RandomForestClassifier
                
                models_to_search = {
                    "LogisticRegression": (
                        LogisticRegression(max_iter=1000, random_state=42),
                        {"C": [0.01, 0.1, 1.0, 10.0]}
                    ),
                    "RandomForestClassifier": (
                        RandomForestClassifier(random_state=42),
                        {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}
                    )
                }
                
                for name, (model_obj, param_grid) in models_to_search.items():
                    logging.info(f"Running GridSearchCV for model: {name}")
                    grid_search = GridSearchCV(
                        estimator=model_obj,
                        param_grid=param_grid,
                        cv=3,
                        scoring="f1_weighted",
                        n_jobs=-1
                    )
                    grid_search.fit(x_train, y_train)
                    score = grid_search.best_score_
                    logging.info(f"F1 (weighted) score on train cross-val for {name}: {score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        best_model = grid_search.best_estimator_
                
                logging.info(f"Best model selected: {best_model.__class__.__name__} with cross-val F1: {best_score:.4f}")
            
            preprocessing_obj = self.utils.load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            if best_score < self.model_trainer_config.expected_accuracy:
                logging.warning(f"Best model score {best_score:.4f} is lower than expected accuracy {self.model_trainer_config.expected_accuracy:.4f}")
                raise Exception(f"No best model found with score more than base score: {self.model_trainer_config.expected_accuracy}")
             
            # Instantiate prediction estimator
            customer_segmentation_model = CustomerSegmentationModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=best_model
            )
            
            # Compute actual metrics on the test split instead of hardcoded values
            y_pred = best_model.predict(x_test)
            test_f1 = float(f1_score(y_test, y_pred, average='weighted'))
            test_precision = float(precision_score(y_test, y_pred, average='weighted'))
            test_recall = float(recall_score(y_test, y_pred, average='weighted'))
            
            logging.info(f"Test Set Metrics - F1: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
            
            # Save the trained estimator
            trained_model_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(trained_model_path, exist_ok=True)
            
            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=customer_segmentation_model
            )
            logging.info(f"Customer Segmentation Model saved successfully at: {self.model_trainer_config.trained_model_file_path}")
            
            metric_artifact = ClassificationMetricArtifact(
                f1_score=test_f1,
                precision_score=test_precision,
                recall_score=test_recall
            )
            
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )

            logging.info("Model training component completed successfully.")
            return model_trainer_artifact

        except Exception as e:
            raise CustomerException(e, sys) from e
