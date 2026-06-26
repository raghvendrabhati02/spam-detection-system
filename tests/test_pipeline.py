import os
import pytest
import pandas as pd
from fastapi.testclient import TestClient

from app import app, get_cluster_persona
from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig

client = TestClient(app)


def test_health_check_endpoint():
    """
    Tests that the FastAPI service health check is responsive
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["service"] == "precision-customer-segmentation"


def test_demo_profiles_endpoint():
    """
    Tests that the demo API yields valid dictionary data to prefill the frontend form
    """
    response = client.get("/api/demo")
    assert response.status_code == 200
    profiles = response.json()
    assert "value_shopper" in profiles
    assert "premium_aficionado" in profiles
    assert "frugal_starter" in profiles
    assert profiles["value_shopper"]["Age"] == 42


def test_cluster_persona_helper():
    """
    Verifies that the helper correctly resolves customer segments to cohort personas
    """
    p0 = get_cluster_persona(0)
    p1 = get_cluster_persona(1)
    p2 = get_cluster_persona(2)
    
    assert p0["name"] == "Value-Seeking Family"
    assert p1["name"] == "Affluent Shopper"
    assert p2["name"] == "Young Frugal Starter"
    assert p0["spending_tier"] == "Medium-Low"


def test_data_ingestion_fallback_execution():
    """
    Validates that the data ingestion module succeeds by falling back to the local
    marketing_campaign.csv file, copying split outputs to the artifact store.
    """
    config = DataIngestionConfig()
    ingestion = DataIngestion(data_ingestion_config=config)
    artifact = ingestion.initiate_data_ingestion()
    
    assert os.path.exists(artifact.trained_file_path)
    assert os.path.exists(artifact.test_file_path)
    
    df_train = pd.read_csv(artifact.trained_file_path)
    assert len(df_train) > 0
    # Ensure dropped columns specified in schema are removed
    assert "ID" not in df_train.columns
    assert "Z_CostContact" not in df_train.columns
