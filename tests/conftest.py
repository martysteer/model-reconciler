"""Shared test fixtures."""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir():
    """Path to test fixture YAMLs."""
    return FIXTURES_DIR


@pytest.fixture
def app_client():
    """FastAPI TestClient with profiles from tests/fixtures/."""
    os.environ["PROFILES_DIR"] = str(FIXTURES_DIR)
    os.environ["LLM_BASE_URL"] = "http://fake:8080/v1"

    from model_reconciler.main import create_app

    app = create_app()
    with TestClient(app) as client:
        yield client
