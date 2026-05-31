"""Tests for LLM client utilities."""

from model_reconciler.llm import ProviderConfig, detect_provider


def test_detect_local_localhost():
    p = detect_provider("http://localhost:8080/v1")
    assert p.supports_json_schema is True
    assert p.supports_seed is True
    assert p.supports_response_format is True


def test_detect_local_docker_internal():
    p = detect_provider("http://host.docker.internal:8080/v1")
    assert p.supports_json_schema is True
    assert p.supports_seed is True
    assert p.supports_response_format is True


def test_detect_local_127():
    p = detect_provider("http://127.0.0.1:8080/v1")
    assert p.supports_json_schema is True


def test_detect_google_ai():
    p = detect_provider("https://generativelanguage.googleapis.com/v1beta/openai")
    assert p.supports_json_schema is False
    assert p.supports_seed is False
    assert p.supports_response_format is True


def test_detect_huggingface():
    p = detect_provider("https://api-inference.huggingface.co/models/meta-llama/Llama-3")
    assert p.supports_response_format is False


def test_detect_generic_hosted():
    p = detect_provider("https://api.openrouter.ai/v1")
    assert p.supports_json_schema is True
    assert p.supports_seed is True
    assert p.supports_response_format is True
