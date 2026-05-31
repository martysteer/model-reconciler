"""Global configuration — environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_base_url: str = "http://host.docker.internal:8080/v1"
    llm_api_key: str | None = None
    llm_concurrency: int = 4
    profiles_dir: str = "profiles"
    log_level: str = "INFO"

    model_config = {"env_prefix": "", "case_sensitive": False}
