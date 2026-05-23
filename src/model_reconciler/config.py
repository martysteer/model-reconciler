"""Global configuration — three environment variables, nothing else."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_base_url: str = "http://host.docker.internal:8080/v1"
    llm_api_key: str | None = None
    profiles_dir: str = "profiles"
    log_level: str = "INFO"

    model_config = {"env_prefix": "", "case_sensitive": False}
