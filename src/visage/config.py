from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VISAGE_", env_file=".env", extra="ignore")

    model_name: str = "buffalo_l"
    detection_size: int = 640
    embedding_dim: int = 512

    match_threshold: float = Field(default=0.35, ge=-1.0, le=1.0)
    detection_confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    index_backend: str = "faiss"
    gallery_path: Path = Path("data/gallery.json")
    index_path: Path = Path("data/index.faiss")

    api_title: str = "Visage"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
