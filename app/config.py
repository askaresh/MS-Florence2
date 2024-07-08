from pydantic_settings import BaseSettings
from logging_config import get_logger

logger = get_logger(__name__)

class ModelConfig(BaseSettings):
    MODEL_ID: str = "microsoft/Florence-2-large"
    RATE_LIMIT: int = 5  # Add this line if you want to use a rate limit

    class Config:
        env_file = ".env"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info(f"Loaded ModelConfig: MODEL_ID={self.MODEL_ID}, RATE_LIMIT={self.RATE_LIMIT}")