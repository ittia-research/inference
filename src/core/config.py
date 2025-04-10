import logging
import os
from pydantic import field_validator, computed_field, ValidationError
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Logging
    LOG_LEVEL_GLOBAL: str = "DEBUG"
    PT_XLA_DEBUG: str = "1"

    # Embedding engine settings
    # Higher limits will result in potential higher RAM consumption
    EMBEDDING_MAX_LENGTH: int = 1024 * 32
    EMBEDDING_BATCH_SIZE: int = 3
    EMBEDDING_MODEL_ID: str = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"

    # Acceleration device
    PJRT_DEVICE: str = "TPU"

    # Save XLA cache
    XLA_CACHE_DIR: str = "/root/.cache/xla"

    # Validator for LOG_LEVEL
    @field_validator('LOG_LEVEL_GLOBAL', mode='before')
    @classmethod
    def validate_and_normalize_log_level(cls, value: str) -> str:
        """
        Validates the input log level string against standard logging levels,
        converts it to upper case, and ensures it's usable.
        """
        level_upper = str(value).upper()
        # Check if corresponds to a valid logging level attribute
        if not hasattr(logging, level_upper) or not isinstance(getattr(logging, level_upper), int):
             valid_levels = [level for level in logging._levelToName.values()] # Get valid names dynamically
             raise ValueError(f"Invalid log level '{value}'. Must be one of: {', '.join(valid_levels)}")
        return level_upper


# --- Instantiate Settings ---
try:
    settings = Settings()
except ValidationError as e:
    logging.critical(f"Configuration Error: Failed to load settings. Details: {e}")
    raise SystemExit(f"Configuration Error: {e}") from e


# --- Configure Root Logger ---
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL_GLOBAL),
    format="%(asctime)s [%(name)s:%(levelname)s] %(message)s", # With logger name
    handlers=[logging.StreamHandler()]
)

# Set XLA debug
os.environ['PT_XLA_DEBUG'] = settings.PT_XLA_DEBUG
