"""
Configuration settings for the Gemini DataFrame Agent
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Gemini API Configuration
    gemini_api_key: str = Field("", env="GEMINI_API_KEY")
    gemini_model: str = Field("gemini-2.0-flash", env="GEMINI_MODEL")
    
    # Application Settings
    debug: bool = Field(False, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    max_file_size: int = Field(10485760, env="MAX_FILE_SIZE")  # 10MB
    allowed_file_types: List[str] = Field(
        ["csv", "xlsx", "json", "parquet"], 
        env="ALLOWED_FILE_TYPES"
    )
    
    # Security Settings
    secret_key: str = Field("dev-secret-key", env="SECRET_KEY")
    session_timeout: int = Field(3600, env="SESSION_TIMEOUT")
    
    # Performance Settings
    query_cache_size: int = Field(100, env="QUERY_CACHE_SIZE")
    max_iterations: int = Field(5, env="MAX_ITERATIONS")
    temperature: float = Field(0.1, env="TEMPERATURE")
    
    # Optional Database
    database_url: Optional[str] = Field(None, env="DATABASE_URL")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Validation
def validate_settings():
    """Validate critical settings"""
    if not settings.gemini_api_key or settings.gemini_api_key == "your_gemini_api_key_here":
        print("⚠️ Warning: GEMINI_API_KEY not set. Please add your API key to .env file")
        return False
    
    if settings.max_file_size < 1024:  # Minimum 1KB
        raise ValueError("MAX_FILE_SIZE must be at least 1024 bytes")
    
    return True

# Run validation on import
validate_settings()