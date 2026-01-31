"""Configuration management for DAVM."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """DAVM configuration settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys
    anthropic_api_key: str = Field(default="", description="Anthropic API key")

    # Default pilot and models
    davm_default_pilot: Literal["anthropic", "ollama"] = Field(
        default="anthropic", description="Default pilot to use"
    )
    davm_anthropic_model: str = Field(
        default="claude-sonnet-4-20250514", description="Default Anthropic model"
    )
    davm_ollama_model: str = Field(
        default="deepseek-r1", description="Default Ollama model"
    )

    # Ollama configuration
    ollama_host: str = Field(
        default="http://localhost:11434", description="Ollama server URL"
    )

    # File system access
    davm_allowed_paths: str = Field(
        default="~", description="Comma-separated list of allowed paths"
    )

    # Autonomy level
    davm_autonomy_level: Literal["ask", "semi", "full"] = Field(
        default="semi", description="Autonomy level for the mech"
    )

    # Data storage
    davm_data_dir: Path = Field(default=Path("./data"), description="Data directory")
    davm_memory_dir: Path = Field(
        default=Path("./data/memory"), description="Memory storage directory"
    )
    davm_conversations_dir: Path = Field(
        default=Path("./data/conversations"), description="Conversations directory"
    )

    @property
    def allowed_paths_list(self) -> list[Path]:
        """Get list of allowed paths as Path objects."""
        paths = []
        for p in self.davm_allowed_paths.split(","):
            p = p.strip()
            if p:
                path = Path(p).expanduser().resolve()
                paths.append(path)
        return paths

    def is_path_allowed(self, path: str | Path) -> bool:
        """Check if a path is within allowed directories."""
        target = Path(path).expanduser().resolve()
        for allowed in self.allowed_paths_list:
            try:
                target.relative_to(allowed)
                return True
            except ValueError:
                continue
        return False


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
