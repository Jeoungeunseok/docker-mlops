from dataclasses import dataclass


@dataclass(frozen=True)
class DatabaseConfig:
    url: str
