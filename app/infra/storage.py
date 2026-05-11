from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectStorageConfig:
    endpoint_url: str
    bucket: str
