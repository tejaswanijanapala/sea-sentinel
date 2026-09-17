import os
import shutil
from abc import ABC, abstractmethod
from typing import List, Optional

class StorageService(ABC):
    """Abstract Base Class for Storage Operations"""
    
    @abstractmethod
    def save(self, file_path: str, data: bytes) -> str:
        pass
        
    @abstractmethod
    def read(self, file_path: str) -> bytes:
        pass
        
    @abstractmethod
    def get_url(self, file_path: str) -> str:
        pass
        
    @abstractmethod
    def exists(self, file_path: str) -> bool:
        pass
        
    @abstractmethod
    def get_absolute_path(self, file_path: str) -> str:
        pass


class LocalStorage(StorageService):
    """Local File System Storage Provider"""
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        
    def _full_path(self, file_path: str) -> str:
        path = os.path.join(self.base_dir, file_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path
        
    def save(self, file_path: str, data: bytes) -> str:
        full = self._full_path(file_path)
        with open(full, 'wb') as f:
            f.write(data)
        return full
        
    def read(self, file_path: str) -> bytes:
        with open(self._full_path(file_path), 'rb') as f:
            return f.read()
            
    def get_url(self, file_path: str) -> str:
        return f"file://{self._full_path(file_path)}"
        
    def exists(self, file_path: str) -> bool:
        return os.path.exists(self._full_path(file_path))
        
    def get_absolute_path(self, file_path: str) -> str:
        return self._full_path(file_path)


class ObjectStorage(StorageService):
    """Cloud Object Storage Provider (Mock for S3/GCS)"""
    
    def __init__(self, bucket: str):
        self.bucket = bucket
        
    def save(self, file_path: str, data: bytes) -> str:
        # Upload to S3/GCS implementation goes here
        return f"s3://{self.bucket}/{file_path}"
        
    def read(self, file_path: str) -> bytes:
        # Download from S3/GCS goes here
        return b""
        
    def get_url(self, file_path: str) -> str:
        return f"https://storage.googleapis.com/{self.bucket}/{file_path}"
        
    def exists(self, file_path: str) -> bool:
        # Check cloud exists
        return True
        
    def get_absolute_path(self, file_path: str) -> str:
        # For object storage, the absolute path is the remote URL
        return f"s3://{self.bucket}/{file_path}"


# Singleton access based on environment
def get_storage_service() -> StorageService:
    mode = os.environ.get("STORAGE_MODE", "offline").lower()
    if mode == "cloud":
        return ObjectStorage(bucket=os.environ.get("STORAGE_BUCKET", "sea-sentinel-data"))
    else:
        # In offline mode, default to the persistent_data directory mapped to the backend
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        persistent_dir = os.environ.get("PERSISTENT_STORAGE_DIR", os.path.join(project_root, "persistent_data"))
        return LocalStorage(base_dir=persistent_dir)

storage = get_storage_service()
