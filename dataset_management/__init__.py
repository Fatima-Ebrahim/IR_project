# Dataset Management Module
"""
This module provides functionality for users to upload and manage their own datasets.

Features:
- Upload CSV, JSON, TXT, or TSV files
- Automatic validation and parsing
- Background processing pipeline
- Dataset status tracking
- CRUD operations for datasets
"""

from .dataset_management_service import DatasetManagementService
from .dataset_management_handler import router

__all__ = ['DatasetManagementService', 'router']
