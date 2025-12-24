"""
Data Source Management Module

Provides abstract base classes and implementations for various data sources
including APIs, databases, and files.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from loguru import logger

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("pandas not available, DataFrame support limited")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not available, API data source limited")

try:
    import sqlalchemy
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    logger.warning("sqlalchemy not available, database data source limited")


class DataSourceType(Enum):
    """Types of data sources."""
    API = "api"
    DATABASE = "database"
    FILE = "file"
    STREAM = "stream"


@dataclass
class DataSchema:
    """Schema information for a data source."""
    
    fields: Dict[str, str] = field(default_factory=dict)
    primary_key: Optional[str] = None
    timestamp_field: Optional[str] = None
    description: str = ""


@dataclass
class DataSourceMetadata:
    """Metadata for a data source."""
    
    name: str
    source_type: DataSourceType
    schema: DataSchema
    update_frequency: float = 3600.0  # seconds
    last_update: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    quality_score: float = 1.0
    description: str = ""


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    def __init__(
        self,
        name: str,
        source_type: DataSourceType,
        schema: Optional[DataSchema] = None,
        update_frequency: float = 3600.0,
        **kwargs
    ) -> None:
        """
        Initialize a data source.
        
        Args:
            name: Unique identifier for the data source
            source_type: Type of data source
            schema: Schema information for the data
            update_frequency: Update frequency in seconds
            **kwargs: Additional configuration
        """
        self.name = name
        self.source_type = source_type
        self.schema = schema or DataSchema()
        self.update_frequency = update_frequency
        self.metadata = DataSourceMetadata(
            name=name,
            source_type=source_type,
            schema=self.schema,
            update_frequency=update_frequency
        )
        self._cache: Optional[Any] = None
        self._cache_timestamp: Optional[datetime] = None
        
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the data source.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the data source."""
        pass
    
    @abstractmethod
    def fetch_data(
        self,
        query: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> Any:
        """
        Fetch data from the source.
        
        Args:
            query: Query parameters for filtering data
            limit: Maximum number of records to fetch
            
        Returns:
            Data in appropriate format (dict, list, DataFrame, etc.)
        """
        pass
    
    def get_schema(self) -> DataSchema:
        """
        Get the schema for this data source.
        
        Returns:
            DataSchema object
        """
        return self.schema
    
    def get_metadata(self) -> DataSourceMetadata:
        """
        Get metadata for this data source.
        
        Returns:
            DataSourceMetadata object
        """
        return self.metadata
    
    def is_cache_valid(self) -> bool:
        """
        Check if cached data is still valid.
        
        Returns:
            True if cache is valid, False otherwise
        """
        if self._cache is None or self._cache_timestamp is None:
            return False
        
        elapsed = (datetime.now() - self._cache_timestamp).total_seconds()
        return elapsed < self.update_frequency
    
    def get_cached_data(self) -> Optional[Any]:
        """
        Get cached data if available and valid.
        
        Returns:
            Cached data or None if not available/invalid
        """
        if self.is_cache_valid():
            return self._cache
        return None
    
    def update_cache(self, data: Any) -> None:
        """
        Update the cache with new data.
        
        Args:
            data: Data to cache
        """
        self._cache = data
        self._cache_timestamp = datetime.now()
        self.metadata.last_update = self._cache_timestamp


class APIDataSource(DataSource):
    """Data source for REST APIs."""
    
    def __init__(
        self,
        name: str,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        auth: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        schema: Optional[DataSchema] = None,
        update_frequency: float = 3600.0,
        **kwargs
    ) -> None:
        """
        Initialize an API data source.
        
        Args:
            name: Unique identifier for the data source
            url: API endpoint URL
            method: HTTP method (GET, POST, etc.)
            headers: HTTP headers
            auth: Authentication credentials
            params: Default query parameters
            schema: Schema information
            update_frequency: Update frequency in seconds
            **kwargs: Additional configuration
        """
        super().__init__(
            name=name,
            source_type=DataSourceType.API,
            schema=schema,
            update_frequency=update_frequency,
            **kwargs
        )
        self.url = url
        self.method = method.upper()
        self.headers = headers or {}
        self.auth = auth
        self.params = params or {}
        self._session: Optional[Any] = None
        
        if not REQUESTS_AVAILABLE:
            logger.warning(f"requests library not available, API source '{name}' may not work")
    
    def connect(self) -> bool:
        """
        Establish connection to the API.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if REQUESTS_AVAILABLE:
                import requests
                self._session = requests.Session()
                if self.auth:
                    self._session.auth = tuple(self.auth.values()) if isinstance(self.auth, dict) else self.auth
                self._session.headers.update(self.headers)
                
                # Test connection
                test_response = self._session.request(
                    method=self.method,
                    url=self.url,
                    params=self.params,
                    timeout=10
                )
                test_response.raise_for_status()
                logger.info(f"Successfully connected to API source '{self.name}'")
                return True
            else:
                logger.error(f"requests library not available for API source '{self.name}'")
                return False
        except Exception as e:
            logger.error(f"Failed to connect to API source '{self.name}': {e}")
            return False
    
    def disconnect(self) -> None:
        """Close connection to the API."""
        if self._session:
            self._session.close()
            self._session = None
            logger.info(f"Disconnected from API source '{self.name}'")
    
    def fetch_data(
        self,
        query: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> Any:
        """
        Fetch data from the API.
        
        Args:
            query: Query parameters to merge with default params
            limit: Maximum number of records (if API supports pagination)
            
        Returns:
            Response data (dict, list, or DataFrame)
        """
        if not self._session:
            if not self.connect():
                raise ConnectionError(f"Not connected to API source '{self.name}'")
        
        try:
            params = {**self.params}
            if query:
                params.update(query)
            if limit:
                params['limit'] = limit
            
            response = self._session.request(
                method=self.method,
                url=self.url,
                params=params if self.method == "GET" else None,
                json=params if self.method in ["POST", "PUT", "PATCH"] else None,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            self.update_cache(data)
            return data
        except Exception as e:
            logger.error(f"Error fetching data from API source '{self.name}': {e}")
            raise


class DatabaseDataSource(DataSource):
    """Data source for databases."""
    
    def __init__(
        self,
        name: str,
        connection_string: str,
        table_name: Optional[str] = None,
        query: Optional[str] = None,
        schema: Optional[DataSchema] = None,
        update_frequency: float = 3600.0,
        **kwargs
    ) -> None:
        """
        Initialize a database data source.
        
        Args:
            name: Unique identifier for the data source
            connection_string: Database connection string
            table_name: Name of the table to query
            query: Custom SQL query
            schema: Schema information
            update_frequency: Update frequency in seconds
            **kwargs: Additional configuration
        """
        super().__init__(
            name=name,
            source_type=DataSourceType.DATABASE,
            schema=schema,
            update_frequency=update_frequency,
            **kwargs
        )
        self.connection_string = connection_string
        self.table_name = table_name
        self.query = query
        self._engine: Optional[Any] = None
        self._connection: Optional[Any] = None
        
        if not SQLALCHEMY_AVAILABLE:
            logger.warning(f"sqlalchemy not available, database source '{name}' may not work")
    
    def connect(self) -> bool:
        """
        Establish connection to the database.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if SQLALCHEMY_AVAILABLE:
                from sqlalchemy import create_engine
                self._engine = create_engine(self.connection_string)
                self._connection = self._engine.connect()
                logger.info(f"Successfully connected to database source '{self.name}'")
                return True
            else:
                logger.error(f"sqlalchemy not available for database source '{self.name}'")
                return False
        except Exception as e:
            logger.error(f"Failed to connect to database source '{self.name}': {e}")
            return False
    
    def disconnect(self) -> None:
        """Close connection to the database."""
        if self._connection:
            self._connection.close()
            self._connection = None
        if self._engine:
            self._engine.dispose()
            self._engine = None
            logger.info(f"Disconnected from database source '{self.name}'")
    
    def fetch_data(
        self,
        query: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> Any:
        """
        Fetch data from the database.
        
        Args:
            query: Additional query parameters (filters, etc.)
            limit: Maximum number of records to fetch
            
        Returns:
            Query results (list of dicts or DataFrame)
        """
        if not self._connection:
            if not self.connect():
                raise ConnectionError(f"Not connected to database source '{self.name}'")
        
        try:
            if self.query:
                sql_query = self.query
            elif self.table_name:
                sql_query = f"SELECT * FROM {self.table_name}"
            else:
                raise ValueError("Either table_name or query must be provided")
            
            if limit:
                sql_query += f" LIMIT {limit}"
            
            if PANDAS_AVAILABLE and SQLALCHEMY_AVAILABLE:
                import pandas as pd
                df = pd.read_sql(sql_query, self._connection)
                self.update_cache(df)
                return df
            else:
                result = self._connection.execute(sql_query)
                data = [dict(row) for row in result]
                self.update_cache(data)
                return data
        except Exception as e:
            logger.error(f"Error fetching data from database source '{self.name}': {e}")
            raise


class FileDataSource(DataSource):
    """Data source for files."""
    
    def __init__(
        self,
        name: str,
        file_path: str,
        file_type: Optional[str] = None,
        schema: Optional[DataSchema] = None,
        update_frequency: float = 3600.0,
        **kwargs
    ) -> None:
        """
        Initialize a file data source.
        
        Args:
            name: Unique identifier for the data source
            file_path: Path to the file
            file_type: File type (csv, json, parquet, etc.) - auto-detected if None
            schema: Schema information
            update_frequency: Update frequency in seconds
            **kwargs: Additional configuration
        """
        super().__init__(
            name=name,
            source_type=DataSourceType.FILE,
            schema=schema,
            update_frequency=update_frequency,
            **kwargs
        )
        self.file_path = file_path
        self.file_type = file_type or self._detect_file_type(file_path)
        self._file_handle: Optional[Any] = None
    
    def _detect_file_type(self, file_path: str) -> str:
        """
        Detect file type from extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected file type
        """
        import os
        ext = os.path.splitext(file_path)[1].lower()
        type_map = {
            '.csv': 'csv',
            '.json': 'json',
            '.parquet': 'parquet',
            '.xlsx': 'excel',
            '.xls': 'excel',
            '.pkl': 'pickle',
            '.pickle': 'pickle'
        }
        return type_map.get(ext, 'unknown')
    
    def connect(self) -> bool:
        """
        Verify file exists and is accessible.
        
        Returns:
            True if file is accessible, False otherwise
        """
        try:
            import os
            if os.path.exists(self.file_path) and os.path.isfile(self.file_path):
                logger.info(f"Successfully connected to file source '{self.name}'")
                return True
            else:
                logger.error(f"File not found: {self.file_path}")
                return False
        except Exception as e:
            logger.error(f"Failed to connect to file source '{self.name}': {e}")
            return False
    
    def disconnect(self) -> None:
        """Close file handle if open."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
    
    def fetch_data(
        self,
        query: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> Any:
        """
        Fetch data from the file.
        
        Args:
            query: Query parameters (filters, etc.) - not all file types support this
            limit: Maximum number of records to fetch
            
        Returns:
            File data (dict, list, or DataFrame)
        """
        if not self.connect():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        try:
            if self.file_type == 'csv':
                if PANDAS_AVAILABLE:
                    import pandas as pd
                    df = pd.read_csv(self.file_path, nrows=limit)
                    self.update_cache(df)
                    return df
                else:
                    import csv
                    with open(self.file_path, 'r') as f:
                        reader = csv.DictReader(f)
                        data = [row for i, row in enumerate(reader) if limit is None or i < limit]
                    self.update_cache(data)
                    return data
            
            elif self.file_type == 'json':
                import json
                with open(self.file_path, 'r') as f:
                    data = json.load(f)
                if limit and isinstance(data, list):
                    data = data[:limit]
                self.update_cache(data)
                return data
            
            elif self.file_type == 'parquet':
                if PANDAS_AVAILABLE:
                    import pandas as pd
                    df = pd.read_parquet(self.file_path)
                    if limit:
                        df = df.head(limit)
                    self.update_cache(df)
                    return df
                else:
                    raise ImportError("pandas required for parquet files")
            
            elif self.file_type in ['excel', 'xlsx', 'xls']:
                if PANDAS_AVAILABLE:
                    import pandas as pd
                    df = pd.read_excel(self.file_path, nrows=limit)
                    self.update_cache(df)
                    return df
                else:
                    raise ImportError("pandas required for Excel files")
            
            else:
                raise ValueError(f"Unsupported file type: {self.file_type}")
        except Exception as e:
            logger.error(f"Error fetching data from file source '{self.name}': {e}")
            raise

