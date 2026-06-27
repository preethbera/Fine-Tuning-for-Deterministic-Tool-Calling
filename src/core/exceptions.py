class SchemaValidationError(Exception):
    """Raised when data fails structural or type validation."""
    pass

class DataPipelineException(Exception):
    """Raised when the data ingestion or processing pipeline encounters an unrecoverable error."""
    pass

class DivergenceException(Exception):
    """Raised when training loss diverges (e.g., NaN or Inf)."""
    pass
