from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union
from abc import ABC, abstractmethod

class UnifiedToolCallingRecord(BaseModel):
    query: str = Field(description="The user's query.")
    tools: List[Dict[str, Any]] = Field(description="List of available tool schemas.")
    answers: str = Field(description="Expected tool call answer, often a JSON string.")

class BaseDatasetParser(ABC):
    @abstractmethod
    def transform(self, raw_record: dict) -> UnifiedToolCallingRecord:
        """Transforms a raw dataset row into a UnifiedToolCallingRecord."""
        pass
