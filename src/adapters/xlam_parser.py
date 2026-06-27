import json
from src.adapters.base_parser import BaseDatasetParser, UnifiedToolCallingRecord
from src.core.exceptions import SchemaValidationError

class XlamParser(BaseDatasetParser):
    def transform(self, raw_record: dict) -> UnifiedToolCallingRecord:
        try:
            query = raw_record["query"]
            tools = raw_record["tools"]
            if isinstance(tools, str):
                tools = json.loads(tools)
                
            answers = raw_record["answers"]
            if isinstance(answers, list) or isinstance(answers, dict):
                answers = json.dumps(answers)
            elif answers is None:
                answers = "[]"
                
            return UnifiedToolCallingRecord(query=query, tools=tools, answers=answers)
        except Exception as e:
            raise SchemaValidationError(f"XlamParser failed to transform record: {e}")
