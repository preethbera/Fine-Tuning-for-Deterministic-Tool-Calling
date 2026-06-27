import json
from src.adapters.base_parser import BaseDatasetParser, UnifiedToolCallingRecord
from src.core.exceptions import SchemaValidationError

class SharegptParser(BaseDatasetParser):
    def transform(self, raw_record: dict) -> UnifiedToolCallingRecord:
        try:
            messages = raw_record.get("conversations") or raw_record.get("messages", [])
            query = ""
            answers = ""
            
            for msg in messages:
                role = msg.get("role", msg.get("from", "")).lower()
                content = msg.get("content", msg.get("value", ""))
                
                if role in ("user", "human"):
                    query = content
                elif role in ("assistant", "gpt", "model"):
                    answers = content
                    
            raw_tools = raw_record.get("tools", "[]")
            if isinstance(raw_tools, str):
                tools = json.loads(raw_tools)
            else:
                tools = raw_tools
                
            return UnifiedToolCallingRecord(query=query, tools=tools, answers=answers)
        except Exception as e:
            raise SchemaValidationError(f"SharegptParser failed to transform record: {e}")
