import json
import re
from src.adapters.base_parser import BaseDatasetParser, UnifiedToolCallingRecord
from src.core.exceptions import SchemaValidationError

class SharegptParser(BaseDatasetParser):
    def transform(self, raw_record: dict) -> UnifiedToolCallingRecord:
        try:
            messages = raw_record.get("conversations") or raw_record.get("messages", [])
            query = ""
            assistant_content = ""
            
            for msg in messages:
                role = msg.get("role", msg.get("from", "")).lower()
                content = msg.get("content", msg.get("value", ""))
                
                if role in ("user", "human"):
                    query = content
                elif role in ("assistant", "gpt", "model"):
                    assistant_content = content
                    
            # Parse tools: handle both flat and OpenAI-nested formats
            raw_tools = raw_record.get("tools", "[]")
            if isinstance(raw_tools, str):
                tools = json.loads(raw_tools)
            else:
                tools = raw_tools
                
            # Normalize OpenAI-nested format to flat format
            # {"type": "function", "function": {"name": "...", "parameters": {...}}}
            # becomes: {"name": "...", "parameters": {...}}
            normalized_tools = []
            if isinstance(tools, list):
                for t in tools:
                    if isinstance(t, dict):
                        if "function" in t and isinstance(t["function"], dict):
                            # OpenAI nested format
                            func = t["function"]
                            normalized_tools.append(func)
                        elif "name" in t:
                            # Already flat format
                            normalized_tools.append(t)
                        else:
                            normalized_tools.append(t)
            
            # Extract answers from assistant content
            # The assistant content in ShareGPT format may contain the tool call
            # as raw text. We need to extract the JSON tool call from it.
            answers = self._extract_answers(assistant_content)
                    
            return UnifiedToolCallingRecord(query=query, tools=normalized_tools, answers=answers)
        except Exception as e:
            raise SchemaValidationError(f"SharegptParser failed to transform record: {e}")
    
    def _extract_answers(self, assistant_content: str) -> str:
        """Extract tool call answers from assistant response text."""
        if not assistant_content:
            return "[]"
        
        # Try to parse the entire content as JSON directly (some datasets store raw JSON)
        try:
            parsed = json.loads(assistant_content)
            if isinstance(parsed, list):
                return assistant_content
            elif isinstance(parsed, dict) and "name" in parsed:
                return json.dumps([parsed])
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Try to extract JSON from <tool_call> tags
        tool_match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", assistant_content, re.DOTALL)
        if tool_match:
            raw = tool_match.group(1).strip()
            # Strip markdown wrappers
            if raw.startswith("```json"):
                raw = raw[7:]
            elif raw.startswith("```"):
                raw = raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
            
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return json.dumps(parsed)
                elif isinstance(parsed, dict):
                    return json.dumps([parsed])
            except json.JSONDecodeError:
                pass
        
        # Try to extract any JSON array from the text
        json_match = re.search(r'\[.*\]', assistant_content, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                if isinstance(parsed, list) and len(parsed) > 0:
                    if isinstance(parsed[0], dict) and "name" in parsed[0]:
                        return json.dumps(parsed)
            except json.JSONDecodeError:
                pass
        
        # Try to extract a single JSON object with "name" key
        json_obj_match = re.search(r'\{[^{}]*"name"[^{}]*\}', assistant_content, re.DOTALL)
        if json_obj_match:
            try:
                parsed = json.loads(json_obj_match.group(0))
                if isinstance(parsed, dict) and "name" in parsed:
                    return json.dumps([parsed])
            except json.JSONDecodeError:
                pass
        
        # If assistant says something like "I'm sorry" or abort, it's a negative case
        abort_phrases = ["i'm sorry", "i cannot", "i don't have", "no matching tool", "[abort"]
        if any(phrase in assistant_content.lower() for phrase in abort_phrases):
            return "[]"
        
        return "[]"
