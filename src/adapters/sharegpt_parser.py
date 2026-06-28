import json
import re
from src.adapters.base_parser import BaseDatasetParser, UnifiedToolCallingRecord
from src.core.exceptions import SchemaValidationError

class SharegptParser(BaseDatasetParser):
    def transform(self, raw_record: dict) -> UnifiedToolCallingRecord:
        try:
            messages = raw_record.get("conversations") or raw_record.get("messages", [])
            
            # Handle stringified conversation lists
            if isinstance(messages, str):
                messages = json.loads(messages.replace("'", '"'))
            
            query = ""
            answers = "[]"
            
            for msg in messages:
                role = msg.get("role", msg.get("from", "")).lower()
                content = msg.get("content", msg.get("value", ""))
                
                if role in ("user", "human"):
                    query = content
                elif role in ("function_call", "tool_call", "tool"):
                    # This is the ground truth tool call — the primary answer source
                    answers = self._normalize_answers(content)
                elif role in ("assistant", "gpt", "model"):
                    # Only use assistant content as fallback if no function_call was found
                    if answers == "[]":
                        answers = self._extract_answers_from_text(content)
                    
            # Parse tools: handle both flat and OpenAI-nested formats
            raw_tools = raw_record.get("tools", "[]")
            if isinstance(raw_tools, str):
                tools = json.loads(raw_tools)
            else:
                tools = raw_tools
                
            # Normalize OpenAI-nested format to flat format
            normalized_tools = []
            if isinstance(tools, list):
                for t in tools:
                    if isinstance(t, dict):
                        if "function" in t and isinstance(t["function"], dict):
                            normalized_tools.append(t["function"])
                        elif "name" in t:
                            normalized_tools.append(t)
                        else:
                            normalized_tools.append(t)
            
            return UnifiedToolCallingRecord(query=query, tools=normalized_tools, answers=answers)
        except Exception as e:
            raise SchemaValidationError(f"SharegptParser failed to transform record: {e}")
    
    def _normalize_answers(self, content: str) -> str:
        """Normalize a raw answer string into a clean JSON array of tool calls."""
        if not content:
            return "[]"
        
        content = content.strip()
        
        # Try direct JSON parse
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return json.dumps(parsed)
            elif isinstance(parsed, dict) and "name" in parsed:
                return json.dumps([parsed])
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Handle Python-style single quotes by replacing them
        try:
            fixed = content.replace("'", '"')
            parsed = json.loads(fixed)
            if isinstance(parsed, list):
                return json.dumps(parsed)
            elif isinstance(parsed, dict) and "name" in parsed:
                return json.dumps([parsed])
        except (json.JSONDecodeError, TypeError):
            pass
        
        return "[]"
    
    def _extract_answers_from_text(self, text: str) -> str:
        """Fallback: extract tool call JSON from free-form assistant text."""
        if not text:
            return "[]"
        
        # Try <tool_call> tags
        tool_match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
        if tool_match:
            raw = tool_match.group(1).strip()
            if raw.startswith("```json"):
                raw = raw[7:]
            elif raw.startswith("```"):
                raw = raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
            return self._normalize_answers(raw)
        
        # Try to find a JSON array with "name" key
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                if isinstance(parsed, list) and len(parsed) > 0:
                    if isinstance(parsed[0], dict) and "name" in parsed[0]:
                        return json.dumps(parsed)
            except json.JSONDecodeError:
                pass
        
        # Detect abort/refusal
        abort_phrases = ["i'm sorry", "i cannot", "i don't have", "no matching tool", "[abort"]
        if any(phrase in text.lower() for phrase in abort_phrases):
            return "[]"
        
        return "[]"
