import json
import re
from typing import Dict, Any

class Validator:
    @staticmethod
    def parse_generated_text(text: str) -> Dict[str, Any]:
        """
        Parses the generated output text for validation testing using robust patterns.
        """
        thought_match = re.search(r"<thought>\s*(.*?)\s*</thought>", text, re.DOTALL | re.IGNORECASE)
        has_thought = bool(thought_match)
        
        tool_match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL | re.IGNORECASE)
        has_tool_call = bool(tool_match)
        
        is_valid_json = False
        parsed_json = None
        if has_tool_call:
            try:
                parsed_json = json.loads(tool_match.group(1).strip())
                is_valid_json = True
            except json.JSONDecodeError:
                pass
                
        is_abort = "[ABORT:" in text.upper()
        
        return {
            "has_thought": has_thought,
            "has_tool_call": has_tool_call,
            "is_valid_json": is_valid_json,
            "parsed_json": parsed_json,
            "is_abort": is_abort
        }
