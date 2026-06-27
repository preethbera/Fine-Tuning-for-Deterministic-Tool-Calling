from unsloth import FastLanguageModel
import re
import json

def generate_response(model, tokenizer, query, tools):
    """
    Runs inference in native 2x faster mode and extracts generated response.
    """
    FastLanguageModel.for_inference(model)
    
    from src.data_processing import SYSTEM_PROMPT
    tools_str = json.dumps(tools, indent=2)
    user_message = f"Available tools:\n{tools_str}\n\nUser Query: {query}"
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")
    
    outputs = model.generate(
        input_ids=inputs, 
        max_new_tokens=512, 
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    input_length = inputs.shape[1]
    response_tokens = outputs[0][input_length:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    return response

def validate_structure(response_text: str):
    """
    Validates structural accuracy using regex to ensure tag ordering and valid JSON.
    """
    print("\n--- RAW RESPONSE ---")
    print(response_text)
    print("--------------------")
    
    thought_match = re.search(r"<thought>\s*(.*?)\s*</thought>", response_text, re.DOTALL)
    if not thought_match:
        print("Validation Failed: Missing <thought> tags.")
        return False
        
    thought_content = thought_match.group(1)
    
    if "1. Tool Assessment:" not in thought_content:
        print("Validation Failed: Missing Tool Assessment in thought block.")
        return False
        
    if "[ABORT:" in thought_content:
        print("Validation Status: Abort sequence detected. Awaiting conversational refusal.")
        if "<tool_call>" in response_text:
            print("Validation Failed: Tool call generated despite ABORT sequence.")
            return False
        print("Validation Passed: Correct refusal generated.")
        return True
        
    tool_match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", response_text, re.DOTALL)
    if not tool_match:
        print("Validation Failed: Missing <tool_call> tags (and no abort sequence detected).")
        return False
        
    tool_content = tool_match.group(1).strip()
    
    # Try parsing multiple tool calls or single JSON array if the dataset represents it that way.
    # Usually it's an array of tool objects.
    try:
        parsed_json = json.loads(tool_content)
        print("Validation Status: Passed. Valid JSON in <tool_call>.")
        return True
    except json.JSONDecodeError as e:
        print(f"Validation Failed: Invalid JSON in <tool_call> ({e}).")
        return False
