import json
import sys
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template

SYSTEM_PROMPT = """You are an expert deterministic tool-calling agent. You are provided with a user query and a list of available tools.
You must follow a strict reasoning process inside <thought> tags before deciding whether to call a tool or not.

Your thought process MUST follow this exact structure:
1. Tool Assessment: List the available tools from the system prompt.
2. Intent Matching: State the user's core request and identify which tool (if any) matches it.
3. Parameter Extraction: Explicitly extract the required parameters from the user's prompt.
4. Validation: Check if all required parameters are present. If a parameter is missing, or if no tool matches, output an abort sequence (e.g., [ABORT: Missing parameter '...'] or [ABORT: No matching tool]) and generate a polite conversational refusal instead of a <tool_call>.

If validation passes, output the tool call strictly within <tool_call> tags as a JSON object.
"""

def generate_prompt_for_row(row, tokenizer):
    """
    Transforms a dataset row into a Tool-CoT formatted prompt, enforcing the curriculum.
    """
    query = row['query']
    
    # Parse JSON strings if they are strings
    tools_json = json.loads(row['tools']) if isinstance(row['tools'], str) else row['tools']
    answers_json = json.loads(row['answers']) if isinstance(row['answers'], str) else row['answers']
    
    tools_str = json.dumps(tools_json, indent=2)
    user_message = f"Available tools:\n{tools_str}\n\nUser Query: {query}"
    
    # Extract tool names for the thought process
    available_tool_names = [t.get("name", "unknown") for t in tools_json]
    tool_list_str = ", ".join(available_tool_names) if available_tool_names else "None"
    
    # Logic for constructing synthetic thought process based on answer existence
    if not answers_json:
        # Abort scenario: No tool matched or missing params (simplifying to no tool match for generation)
        thought = (
            "<thought>\n"
            f"1. Tool Assessment: The available tools are: {tool_list_str}.\n"
            f"2. Intent Matching: The user's query is '{query}'. None of the available tools match this intent.\n"
            "3. Parameter Extraction: N/A\n"
            "4. Validation: [ABORT: No matching tool]\n"
            "</thought>"
        )
        assistant_response = f"{thought}\nI'm sorry, but I don't have the tools to help you with that request."
    else:
        # Tool call scenario
        # We take the first tool call if there are multiple, simplifying to single call for deterministic router
        target_tool = answers_json[0]['name']
        params = answers_json[0]['arguments']
        
        # Find required parameters from tool schema if possible
        required_params = []
        for t in tools_json:
            if t.get("name") == target_tool:
                if "parameters" in t and "required" in t["parameters"]:
                    required_params = t["parameters"]["required"]
                elif "required" in t: # Some schemas might have it at root level
                    required_params = t["required"]
                break
                
        req_params_str = str(required_params) if required_params else "None defined"
        
        thought = (
            "<thought>\n"
            f"1. Tool Assessment: The available tools are: {tool_list_str}.\n"
            f"2. Intent Matching: The user's query is '{query}'. The tool '{target_tool}' matches this intent.\n"
            f"3. Parameter Extraction: The required parameters are {req_params_str}. Values extracted: {params}.\n"
            "4. Validation: All required parameters are present. Proceeding with tool call.\n"
            "</thought>"
        )
        tool_call_str = json.dumps(answers_json, indent=2)
        assistant_response = f"{thought}\n<tool_call>\n{tool_call_str}\n</tool_call>"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_response}
    ]
    
    # We apply the Qwen chat template to format the prompt natively
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

def prepare_dataset(dataset_name: str, tokenizer):
    """
    Streams the dataset via HF datasets and maps it to the unified format.
    """
    try:
        dataset = load_dataset(dataset_name, split="train")
    except Exception as e:
        print(f"\n[ERROR] Failed to load dataset '{dataset_name}'.")
        print("Please check your internet connection. If you reverted to a gated dataset, ensure you have set your Hugging Face token.")
        print(f"Original error: {e}")
        sys.exit(1)
        
    def formatting_prompts_func(examples):
        texts = []
        for query, tools, answers in zip(examples["query"], examples["tools"], examples["answers"]):
            row = {"query": query, "tools": tools, "answers": answers}
            text = generate_prompt_for_row(row, tokenizer)
            texts.append(text)
        return {"text": texts}
    
    # Process in batches
    dataset = dataset.map(formatting_prompts_func, batched=True, num_proc=2)
    return dataset
