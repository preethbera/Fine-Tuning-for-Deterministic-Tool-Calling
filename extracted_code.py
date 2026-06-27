%%capture
import os, re
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth  # Do this in local & cloud setups
else:
    import torch; v = re.match(r'[\d]{1,}\.[\d]{1,}', str(torch.__version__)).group(0)
    xformers = 'xformers==' + {'2.10':'0.0.34','2.9':'0.0.33.post1','2.8':'0.0.32.post2'}.get(v, "0.0.34")
    !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth
    !pip install --no-deps --upgrade "torchao>=0.16.0"
!pip install transformers==4.56.2
!pip install --no-deps trl==0.22.2
!pip install protobuf==3.20.3 # required
!pip install --no-deps transformers-cfg

from unsloth import FastQwen2Model
import torch

max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Llama-3.1-8B-bnb-4bit",  # Llama-3.1 2x faster
    "unsloth/Llama-3.1-70B-bnb-4bit",
    "unsloth/Mistral-Small-Instruct-2409",  # Mistral 22b 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",  # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-27b-bnb-4bit",  # Gemma 2x faster!

    "unsloth/Llama-3.2-1B-bnb-4bit",  # NEW! Llama 3.2 models
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
]  # More models at https://huggingface.co/unsloth

qwen_models = [
    "unsloth/Qwen2.5-Coder-32B-Instruct",  # Qwen 2.5 Coder 2x faster
    "unsloth/Qwen2.5-Coder-7B",
    "unsloth/Qwen2.5-14B-Instruct",  # 14B fits in a 16GB card
    "unsloth/Qwen2.5-7B",
    "unsloth/Qwen2.5-72B-Instruct",  # 72B fits in a 48GB card
]  # More models at https://huggingface.co/unsloth

model, tokenizer = FastQwen2Model.from_pretrained(
    model_name = "unsloth/Qwen2.5-Coder-1.5B-Instruct",
    max_seq_length = None,
    dtype = None,
    load_in_4bit = False,
    fix_tokenizer = False
    # token = "YOUR_HF_TOKEN", # HF Token for gated models
)

# save a copy because apply_chat_template() has in-place modifications
import copy

tokenizer_orig = copy.deepcopy(tokenizer)

def get_tool_definition_list():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_vector_sum",
                "description": "Get the sum of two vectors",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "list", "description": "First vector"},
                        "b": {"type": "list", "description": "Second vector"}
                    },
                    "required": ["a", "b"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_dot_product",
                "description": "Get the dot product of two vectors",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "list", "description": "First vector"},
                        "b": {"type": "list", "description": "Second vector"}
                    },
                    "required": ["a", "b"]
                }
            }
        },

    ]

user_query = {
    "role": "user",
    "content": "Find the sum of a = [1, -1, 2] and b = [3, 0, -4]."
}

def get_vector_sum(a: list[float], b: list[float]) -> list[float]:
    """
    Performs element-wise addition of two numerical vectors.

    Both vectors must be of the same length and contain numerical values.

    Args:
        a: First vector containing numerical values
        b: Second vector containing numerical values

    Returns:
        Resulting vector where each element is the sum of corresponding elements in a and b

    Raises:
        ValueError: If vectors have different lengths

    Example:
        >>> get_vector_sum([1, 2], [3, 4])
        [4, 6]
    """
    if len(a) != len(b):
        raise ValueError("Vectors must be of the same length")

    return [x + y for x, y in zip(a, b)]

messages = []

messages.append(user_query)

tokenizer = copy.deepcopy(tokenizer_orig)
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,
    add_special_tokens = False,
    padding = True,
    tools = [get_vector_sum],
    return_tensors = "pt",
).to("cuda")

print(tokenizer.decode(input_ids[0]))

#@title Function for Generation Constraint { display-mode: "form" }

from functools import partial
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor

JSON_ARR_GBNF = r"""
# This is the same as json.gbnf but we restrict whitespaces at the end of the root array
# Useful for generating JSON arrays
root   ::= arr
value  ::= object | array | string | number | ("true" | "false" | "null") ws
arr  ::=
  "[\n" ws (
            value
    (",\n" ws value)*
  )? "]"
object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws
array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws
string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* "\"" ws
number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= ([ \t\n] ws)?
"""

def generate_with_grammar(model, input_ids, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
    grammar = IncrementalGrammarConstraint(JSON_ARR_GBNF, start_rule_name = "root", tokenizer = tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

    partial_generate = partial(
        model.generate,
        do_sample = False,
        repetition_penalty = 1.1,
        num_return_sequences = 1,
        logits_processor = [grammar_processor],  # Ensure grammar_processor is accessible
        temperature = None,
        top_p = None,
        top_k = None,
        sliding_window = None,
    )

    # Execute generation with merged parameters
    return partial_generate(
        input_ids = input_ids,
        **kwargs
    )

output = generate_with_grammar(
    model = model,
    input_ids = input_ids
)

generated_tokens = output[:, input_ids.shape[1]:]

decoded_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens = True)

for i, message in enumerate(decoded_output):
    print(f"{message}")

import json

content = json.loads(decoded_output[0])
arguments = content[0]['arguments']
vector_a = arguments['a']
vector_b = arguments['b']
print(f"args a: {vector_a}, b: {vector_b}")

result = get_vector_sum(vector_a, vector_b)
print(f"result: {result}")

import random
import string


def generate_alphanumeric():
    characters = string.ascii_letters + string.digits
    result = ''.join(random.choice(characters) for _ in range(9))
    return result


messages = []

original_prompt = user_query['content']

prompt_with_context = f"""You are a super helpful AI assistant.
You are asked to answer a question based on the following context information.
Question:
{original_prompt}"""

messages.append({
    "role": "user",
    "content": prompt_with_context
})

tool_call_id = generate_alphanumeric()
tool_calls = [{
    "id": tool_call_id,
    "type": "function",
    "function": {
        "name": "get_vector_sum",
        "arguments": arguments
    }
}]

messages.append({
    "role": "assistant",
    "tool_calls": tool_calls
})
messages.append({
    "role": "tool",
    "name": "get_vector_sum",
    "content": result
})

messages.append({
    "role": "assistant",
    "content": "Answer:\n"
})

tokenizer = copy.deepcopy(tokenizer_orig)
tool_prompt = tokenizer.apply_chat_template(
    messages,
    continue_final_message = True,
    add_special_tokens = True,
    return_tensors = "pt",
    return_dict = True,
    tools = None,
)
tool_prompt = tool_prompt.to(model.device)

print(tokenizer.decode(tool_prompt['input_ids'][0]))

out = model.generate(**tool_prompt, max_new_tokens = 128)
generated_text = out[0, tool_prompt['input_ids'].shape[1]:]

print(tokenizer.decode(generated_text, skip_special_tokens = True))

tokenizer = copy.deepcopy(tokenizer_orig)
input_ids = tokenizer.apply_chat_template(
    [user_query],
    tokenize = True,
    add_generation_prompt = True,
    add_special_tokens = False,
    padding = True,
    tools = None,
    return_tensors = "pt",
).to("cuda")

print(tokenizer.decode(input_ids[0]))

output = model.generate(
    input_ids = input_ids,
    max_new_tokens = 1024
)

generated_tokens = output[:, input_ids.shape[1]:]
decoded_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens = True)

for i, message in enumerate(decoded_output):
    print(f"{message}")

user_query = {
    "role": "user",
    "content": "How much is the total cost of all inventory items in Euros?"
}

from typing import List, Optional, AnyStr
from pydantic import BaseModel, Field
import requests


class Item(BaseModel):
    id: int | None = Field(
        default = None,
        description = "Unique identifier for the item (auto-generated by database)"
    )
    item_code: str = Field(
        ...,
        min_length = 3,
        max_length = 20,
        description = "Unique SKU or product code for the item"
    )
    name: str = Field(
        ...,
        min_length = 2,
        max_length = 50,
        description = "Human-readable name of the item"
    )
    cost: float = Field(
        ...,
        gt = 0,
        description = "Unit cost in local currency (must be positive)"
    )
    quantity: int = Field(
        ...,
        ge = 0,
        description = "Current inventory quantity (non-negative integer)"
    )


def inventory_check(item_codes: Optional[List[str]], conversion_rate: float) -> float:
    """
    Calculates the total value of inventory items in the target conversion rate.
    When item_codes = None, calculates total value for all items.

    Args:
        item_codes: List of item codes to include. (None for all items)
        conversion_rate: Exchange rate to convert costs to target currency
    Returns:
        Total value of matching items in target currency, rounded to 2 decimals
    """
    all_items = get_all_items()

    # Process all items if None is passed
    if item_codes is None or len(item_codes) == 0:
        items_to_process = all_items
    else:
        # Convert to set for faster lookups
        target_codes = set(item_codes)
        items_to_process = [item for item in all_items if item.item_code in target_codes]

    # Calculate total value with conversion
    total = sum(
        item.cost * item.quantity * conversion_rate
        for item in items_to_process
    )

    return round(total, 2)


def get_all_items() -> List[Item]:
    """Fetches all the inventory items"""
    return [
        Item(
            item_code = "ITEM-001",
            name = "Apple",
            cost = 1.13,
            quantity = 4
        ),
        Item(
            item_code = "ITEM-002",
            name = "Bottled Water",
            cost = 1.04,
            quantity = 20
        ),
        Item(
            item_code = "ITEM-003",
            name = "Instant Ramen",
            cost = 10.13,
            quantity = 4
        )
    ]


def get_usd_to_euro_conversion_rate() -> float:
    """Gets the conversion rate from USD to EURO"""
    response = requests.get("https://api.frankfurter.app/latest?from=USD")
    response.raise_for_status()
    rate = response.json()["rates"]["EUR"]
    return rate

from transformers.utils import chat_template_utils

tools = [get_all_items, inventory_check, get_usd_to_euro_conversion_rate]

orig_tools = copy.deepcopy(tools)  # save a copy for later

for tool in tools:
    _ = chat_template_utils.get_json_schema(tool)

messages = []

messages.append(user_query)

tokenizer = copy.deepcopy(tokenizer_orig)
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,
    add_special_tokens = False,
    padding = True,
    tools = tools,  # pass the tools
    return_tensors = "pt",
).to("cuda")

print(tokenizer.decode(input_ids[0]))

output = generate_with_grammar(
    model = model,
    input_ids = input_ids
)

generated_tokens = output[:, input_ids.shape[1]:]

decoded_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens = True)

for i, message in enumerate(decoded_output):
    print(f"{message}")

import json

content = json.loads(decoded_output[0])
arguments = content[0]['arguments']
item_codes = arguments['item_codes']
conversion_rate = arguments['conversion_rate']
print(f"item_codes: {item_codes}, conversion_rate: {conversion_rate}")

result_total = inventory_check(item_codes, conversion_rate)
result_total

messages = []

original_prompt = user_query['content']

prompt_with_context = f"""You are a super helpful AI assistant.
You are asked to answer a question based on the following context information.
Question:
{original_prompt}"""

messages.append({
    "role": "user",
    "content": prompt_with_context
})

tool_call_id = generate_alphanumeric()
tool_calls = [{
    "id": tool_call_id,
    "type": "function",
    "function": {
        "name": "inventory_check",
        "arguments": arguments
    }
}]

messages.append({
    "role": "assistant",
    "tool_calls": tool_calls
})
messages.append({
    "role": "tool",
    "name": "inventory_check",
    "content": result_total
})

messages.append({
    "role": "assistant",
    "content": "Answer:\n"
})

tokenizer = copy.deepcopy(tokenizer_orig)
tool_prompt = tokenizer.apply_chat_template(
    messages,
    continue_final_message = True,
    add_special_tokens = True,
    return_tensors = "pt",
    return_dict = True,
    tools = None,
)
tool_prompt = tool_prompt.to(model.device)

print(tokenizer.decode(tool_prompt['input_ids'][0]))

out = model.generate(**tool_prompt, max_new_tokens = 128)
generated_text = out[0, tool_prompt['input_ids'].shape[1]:]

print(tokenizer.decode(generated_text, skip_special_tokens = True))

user_query = {
    "role": "user",
    "content": f"""How much is the total inventory cost of item name: Bottled Water in Euros? Ensure to use fetch_item_by_name first for fetching the item code"""
}

def fetch_item_by_name(item_name: str) -> Optional[Item]:
    """
    Fetch an item by name and returns the Item object.

    Args:
        item_name: The human-readable name of the item to fetch
    Returns:
        Optional[Item]: The Item with the given name, or None if not found
    """
    all_items = get_all_items()
    return next((item for item in all_items if item.name == item_name), None)


# append to the tools list
tools = copy.deepcopy(orig_tools)

# place it at the top of the list
tools.insert(0, fetch_item_by_name)

from transformers.utils import chat_template_utils

for tool in tools:
    _ = chat_template_utils.get_json_schema(tool)

messages = []

messages.append(user_query)

tokenizer = copy.deepcopy(tokenizer_orig)
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,
    add_special_tokens = False,
    padding = True,
    tools = tools,
    return_tensors = "pt",
).to("cuda")

print(tokenizer.decode(input_ids[0]))

output = generate_with_grammar(
    model = model,
    input_ids = input_ids
)

generated_tokens = output[:, input_ids.shape[1]:]

decoded_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens = True)

for i, message in enumerate(decoded_output):
    print(f"{message}")

import json

content = json.loads(decoded_output[0])
arguments_for_item_name = content[0]['arguments']
item_name = arguments_for_item_name['item_name']

item_code = fetch_item_by_name(item_name).item_code
print(f"item_name: {item_name}, item_code: {item_code}")

arguments_for_inventory_check = content[1]['arguments']
conversion_rate = arguments_for_inventory_check['conversion_rate']
print(f"conversion_rate: {conversion_rate}")

result_total = inventory_check([item_code], conversion_rate)
result_total

messages = []

original_prompt = user_query['content']

prompt_with_context = f"""You are a super helpful AI assistant.
You are asked to answer a question based on the following context information.
Question:
{original_prompt}"""

messages.append({
    "role": "user",
    "content": prompt_with_context
})

tool_call_id = generate_alphanumeric()
tool_calls = [{
    "id": tool_call_id,
    "type": "function",
    "function": {
        "name": "inventory_check",
        "arguments": arguments
    }
}]

messages.append({
    "role": "assistant",
    "tool_calls": tool_calls
})
messages.append({
    "role": "tool",
    "name": "inventory_check",
    "content": result_total  # pass the result total
})

messages.append({
    "role": "assistant",
    "content": "Answer:\n"
})

tokenizer = copy.deepcopy(tokenizer_orig)
tool_prompt = tokenizer.apply_chat_template(
    messages,
    continue_final_message = True,
    add_special_tokens = True,
    return_tensors = "pt",
    return_dict = True,
    tools = None,
)
tool_prompt = tool_prompt.to(model.device)

print(tokenizer.decode(tool_prompt['input_ids'][0]))

out = model.generate(**tool_prompt, max_new_tokens = 128)
generated_text = out[0, tool_prompt['input_ids'].shape[1]:]

print(tokenizer.decode(generated_text, skip_special_tokens = True))

