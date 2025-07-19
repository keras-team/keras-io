"""
Title: Function Calling with KerasHub models
Author: [Laxmareddy Patlolla](https://github.com/laxmareddyp), [Divyashree Sreepathihalli](https://github.com/divyashreepathihalli)
Date created: 2025/07/08
Last modified: 2025/07/10
Description: A guide to using the function calling feature in KerasHub with Gemma 3 and Mistral.
Accelerator: GPU
"""

"""
## Introduction

Tool calling is a powerful new feature in modern large language models that allows them to use external tools, such as Python functions, to answer questions and perform actions. Instead of just generating text, a tool-calling model can generate code that calls a function you've provided, allowing it to interact with the real world, access live data, and perform complex calculations.

In this guide, we'll walk you through a simple example of tool calling with the Gemma 3 and Mistral models and KerasHub. We'll show you how to:

1. Define a tool (a Python function).
2. Tell the models about the tool.
3. Use the model to generate code that calls the tool.
4. Execute the code and feed the result back to the model.
5. Get a final, natural-language response from the model.

Let's get started!
"""

"""
## Setup

First, let's import the necessary libraries and configure our environment. We'll be using KerasHub to download and run the language models, and we'll need to authenticate with Kaggle to access the model weights.
"""

import os
import json
import random
import string
import re
import ast
import io
import sys
import contextlib

# Set backend before importing Keras
os.environ["KERAS_BACKEND"] = "jax"

import keras
import keras_hub
import kagglehub
import numpy as np

# Constants
USD_TO_EUR_RATE = 0.85

# Set the default dtype policy to bfloat16 for improved performance and reduced memory usage on supported hardware (e.g., TPUs, some GPUs)
keras.config.set_dtype_policy("bfloat16")

# Authenticate with Kaggle
# In Google Colab, you can set KAGGLE_USERNAME and KAGGLE_KEY as secrets,
# and kagglehub.login() will automatically detect and use them:
# kagglehub.login()

"""
## Loading the Model

Next, we'll load the Gemma 3 model from KerasHub. We're using the `gemma3_instruct_4b` preset, which is a version of the model that has been specifically fine-tuned for instruction following and tool calling.
"""

try:
    gemma = keras_hub.models.Gemma3CausalLM.from_preset("gemma3_instruct_4b")
    print("✅ Gemma 3 model loaded successfully")
except Exception as e:
    print(f"❌ Error loading Gemma 3 model: {e}")
    print("Please ensure you have the correct model preset and sufficient resources.")
    raise

"""
## Defining a Tool

Now, let's define a simple tool that we want our model to be able to use. For this example, we'll create a Python function called `convert` that can convert one currency to another.
"""


def convert(amount, currency, new_currency):
    """Convert the currency with the latest exchange rate

    Args:
      amount: The amount of currency to convert
      currency: The currency to convert from
      new_currency: The currency to convert to
    """
    # Input validation
    if amount < 0:
        raise ValueError("Amount cannot be negative")

    if not isinstance(currency, str) or not isinstance(new_currency, str):
        raise ValueError("Currency codes must be strings")

    # Normalize currency codes to uppercase to handle model-generated lowercase codes
    currency = currency.upper().strip()
    new_currency = new_currency.upper().strip()

    # In a real application, this function would call an API to get the latest
    # exchange rate. For this example, we'll just use a fixed rate.
    if currency == "USD" and new_currency == "EUR":
        return amount * USD_TO_EUR_RATE
    elif currency == "EUR" and new_currency == "USD":
        return amount / USD_TO_EUR_RATE
    else:
        raise NotImplementedError(
            f"Currency conversion from {currency} to {new_currency} is not supported."
        )


"""
## Telling the Model About the Tool

Now that we have a tool, we need to tell the Gemma 3 model about it. We do this by providing a carefully crafted prompt that includes:

1. A description of the tool calling process.
2. The Python code for the tool, including its function signature and docstring.
3. The user's question.

Here's the prompt we'll use:
"""

message = '''
<start_of_turn>user
At each turn, if you decide to invoke any of the function(s), it should be wrapped with ```tool_code```. The python methods described below are imported and available, you can only use defined methods and must not reimplement them. The generated code should be readable and efficient. I will provide the response wrapped in ```tool_output```, use it to call more tools or generate a helpful, friendly response. When using a ```tool_call``` think step by step why and how it should be used.

The following Python methods are available:

~~~python
```python
def convert(amount, currency, new_currency):
    """Convert the currency with the latest exchange rate

    Args:
      amount: The amount of currency to convert
      currency: The currency to convert from
      new_currency: The currency to convert to
    """
```
~~~

User: What is $200,000 in EUR?<end_of_turn>
<start_of_turn>model
'''

"""
## Generating the Tool Call

Now, let's pass this prompt to the model and see what it generates.
"""

print(gemma.generate(message))

"""
As you can see, the model has correctly identified that it can use the `convert` function to answer the question, and it has generated the corresponding Python code.
"""

"""
## Executing the Tool Call and Getting a Final Answer

In a real application, you would now take this generated code, execute it, and feed the result back to the model. Let's create a practical example that shows how to do this:
"""

# First, let's get the model's response
response = gemma.generate(message)
print("Model's response:")
print(response)


# Extract the tool call from the response
def extract_tool_call(response_text):
    """Extract tool call from the model's response."""
    tool_call_pattern = r"```tool_code\s*\n(.*?)\n```"
    match = re.search(tool_call_pattern, response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def capture_code_output(code_string, globals_dict=None, locals_dict=None):
    """
    Executes Python code and captures any stdout output.


    This function uses eval() and exec() which can execute arbitrary code.
    NEVER use this function with untrusted code in production environments.
    Always validate and sanitize code from LLMs before execution.

    Args:
        code_string (str): The code to execute (expression or statements).
        globals_dict (dict, optional): Global variables for execution.
        locals_dict (dict, optional): Local variables for execution.

    Returns:
        The captured stdout output if any, otherwise the return value of the expression,
        or None if neither.
    """
    if globals_dict is None:
        globals_dict = {}
    if locals_dict is None:
        locals_dict = globals_dict

    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            try:
                # Try to evaluate as an expression
                result = eval(code_string, globals_dict, locals_dict)
            except SyntaxError:
                # If not an expression, execute as statements
                exec(code_string, globals_dict, locals_dict)
                result = None
    except Exception as e:
        return f"Error during code execution: {e}"

    stdout_output = output.getvalue()
    if stdout_output.strip():
        return stdout_output
    return result


# Extract and execute the tool call
tool_code = extract_tool_call(response)
if tool_code:
    print(f"\nExtracted tool call: {tool_code}")
    try:
        local_vars = {"convert": convert}
        tool_result = capture_code_output(tool_code, globals_dict=local_vars)
        print(f"Tool execution result: {tool_result}")

        # Create the next message with the tool result
        message_with_result = f'''
<start_of_turn>user
At each turn, if you decide to invoke any of the function(s), it should be wrapped with ```tool_code```. The python methods described below are imported and available, you can only use defined methods and must not reimplement them. The generated code should be readable and efficient. I will provide the response wrapped in ```tool_output```, use it to call more tools or generate a helpful, friendly response. When using a ```tool_call``` think step by step why and how it should be used.

The following Python methods are available:

~~~python
```python
def convert(amount, currency, new_currency):
    """Convert the currency with the latest exchange rate

    Args:
      amount: The amount of currency to convert
      currency: The currency to convert from
      new_currency: The currency to convert to
    """
```
~~~

User: What is $200,000 in EUR?<end_of_turn>
<start_of_turn>model
```tool_code
print(convert(200000, "USD", "EUR"))
```<end_of_turn>
<start_of_turn>user
```tool_output
{tool_result}
```<end_of_turn>
<start_of_turn>model
'''

        # Get the final response
        final_response = gemma.generate(message_with_result)
        print("\nFinal response:")
        print(final_response)

    except Exception as e:
        print(f"Error executing tool call: {e}")
else:
    print("No tool call found in the response")

"""
## Automated Tool Call Execution Loop

Let's create a more sophisticated example that shows how to automatically handle multiple tool calls in a conversation:
"""


def automated_tool_calling_example():
    """Demonstrate automated tool calling with a conversation loop."""

    conversation_history = []
    max_turns = 5

    # Initial user message
    user_message = "What is $500 in EUR, and then what is that amount in USD?"

    # Define base prompt outside the loop for better performance
    base_prompt = f'''
<start_of_turn>user
At each turn, if you decide to invoke any of the function(s), it should be wrapped with ```tool_code```. The python methods described below are imported and available, you can only use defined methods and must not reimplement them. The generated code should be readable and efficient. I will provide the response wrapped in ```tool_output```, use it to call more tools or generate a helpful, friendly response. When using a ```tool_call``` think step by step why and how it should be used.

The following Python methods are available:

~~~python
```python
def convert(amount, currency, new_currency):
    """Convert the currency with the latest exchange rate

    Args:
      amount: The amount of currency to convert
      currency: The currency to convert from
      new_currency: The currency to convert to
    """
```
~~~

User: {user_message}<end_of_turn>
<start_of_turn>model
'''

    for turn in range(max_turns):
        print(f"\n--- Turn {turn + 1} ---")

        # Build conversation context by appending history to base prompt
        context = base_prompt
        for hist in conversation_history:
            context += hist + "\n"

        # Get model response
        response = gemma.generate(context)
        print(f"Model response: {response}")

        # Extract tool call
        tool_code = extract_tool_call(response)

        if tool_code:
            print(f"Executing: {tool_code}")
            try:
                local_vars = {"convert": convert}
                tool_result = capture_code_output(tool_code, globals_dict=local_vars)
                conversation_history.append(
                    f"```tool_code\n{tool_code}\n```<end_of_turn>"
                )
                conversation_history.append(
                    f"<start_of_turn>user\n```tool_output\n{tool_result}\n```<end_of_turn>"
                )
                conversation_history.append(f"<start_of_turn>model\n")
                print(f"Tool result: {tool_result}")
            except Exception as e:
                print(f"Error executing tool: {e}")
                break
        else:
            print("No tool call found - conversation complete")
            break

    print("\n--- Final Conversation ---")
    print(context)
    for hist in conversation_history:
        print(hist)


# Run the automated example
print("Running automated tool calling example:")
automated_tool_calling_example()

"""
## Mistral

Mistral differs from Gemma in its approach to tool calling, as it requires a specific format and defines special control tokens for this purpose. This JSON-based syntax for tool calling is also adopted by other models, such as Qwen and Llama.

We will now extend the example to a more exciting use case: building a flight booking agent. This agent will be able to search for appropriate flights and book them automatically.

To do this, we will first download the Mistral model using KerasHub. For agentic AI with Mistral, low-level access to tokenization is necessary due to the use of control tokens. Therefore, we will instantiate the tokenizer and model separately, and disable the preprocessor for the model.
"""

tokenizer = keras_hub.tokenizers.MistralTokenizer.from_preset(
    "kaggle://keras/mistral/keras/mistral_0.3_instruct_7b_en"
)

try:
    mistral = keras_hub.models.MistralCausalLM.from_preset(
        "kaggle://keras/mistral/keras/mistral_0.3_instruct_7b_en", preprocessor=None
    )
    print("✅ Mistral model loaded successfully")
except Exception as e:
    print(f"❌ Error loading Mistral model: {e}")
    print("Please ensure you have the correct model preset and sufficient resources.")
    raise

"""
Next, we'll define functions for tokenization. The `preprocess` function will take a tokenized conversation in list form and format it correctly for the model. We'll also create an additional function, `encode_instruction`, for tokenizing text and adding instruction control tokens.
"""


def preprocess(messages, sequence_length=8192):
    """Preprocess tokenized messages for the Mistral model.

    Args:
        messages: List of tokenized message sequences
        sequence_length: Maximum sequence length for the model

    Returns:
        Dictionary containing token_ids and padding_mask
    """
    concatd = np.expand_dims(np.concatenate(messages), 0)

    # Truncate if the sequence is too long
    if concatd.shape[1] > sequence_length:
        concatd = concatd[:, :sequence_length]

    # Calculate padding needed
    padding_needed = max(0, sequence_length - concatd.shape[1])

    return {
        "token_ids": np.pad(concatd, ((0, 0), (0, padding_needed))),
        "padding_mask": np.expand_dims(
            np.arange(sequence_length) < concatd.shape[1], 0
        ).astype(int),
    }


def encode_instruction(text):
    """Encode instruction text with Mistral control tokens.

    Args:
        text: The instruction text to encode

    Returns:
        List of tokenized sequences with instruction control tokens
    """
    return [
        [tokenizer.token_to_id("[INST]")],
        tokenizer(text),
        [tokenizer.token_to_id("[/INST]")],
    ]


"""
Now, we'll define a function, `try_parse_funccall`, to handle the model's function calls. These calls are identified by the `[TOOL_CALLS]` control token. The function will parse the subsequent data, which is in JSON format. Mistral also requires us to add a random call ID to each function call. Finally, the function will call the matching tool and encode its results using the `[TOOL_RESULTS]` control token.
"""


def try_parse_funccall(response):
    """Parse function calls from Mistral model response and execute tools.

    Args:
        response: Tokenized model response

    Returns:
        List of tokenized sequences including tool results
    """
    # find the tool call in the response, if any
    tool_call_id = tokenizer.token_to_id("[TOOL_CALLS]")
    pos = np.where(response == tool_call_id)[0]
    if not len(pos):
        return [response]
    pos = pos[0]

    try:
        decoder = json.JSONDecoder()
        tool_calls, _ = decoder.raw_decode(tokenizer.detokenize(response[pos + 1 :]))
        if not isinstance(tool_calls, list) or not all(
            isinstance(item, dict) for item in tool_calls
        ):
            return [response]

        res = []  # Initialize result list
        # assign a random call ID
        for call in tool_calls:
            call["id"] = "".join(
                random.choices(string.ascii_letters + string.digits, k=9)
            )
            if call["name"] not in tools:
                continue  # Skip unknown tools
            res.append([tokenizer.token_to_id("[TOOL_RESULTS]")])
            res.append(
                tokenizer(
                    json.dumps(
                        {
                            "content": tools[call["name"]](**call["arguments"]),
                            "call_id": call["id"],
                        }
                    )
                )
            )
            res.append([tokenizer.token_to_id("[/TOOL_RESULTS]")])
        return res
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        # Log the error for debugging
        print(f"Error parsing tool call: {e}")
        return [response]


"""
We will extend our set of tools to include functions for currency conversion, finding flights, and booking flights. For this example, we'll use mock implementations for these functions, meaning they will return dummy data instead of interacting with real services.
"""

tools = {
    "convert_currency": lambda amount, currency, new_currency: (
        f"{amount*USD_TO_EUR_RATE:.2f}"
        if currency == "USD" and new_currency == "EUR"
        else (
            f"{amount/USD_TO_EUR_RATE:.2f}"
            if currency == "EUR" and new_currency == "USD"
            else f"Error: Unsupported conversion from {currency} to {new_currency}"
        )
    ),
    "find_flights": lambda origin, destination, date: [
        {"id": 1, "price": "USD 220", "stops": 2, "duration": 4.5},
        {"id": 2, "price": "USD 22", "stops": 1, "duration": 2.0},
        {"id": 3, "price": "USD 240", "stops": 2, "duration": 13.2},
    ],
    "book_flight": lambda id: {
        "status": "success",
        "message": f"Flight {id} booked successfully",
    },
}

"""
It's crucial to inform the model about these available functions at the very beginning of the conversation. To do this, we will define the available tools in a specific JSON format, as shown in the following code block.
"""

tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "Convert the currency with the latest exchange rate",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "The amount"},
                    "currency": {
                        "type": "string",
                        "description": "The currency to convert from",
                    },
                    "new_currency": {
                        "type": "string",
                        "description": "The currency to convert to",
                    },
                },
                "required": ["amount", "currency", "new_currency"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_flights",
            "description": "Query price, time, number of stopovers and duration in hours for flights for a given date",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": "The city to depart from",
                    },
                    "destination": {
                        "type": "string",
                        "description": "The destination city",
                    },
                    "date": {
                        "type": "string",
                        "description": "The date in YYYYMMDD format",
                    },
                },
                "required": ["origin", "destination", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_flight",
            "description": "Book the flight with the given id",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "number",
                        "description": "The numeric id of the flight to book",
                    },
                },
                "required": ["id"],
            },
        },
    },
]

"""
We will define the conversation as a `messages` list. At the very beginning of this list, we need to include a Beginning-Of-Sequence (BOS) token. This is followed by the tool definitions, which must be wrapped in `[AVAILABLE_TOOLS]` and `[/AVAILABLE_TOOLS]` control tokens.
"""

messages = [
    [tokenizer.token_to_id("<s>")],
    [tokenizer.token_to_id("[AVAILABLE_TOOLS]")],
    tokenizer(json.dumps(tool_definitions)),
    [tokenizer.token_to_id("[/AVAILABLE_TOOLS]")],
]

"""
Now, let's get started! We will task the model with the following: **Book the most comfortable flight from Linz to London on the 24th of July 2025, but only if it costs less than 20€ as of the latest exchange rate.**
"""

messages.extend(
    encode_instruction(
        "Book the most comfortable flight from Linz to London on the 24th of July 2025, but only if it costs less than 20€ as of the latest exchange rate."
    )
)

"""
In an agentic AI system, the model interacts with its tools through a sequence of messages. We will continue to handle these messages until the flight is successfully booked.
For educational purposes, we will output the tool calls issued by the model; typically, a user would not see this level of detail. It's important to note that after the tool call JSON, the data must be truncated. If not, a less capable model may 'babble', outputting redundant or confused data.
"""

flight_booked = False
max_iterations = 10  # Prevent infinite loops
iteration_count = 0

while not flight_booked and iteration_count < max_iterations:
    iteration_count += 1
    # query the model
    res = mistral.generate(
        preprocess(messages), max_length=8192, stop_token_ids=[2], strip_prompt=True
    )
    # output the model's response, add separator line for legibility
    response_text = tokenizer.detokenize(
        res["token_ids"][0, : np.argmax(~res["padding_mask"])]
    )
    print(response_text, f"\n\n\n{'-'*100}\n\n")

    # Check for tool calls and track booking status
    tool_call_id = tokenizer.token_to_id("[TOOL_CALLS]")
    pos = np.where(res["token_ids"][0] == tool_call_id)[0]
    if len(pos) > 0:
        try:
            decoder = json.JSONDecoder()
            tool_calls, _ = decoder.raw_decode(
                tokenizer.detokenize(res["token_ids"][0][pos[0] + 1 :])
            )
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    if isinstance(call, dict) and call.get("name") == "book_flight":
                        # Check if book_flight was called successfully
                        flight_booked = True
                        break
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass

    # perform tool calls and extend `messages`
    messages.extend(try_parse_funccall(res["token_ids"][0]))

if not flight_booked:
    print("Maximum iterations reached. Flight booking was not completed.")

"""
For understandability, here's the conversation as received by the model, i.e. when truncating after the tool calling JSON:

* **User:**
```
Book the most comfortable flight from Linz to London on the 24th of July 2025, but only if it costs less than 20€ as of the latest exchange rate.
```

* **Model:**
```
[{"name": "find_flights", "arguments": {"origin": "Linz", "destination": "London", "date": "20250724"}}]
```
* **Tool Output:**
```
[{"id": 1, "price": "USD 220", "stops": 2, "duration": 4.5}, {"id": 2, "price": "USD 22", "stops": 1, "duration": 2.0}, {"id": 3, "price": "USD 240", "stops": 2, "duration": 13.2}]
```
* **Model:**
```
Now let's convert the price from USD to EUR using the latest exchange rate:

 [{"name": "convert_currency", "arguments": {"amount": 22, "currency": "USD", "new_currency": "EUR"}}]
```
* **Tool Output:**
```
"18.70"
```
* **Model:**
```
The price of the flight with the id 2 in EUR is 18.70. Since it is below the 20€ limit, let's book this flight:

 [{"name": "book_flight", "arguments": {"id": 2}}]
```

It's important to acknowledge that you might have to run the model a few times to obtain a good output as depicted above. As a 7-billion parameter model, Mistral may still make several mistakes, such as misinterpreting data, outputting malformed tool calls, or making incorrect decisions. However, the continued development in this field paves the way for increasingly powerful agentic AI in the future.
"""

"""
## Conclusion

Tool calling is a powerful feature that allows large language models to interact with the real world, access live data, and perform complex calculations. By defining a set of tools and telling the model about them, you can create sophisticated applications that go far beyond simple text generation.
"""
