"""
Title: Function Calling with Keras Hub models
Author: [Laxmareddy Patlolla](https://github.com/laxmareddyp), [Divyashree Sreepathihalli](https://github.com/divyashreepathihalli)
Date created: 2025/07/08
Last modified: 2025/07/08
Description: A guide to using the function calling feature in Keras Hub with Gemma 3 and Mistral.
Accelerator: GPU
"""

"""
## Introduction

Tool calling is a powerful new feature in modern large language models that allows them to use external tools, such as Python functions, to answer questions and perform actions. Instead of just generating text, a tool-calling model can generate code that calls a function you've provided, allowing it to interact with the real world, access live data, and perform complex calculations.

In this guide, we'll walk you through a simple example of tool calling with the Gemma 3 and Mistral models and Keras Hub. We'll show you how to:

1.  Define a tool (a Python function).
2.  Tell the models about the tool.
3.  Use the model to generate code that calls the tool.
4.  Execute the code and feed the result back to the model.
5.  Get a final, natural-language response from the model.

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
import keras
import keras_hub
import kagglehub
import numpy as np

os.environ["KERAS_BACKEND"] = "jax"
keras.config.set_dtype_policy("bfloat16")

# Authenticate with Kaggle
kagglehub.login()

"""
### Loading the Model

Next, we'll load the Gemma 3 model from KerasHub. We're using the `gemma3_instruct_4b` preset, which is a version of the model that has been specifically fine-tuned for instruction following and tool calling.
"""

gemma = keras_hub.models.Gemma3CausalLM.from_preset("gemma3_instruct_4b")

"""
### Defining a Tool

Now, let's define a simple tool that we want our model to be able to use. For this example, we'll create a Python function called `convert` that can convert one currency to another.
"""


def convert(amount: float, currency: str, new_currency: str) -> float:
    """Convert the currency with the latest exchange rate

    Args:
      amount: The amount of currency to convert
      currency: The currency to convert from
      new_currency: The currency to convert to
    """
    # In a real application, this function would call an API to get the latest
    # exchange rate. For this example, we'll just use a fixed rate.
    if currency == "USD" and new_currency == "EUR":
        return amount * 0.9
    else:
        return amount


"""
### Telling the Model About the Tool

Now that we have a tool, we need to tell the Gemma 3 model about it. We do this by providing a carefully crafted prompt that includes:

1.  A description of the tool calling process.
2.  The Python code for the tool, including its function signature and docstring.
3.  The user's question.

Here's the prompt we'll use:
"""

message = '''
<start_of_turn>user
At each turn, if you decide to invoke any of the function(s), it should be wrapped with ```tool_code```. The python methods described below are imported and available, you can only use defined methods and must not reimplement them. The generated code should be readable and efficient. I will provide the response wrapped in ```tool_output```, use it to call more tools or generate a helpful, friendly response. When using a ```tool_call``` think step by step why and how it should be used.

The following Python methods are available:

```python
def convert(amount: float, currency: str, new_currency: str) -> float:
    """Convert the currency with the latest exchange rate

    Args:
      amount: The amount of currency to convert
      currency: The currency to convert from
      new_currency: The currency to convert to
    """
```


User: What is $200,000 in EUR?<end_of_turn>
<start_of_turn>model
'''

"""
### Generating the Tool Call

Now, let's pass this prompt to the model and see what it generates.
"""

print(gemma.generate(message))

"""
As you can see, the model has correctly identified that it can use the `convert` function to answer the question, and it has generated the corresponding Python code.
"""

"""
### Executing the Tool Call and Getting a Final Answer

In a real application, you would now take this generated code, execute it, and feed the result back to the model. We can simulate this by creating a new prompt that includes the output of the tool call.
"""

message2 = '''
<start_of_turn>user
At each turn, if you decide to invoke any of the function(s), it should be wrapped with ```tool_code```. The python methods described below are imported and available, you can only use defined methods and must not reimplement them. The generated code should be readable and efficient. I will provide the response wrapped in ```tool_output```, use it to call more tools or generate a helpful, friendly response. When using a ```tool_call``` think step by step why and how it should be used.

The following Python methods are available:

```python
def convert(amount: float, currency: str, new_currency: str) -> float:
    """Convert the currency with the latest exchange rate

    Args:
      amount: The amount of currency to convert
      currency: The currency to convert from
      new_currency: The currency to convert to
    """
```


User: What is $200,000 in EUR?<end_of_turn>
<start_of_turn>model
```tool_code
print(convert(200000, "USD", "EUR"))
```<end_of_turn>
<start_of_turn>user
```tool_output
180000.0
```
<end_of_turn>
<start_of_turn>model
'''

"""
Now, let's pass this new prompt to the model.
"""

print(gemma.generate(message2))

"""
This time, the model has the information it needs to answer the question, and it generates a natural-language response.
"""

"""
## Mistral

Mistral differs from Gemma in its approach to tool calling, as it requires a specific format and defines special control tokens for this purpose. This JSON-based syntax for tool calling is also adopted by other models, such as Qwen and Llama.

We will now extend the example to a more exciting use case: building a flight booking agent. This agent will be able to search for appropriate flights and book them automatically.

To do this, we will first download the Mistral model using Keras Hub. For agentic AI with Mistral, low-level access to tokenization is necessary due to the use of control tokens. Therefore, we will instantiate the tokenizer and model separately, and disable the preprocessor for the model.
"""

tokenizer = keras_hub.tokenizers.MistralTokenizer.from_preset(
    "kaggle://keras/mistral/keras/mistral_0.3_instruct_7b_en"
)

mistral = keras_hub.models.MistralCausalLM.from_preset(
    "kaggle://keras/mistral/keras/mistral_0.3_instruct_7b_en", preprocessor=None
)

"""
Next, we'll define functions for tokenization. The `preprocess` function will take a tokenized conversation in list form and format it correctly for the model. We'll also create an additional function, `encode_instruction`, for tokenizing text and adding instruction control tokens.
"""


def preprocess(messages, sequence_length=8192):
    concatd = np.expand_dims(np.concatenate(messages), 0)
    return {
        "token_ids": np.pad(concatd, ((0, 0), (0, sequence_length - concatd.shape[1]))),
        "padding_mask": np.expand_dims(
            np.arange(sequence_length) < concatd.shape[1], 0
        ).astype(int),
    }


def encode_instruction(text):
    return [
        [tokenizer.token_to_id("[INST]")],
        tokenizer(text),
        [tokenizer.token_to_id("[/INST]")],
    ]


"""
Now, we'll define a function, `try_parse_funccall`, to handle the model's function calls. These calls are identified by the `[TOOL_CALLS]` control token. The function will parse the subsequent data, which is in JSON format. Mistral also requires us to add a random call ID to each function call. Finally, the function will call the matching tool and encode its results using the `[TOOL_RESULTS]` control token.
"""


def try_parse_funccall(response):
    # find the tool call in the response, if any
    tool_call_id = tokenizer.token_to_id("[TOOL_CALLS]")
    pos = np.where(response == tool_call_id)[0]
    if not len(pos):
        return [response]
    pos = pos[0]
    # try to decode it as JSON
    decoder = json.JSONDecoder()
    tool_calls, _ = decoder.raw_decode(tokenizer.detokenize(response[pos + 1 :]))
    if not isinstance(tool_calls, list) or not all(
        isinstance(item, dict) for item in tool_calls
    ):
        return
    # assign a random call ID
    for call in tool_calls:
        call["id"] = "".join(random.choices(string.ascii_letters + string.digits, k=9))
    res = [response[:pos], [tool_call_id], tokenizer(json.dumps(tool_calls))]
    # call the tools and extend the conversation
    for call in tool_calls:
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


"""
We will extend our set of tools to include functions for currency conversion, finding flights, and booking flights. For this example, we'll use mock implementations for these functions, meaning they will return dummy data instead of interacting with real services.
"""

tools = {
    "convert_currency": lambda amount, currency, new_currency: (
        f"{amount*0.85:.2f}" if currency == "USD" else f"{amount/0.85:.2f}"
    ),
    "find_flights": lambda origin, destination, date: [
        {"id": 1, "price": "USD 220", "stops": 2, "duration": 4.5},
        {"id": 2, "price": "USD 22", "stops": 1, "duration": 2.0},
        {"id": 3, "price": "USD 240", "stops": 2, "duration": 13.2},
    ],
    "book_flight": lambda id: globals().update(flight_booked=True),
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
while not flight_booked:
    # query the model
    res = mistral.generate(
        preprocess(messages), max_length=8192, stop_token_ids=[2], strip_prompt=True
    )
    # output the model's response, add separator line for legibility
    print(
        tokenizer.detokenize(res["token_ids"][0, : np.argmax(~res["padding_mask"])]),
        f"\n\n\n{'-'*100}\n\n",
    )
    # perform tool calls and extend `messages`
    messages.extend(try_parse_funccall(res["token_ids"][0]))

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
