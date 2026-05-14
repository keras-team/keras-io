# Native Function Calling with FunctionGemma in KerasHub

**Author:** [Laxmareddy Patlolla](https://github.com/laxmareddyp)<br>
**Date created:** 2026/02/24<br>
**Last modified:** 2026/03/06<br>
**Description:** A guide to using the function calling feature in KerasHub with FunctionGemma.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_hub/function_gemma_with_keras.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/keras_hub/function_gemma_with_keras.py)



---
## Introduction

This example demonstrates how to build a multi-tool AI Assistant using FunctionGemma native
function calling capabilities in KerasHub. The Assistant can execute multiple tools
including web search, stock prices, world time, and system stats.

---
## Overview

Function calling enables language models to interact with
external APIs and tools by generating structured function calls. This implementation
uses FunctionGemma native function calling format with proper tool declarations and
function call parsing.

---
## Architecture

The Assistant follows this flow:

-  **Tool Definition**: Tools are defined in Function Calling Format JSON format
-  **Prompt Formatting**: Tools are injected into the prompt using FunctionGemma template
-  **Model Generation**: The model generates a structured function call
-  **Parsing**: Extract function name and arguments from the response
-  **Execution**: Execute the corresponding Python function
-  **Response**: Return formatted results to the user

---
## Key Components

- `TOOL_DEFINITIONS`: Function Calling Format tool schemas
- `format_prompt_with_tools()`: Manual chat template implementation
- `parse_function_call()`: Extract function calls from model output
- `TOOLS`: Registry mapping tool names to Python functions

---
## Setup
Before starting this tutorial, complete the following steps:
```
pip install -U keras-hub
pip install yfinance ddgs psutil pytz
```

Get access to `FunctionGemma` by logging into your [Kaggle](https://www.kaggle.com/models/keras/function-gemma) account and selecting Acknowledge license for a FunctionGemma model.

Generate a `Kaggle Access Token` by visiting kaggle [settings](https://www.kaggle.com/settings) page and add it to your Colab Secrets:

`KAGGLE_USERNAME = "Your kaggle user name"`

`KAGGLE_KEY=" Your kaggle Key"`

This notebook will run on either CPU or GPU.

---
## Install Python packages


```python
import os
import datetime
import psutil
import yfinance as yf
from ddgs import DDGS
import keras_hub
import re
import pytz
import warnings

warnings.filterwarnings("ignore")

os.environ["KERAS_BACKEND"] = "jax"
```

# Tool Implementations

We'll start by defining the actual Python functions that our Assistant will be able
to call. Each of these functions represents a "tool" that extends the model's
capabilities beyond just text generation.

---
## Web Search Tool

This function allows the model to search the web for information using DuckDuckGo.
In a real application, you might use more sophisticated search APIs or custom
search implementations tailored to your domain.


```python

def web_search(query, max_results=5):
    """Search the web using DuckDuckGo.

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 5)

    Returns:
        Dictionary with "results" key containing list of search results,
        each with "title", "description", and "link" keys.
        Returns {"error": message} on failure.
    """
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    {
                        "title": r.get("title"),
                        "description": r.get("body"),
                        "link": r.get("href"),
                    }
                )
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}
    return {"results": results}

```

---
## Stock Price Tool

This function retrieves real-time stock prices from Yahoo Finance. The model
can use this to answer questions about current market prices for publicly
traded companies.


```python

def get_stock_price(symbol):
    """Get real-time stock price from Yahoo Finance.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "GOOGL")

    Returns:
        Dictionary with "symbol", "price", and "currency" keys.
        Returns {"error": message} on failure.
    """
    try:
        t = yf.Ticker(symbol)
        info = t.fast_info
        if info.last_price is None:
            raise ValueError("Price not available")
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        return {
            "symbol": symbol,
            "price": round(info.last_price, 2),
            "currency": info.currency,
            "date": current_date,
        }
    except Exception as e:
        return {"error": f"Failed to get stock price for '{symbol}'. Reason: {e}"}

```

---
## System Statistics Tool

This function provides information about the current system's CPU and memory
usage. It demonstrates how the model can interact with the local environment.


```python

def get_system_stats(**kwargs):
    """Get CPU and memory usage.

    Args:
        **kwargs: Ignore any additional arguments the model might hallucinate.
    """

    mem = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "mem_percent": mem.percent,
        "used_gb": round(mem.used / 1e9, 2),
        "total_gb": round(mem.total / 1e9, 2),
    }

```

---
## Time and Timezone Helpers

These functions work together to provide accurate time information for locations
worldwide. We maintain a mapping of common locations to their timezone identifiers,
then use Python's pytz library to calculate the current time.


```python

TIMEZONE_MAP = {
    # Countries
    "india": "Asia/Kolkata",
    "usa": "America/New_York",
    "uk": "Europe/London",
    "japan": "Asia/Tokyo",
    "china": "Asia/Shanghai",
    "australia": "Australia/Sydney",
    "france": "Europe/Paris",
    "germany": "Europe/Berlin",
    "russia": "Europe/Moscow",
    "brazil": "America/Sao_Paulo",
    "canada": "America/Toronto",
    "mexico": "America/Mexico_City",
    "spain": "Europe/Madrid",
    "italy": "Europe/Rome",
    # US States & Major Cities
    "california": "America/Los_Angeles",
    "new york": "America/New_York",
    "texas": "America/Chicago",
    "florida": "America/New_York",
    "washington": "America/Los_Angeles",
    "oregon": "America/Los_Angeles",
    "nevada": "America/Los_Angeles",
    "arizona": "America/Phoenix",
    "colorado": "America/Denver",
    "utah": "America/Denver",
    "illinois": "America/Chicago",
    "michigan": "America/Detroit",
    "massachusetts": "America/New_York",
    "pennsylvania": "America/New_York",
    # Major World Cities
    "los angeles": "America/Los_Angeles",
    "san francisco": "America/Los_Angeles",
    "seattle": "America/Los_Angeles",
    "chicago": "America/Chicago",
    "boston": "America/New_York",
    "miami": "America/New_York",
    "london": "Europe/London",
    "paris": "Europe/Paris",
    "tokyo": "Asia/Tokyo",
    "beijing": "Asia/Shanghai",
    "shanghai": "Asia/Shanghai",
    "dubai": "Asia/Dubai",
    "singapore": "Asia/Singapore",
    "hong kong": "Asia/Hong_Kong",
    "seoul": "Asia/Seoul",
    "moscow": "Europe/Moscow",
    "sydney": "Australia/Sydney",
    "mumbai": "Asia/Kolkata",
    "delhi": "Asia/Kolkata",
    "berlin": "Europe/Berlin",
    "rome": "Europe/Rome",
    "madrid": "Europe/Madrid",
    "toronto": "America/Toronto",
    "vancouver": "America/Vancouver",
    "montreal": "America/Toronto",
    # Special
    "utc": "UTC",
    "gmt": "GMT",
}


def get_timezone_for_location(location):
    """Map location to timezone."""

    loc = location.lower().strip()
    return TIMEZONE_MAP.get(loc)

```

---
## World Time Tool

This function gets the current time for a specific location. It uses the
helper function `get_timezone_for_location` to determine the correct timezone
and then returns the formatted time, date, and day.


```python

def get_current_time(time_location=None):
    """Get current time for a location."""

    if not time_location:
        time_location = "UTC"

    try:
        timezone = get_timezone_for_location(time_location)
        note = ""

        if not timezone:
            # Default to UTC if location not found
            timezone = "UTC"
            note = (
                f"Location '{time_location}' not found in database. Showing UTC time."
            )

        tz = pytz.timezone(timezone)
        now = datetime.datetime.now(tz)
        return {
            "location": time_location if not note else "UTC",
            "time": now.strftime("%H:%M:%S"),
            "date": now.strftime("%Y-%m-%d"),
            "day": now.strftime("%A"),
            "timezone": timezone,
            "note": note,
        }
    except Exception as e:
        return {"error": str(e)}

```

---
## Tool Definitions (Function Calling Format)

Now we need to tell the model about these tools. We do this by defining tool
schemas in a format that the model can understand. Each tool definition includes:

- Name: The function name
- Description: What the function does (helps the model decide when to use it)
- Parameters: The arguments the function accepts

This format follows the standard function calling convention used by FunctionGemma.


```python
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a given ticker symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., AAPL, GOOGL, TSLA)",
                    }
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current clock time for a specific city or country.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time_location": {
                        "type": "string",
                        "description": "The city or country for which to get the time, e.g., 'Tokyo' or 'Japan'",
                    }
                },
                "required": ["time_location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_stats",
            "description": "Get current CPU and memory usage statistics",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]
```

---
## Tool Registry

This dictionary maps tool names (as strings) to their actual Python function
implementations. When the model generates a function call like `get_stock_price`,
we use this registry to look up and execute the corresponding Python function.

This pattern allows for dynamic tool execution - we can easily add or remove tools
by updating both `TOOL_DEFINITIONS` and this registry.


```python
# Tool registry
TOOLS = {
    "web_search": web_search,
    "get_stock_price": get_stock_price,
    "get_current_time": get_current_time,
    "get_system_stats": get_system_stats,
}
```

---
## Helper Functions For Function Calling

The following functions handle the mechanics of function calling:

-  Parsing function calls from model output
-  Formatting tool declarations for the prompt
-  Constructing the full conversation prompt
-  Formatting tool results for display

---
## Function Call Parser

This function is responsible for extracting the structured function call from
the model's text output. It looks for the special `<start_function_call>` tags
and parses the function name and arguments into a Python dictionary.


```python

FUNCTION_CALL_PATTERN = re.compile(
    r"<start_function_call>call:(\w+)\{([^}]*)\}<end_function_call>"
)
ARGUMENT_PATTERN = re.compile(
    r'(\w+):\s*("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|[^,]+)'
)


def parse_function_call(output):
    """Parse function call from FunctionGemma model output.

    Extracts function name and arguments from the special FunctionGemma function calling
    format: <start_function_call>call:tool_name{arg1:value1}<end_function_call>

    Args:
        output: Raw model output string

    Returns:
        Dictionary with "name" (function name) and "arguments" (dict of args),

    """
    match = FUNCTION_CALL_PATTERN.search(output)

    if not match:
        return None

    tool_name = match.group(1)
    args_str = match.group(2).strip()

    # Parse arguments
    args = {}
    if args_str:
        # Matches key:"value" or key:'value' or key:value
        arg_pairs = ARGUMENT_PATTERN.findall(args_str)
        for key, value in arg_pairs:
            value = value.strip()

            # Remove <escape> tags
            value = value.replace("<escape>", "").replace("</escape>", "")

            # Remove quotes if present
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]

            args[key] = value

    return {"name": tool_name, "arguments": args}

```

---
## Tool Declaration Formatter

This function converts our Python tool definitions into the specific format
expected by `FunctionGemma`. It handles the mapping of types and descriptions to the
`declaration:..` string format used in the system prompt.


```python

def format_tool_declaration(tool):
    """Format a tool declaration for FunctionGemma function calling."""

    func = tool["function"]
    params = func.get("parameters", {})
    props = params.get("properties", {})
    required = params.get("required", [])

    # Build parameter string
    param_parts = []
    for name, details in props.items():
        param_str = f"{name}:{{description:<escape>{details.get('description', '')}<escape>,type:<escape>{details['type'].upper()}<escape>}}"
        param_parts.append(param_str)

    properties_str = ",".join(param_parts) if param_parts else ""
    required_str = ",".join(f"<escape>{r}<escape>" for r in required)

    # Build full declaration
    declaration = f"declaration:{func['name']}"
    declaration += f"{{description:<escape>{func['description']}<escape>"

    if properties_str or required_str:
        declaration += ",parameters:{"
        if properties_str:
            declaration += f"properties:{{{properties_str}}}"
        if required_str:
            if properties_str:
                declaration += ","
            declaration += f"required:[{required_str}]"
        declaration += ",type:<escape>OBJECT<escape>}"

    declaration += "}"

    return declaration

```

---
## Prompt Constructor

This is the core function for building the prompt. It combines:

-  The system prompt with tool declarations
-  Few-shot examples to guide the model
-  The conversation history (user and assistant messages)
-  The final generation prompt


```python

def format_prompt_with_tools(messages, tools):
    """Manually construct FunctionGemma function calling prompt.

    This function replicates the behavior of HuggingFace's apply_chat_template()
    for FunctionGemma native function calling format, since KerasHub doesn't yet
    provide this functionality.

    Args:
        messages: List of message dicts with "role" and "content" keys
        tools: List of tool definition dicts in OpenAI format

    Returns:
        Formatted prompt string with tool declarations and conversation history
    """
    prompt = "<bos>"

    # Add tool declarations
    if tools:
        prompt += "<start_of_turn>developer\n"
        for tool in tools:
            prompt += "<start_function_declaration>"
            prompt += format_tool_declaration(tool)
            prompt += "<end_function_declaration>"

        # Add few-shot examples to help the model distinguish between tools
        prompt += "\nHere are some examples of how to use the tools correctly:\n"
        prompt += "User: Who is Isaac Newton?\n"
        prompt += "Model: <start_function_call>call:web_search{query:Isaac Newton}<end_function_call>\n"
        prompt += "User: What is the time in USA?\n"
        prompt += "Model: <start_function_call>call:get_current_time{time_location:USA}<end_function_call>\n"
        prompt += "User: Stock price of Apple\n"
        prompt += "Model: <start_function_call>call:get_stock_price{symbol:AAPL}<end_function_call>\n"
        prompt += "User: Who is the PM of Japan?\n"
        prompt += "Model: <start_function_call>call:web_search{query:PM of Japan}<end_function_call>\n"
        prompt += "User: Check the weather in London\n"
        prompt += "Model: <start_function_call>call:web_search{query:weather in London}<end_function_call>\n"
        prompt += "User: Latest news about AI\n"
        prompt += "Model: <start_function_call>call:web_search{query:Latest news about AI}<end_function_call>\n"
        prompt += "User: System memory usage\n"
        prompt += (
            "Model: <start_function_call>call:get_system_stats{}<end_function_call>\n"
        )

        prompt += "<end_of_turn>\n"

    # Add conversation messages
    for message in messages:
        role = message["role"]
        if role == "assistant":
            role = "model"

        content = message.get("content", "")

        # Regular user/assistant messages
        prompt += f"<start_of_turn>{role}\n"
        prompt += content
        prompt += "<end_of_turn>\n"

    # Add generation prompt
    prompt += "<start_of_turn>model\n"

    return prompt

```

---
## Tool Response Formatter

After a tool is executed, this function formats the result back into a string
that can be displayed to the user or fed back into the model. It handles
specific formatting for different tools (like adding currency to stock prices).


```python

def format_tool_response(tool_name, result):
    """Format tool result for conversation."""
    if "error" in result:
        return f"Error: {result['error']}"

    # Handle Web Search Results (from web_search tool OR fallback from other tools)
    if "results" in result:
        results = result.get("results", [])
        if results:
            output = f"Found {len(results)} results:\n"
            for i, r in enumerate(results[:3], 1):
                output += f"\n{i}. {r['title']}\n   {r['description'][:150]}...\n   {r['link']}"
            return output
        return "No results found"

    if tool_name == "get_stock_price":
        return f"Stock {result['symbol']}: ${result['price']} {result['currency']} (as of {result['date']})"

    elif tool_name == "get_current_time":
        response = f"Time in {result['location']}: {result['time']} ({result['day']}, {result['date']})\nTimezone: {result['timezone']}"
        if result.get("note"):
            response += f"\nNote: {result['note']}"
        return response

    elif tool_name == "get_system_stats":
        return f"CPU: {result['cpu_percent']}% | Memory: {result['mem_percent']}% ({result['used_gb']}/{result['total_gb']}GB)"

    return str(result)

```

# Interactive Chat Loop

This section implements the main conversation loop. The agent:

-  Formats the prompt with tool declarations
-  Generates a response (which may include a function call)
-  Executes the function if called
-  Returns the result to the user

After each successful tool execution, we clear the conversation history to
prevent token overflow and keep the model focused.

---
## Model Loading

This function loads the FunctionGemma, a specialized version of our Gemma 3 270M model tuned for function calling using KerasHub's `from_preset` method.
You can load it from Kaggle using the preset name `(e.g. "function_gemma_instruct_270m")`
or from a local path if you've downloaded the model to your machine.

Update the path to match your model location. The model is loaded once at
startup and then used throughout the chat session.


```python

def load_model():
    """Load FunctionGemma 270M model."""
    print("Loading FunctionGemma 270M model...")
    model = keras_hub.models.Gemma3CausalLM.from_preset("function_gemma_instruct_270m")
    print("Model loaded!\n")
    return model

```

This is the main function that runs the interactive conversation with the user.

It handles the complete cycle of:

-  Displaying capabilities and waiting for user input
-  Formatting the prompt with tool declarations and conversation history
-  Generating a response from the model
-  Parsing any function calls from the response
-  Executing the requested tools
-  Formatting and displaying results to the user
-  Managing conversation history to prevent token overflow

The loop continues until the user types 'exit', 'quit', or 'q'.


```python

def chat(model):
    """Main chat loop with native function calling."""
    print("=" * 70)
    print(" " * 20 + "FUNCTION CALLING")
    print("=" * 70)
    print("\nCapabilities:")
    print("  🔍 Web Search        - e.g., 'Who is Newton?', 'Latest AI news'")
    print("  📈 Stock Prices      - e.g., 'Stock price of AAPL', 'TSLA price'")
    print(
        "  🕐 World Time        - e.g., 'Time in London', 'What time is it in Tokyo?'"
    )
    print("  💻 System Stats      - e.g., 'System memory', 'CPU usage'")
    print("\nType 'exit' to quit.\n")
    print("-" * 70)

    # Conversation history
    messages = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            break

        if user_input.lower() in {"exit", "quit", "q"}:
            print("\nGoodbye! 👋")
            break

        if not user_input:
            continue

        # Add user message
        messages.append({"role": "user", "content": user_input})

        # Generate response with tools
        print("🤔 Thinking...", end="", flush=True)

        try:
            # Manually format prompt with tools (KerasHub doesn't have apply_chat_template)
            prompt = format_prompt_with_tools(messages, TOOL_DEFINITIONS)

            # Generate
            output = model.generate(prompt, max_length=2048)

            # Extract only the new response
            response = output[len(prompt) :].strip()
            if "<start_function_response>" in response:
                response = response.split("<start_function_response>")[0].strip()
            if "<start_of_turn>" in response:
                response = response.split("<start_of_turn>")[0].strip()
            # Remove end tokens
            response = response.replace("<end_of_turn>", "").strip()

            print(f"\r{' ' * 40}\r", end="")  # Clear status
            print(f"[DEBUG] Raw Model Response: {response}")

            # Check if model made a function call
            function_call = parse_function_call(response)

            if function_call:
                tool_name = function_call["name"]
                tool_args = function_call["arguments"]

                print(f"🔧 Calling {tool_name}...", end="", flush=True)

                # Execute tool
                if tool_name in TOOLS:
                    try:
                        result = TOOLS[tool_name](**tool_args)
                        formatted_result = format_tool_response(tool_name, result)

                        print(f"\r{' ' * 40}\r", end="")  # Clear status
                        print(f"\nAssistant: {formatted_result}")

                        # Clear conversation history after successful tool execution
                        # This prevents token overflow and keeps the model focused
                        messages = []

                    except Exception as e:
                        print(f"\r❌ Error executing {tool_name}: {e}")
                        messages.pop()  # Remove user message on error
                else:
                    print(f"\r❌ Unknown tool: {tool_name}")
                    messages.pop()
            else:
                # Regular text response (no function call)
                # Check if response is empty
                if not response or response == "...":
                    print(
                        f"\nAssistant: I'm not sure how to help with that. Please try rephrasing your question."
                    )
                    messages.pop()  # Remove problematic message
                else:
                    print(f"\nAssistant: {response}")
                    messages.append({"role": "assistant", "content": response})

                    # Keep only last 3 turns to prevent context overflow
                    if len(messages) > 6:  # 3 user + 3 assistant
                        messages = messages[-6:]

        except Exception as e:
            print(f"\r❌ Error: {e}")
            messages.pop()  # Remove user message on error
            continue

```

This is where the program starts execution. When you run this script:

-  The FunctionGemma model is loaded from the specified path
-  The interactive chat loop begins, waiting for user input
-  The Assistant processes queries and executes tools until the user types 'exit'


```python
 if __name__ == "__main__":
     model = load_model()
     chat(model)
```

# Conclusion

---
## What You've Achieved

By following this guide, you've learned how to build a production-ready AI Assistant with
native function calling capabilities using FunctionGemma in KerasHub. Here's what you've accomplished:

### Native Function Calling Implementation:

- Implemented FunctionGemma native function calling format without relying on external APIs
- Learned to construct proper tool declarations and parse structured function calls
- Built a manual chat template that replicates HuggingFace's apply_chat_template()

### Multi-Tool Assistant Architecture:

- Created four distinct tools: web search, stock prices, world time, and system stats
- Designed tools with clear error handling and user-friendly responses
- Implemented explicit error messages.

### Prompt Engineering Best Practices:

- Used few-shot examples to guide model behavior and improve tool selection
- Structured prompts with proper role separation (developer, user, model)
- Managed conversation history to prevent token overflow

### Error Handling and User Experience:

- Implemented graceful degradation (e.g., UTC fallback for unknown timezones)
- Provided clear, actionable error messages for invalid inputs
- Added contextual information (dates, notes) to make responses more informative

### Real-World Applications:

This pattern can be extended to build:

- Customer service chatbots with database access
- Data analysis assistants with computation tools
- Personal productivity Assistants with calendar and email integration
- Domain-specific assistants (medical, legal, financial) with specialized tools

---
## Key Takeaways

- **Small models can be powerful**: Even a 270M parameter model can reliably use tools
  when given proper guidance through few-shot examples and clear tool descriptions.

- **Transparency matters**: Explicit error handling and clear responses build user trust
  more effectively.

- **Function calling is composable**: The same pattern works for any number of tools,
  making it easy to extend your Assistant's capabilities.

---
## Next Steps

To further improve this agent, consider:

- Fine-tuning the model on domain-specific tool usage examples to optimize performance.
- Adding authentication and rate limiting for production deployment.
- Implementing tool result caching to reduce API calls.
- Creating a web interface using Gradio or Streamlit.
- Adding more sophisticated error recovery strategies.

Happy building! 🚀
