"""
Title: Model Context Protocol (MCP) with KerasHub Models
Author: [Laxmareddypatlolla](https://github.com/laxmareddypatlolla),[Divyashree Sreepathihalli](https://github.com/divyashreepathihalli)
Date created: 2025/08/16
Last modified: 2025/08/16
Description: A guide to building MCP systems using KerasHub models for intelligent tool calling.
Accelerator: GPU
"""

"""
## Introduction

**View in Colab** • **GitHub source**

## Welcome to Your MCP Adventure! 🚀

Hey there! Ready to dive into something really exciting? We're about to build a system that can make AI models actually "do things" in the real world - not just chat, but actually execute functions, call APIs, and interact with external tools!

**What makes this special?** Instead of just having a conversation with an AI, we're building something that's like having a super-smart assistant who can actually take action on your behalf. It's like the difference between talking to someone about cooking versus having them actually cook dinner for you!

**What we're going to discover together:**

* How to make AI models work with external tools and functions
* Why MCP (Model Context Protocol) is the future of AI interaction
* How to build systems that are both intelligent AND actionable
* The magic of combining language understanding with tool execution

Think of this as your journey into the future of AI-powered automation. By the end, you'll have built something that could potentially revolutionize how we interact with AI systems!

Ready to start this adventure? Let's go!

## Understanding the Magic Behind MCP! ✨

Alright, let's take a moment to understand what makes MCP so special! Think of MCP as having a super-smart assistant who doesn't just answer questions, but actually knows how to use tools to get things done.

**The Three Musketeers of MCP:**

1. **The Language Model** 🧠: This is like having a brilliant conversationalist who can understand what you want and figure out what tools might help
2. **The Tool Registry** 🛠️: This is like having a well-organized toolbox where every tool has a clear purpose and instructions
3. **The Execution Engine** ⚡: This is like having a skilled worker who can actually use the tools to accomplish tasks

**Here's what our amazing MCP system will do:**

* **Step 1:** Our Gemma3 model will understand your request and determine if it needs a tool
* **Step 2:** It will identify the right tool from our registry (weather, calculator, search, etc.)
* **Step 3:** It will format the tool call with the correct parameters
* **Step 4:** Our system will execute the tool and get real results
* **Step 5:** We'll present you with actionable information instead of just text

**Why this is revolutionary:** Instead of the AI just telling you what it knows, it's actually doing things for you! It's like the difference between a librarian who tells you where to find a book versus one who actually goes and gets the book for you!

Ready to see this magic in action? Let's start building! 🎯

## Setting Up Our AI Workshop 🛠️

Alright, before we start building our amazing MCP system, we need to set up our digital workshop! Think of this like gathering all the tools a master craftsman needs before creating a masterpiece.

**What we're doing here:** We're importing all the powerful libraries that will help us build our MCP system. It's like opening our toolbox and making sure we have every tool we need - from the precision screwdrivers (our AI models) to the heavy machinery (our tool execution engine).

**Why KerasHub?** We're using KerasHub because it's like having access to a massive library of pre-trained AI models. Instead of training models from scratch (which would take forever), we can grab models that are already experts at understanding language and generating responses. It's like having a team of specialists ready to work for us!

**The magic of MCP:** This is where things get really exciting! MCP is like having a universal translator between AI models and the real world. It allows our AI to not just think, but to act!

Let's get our tools ready and start building something amazing!

"""

import os
import re
import json
from typing import Dict, List, Any, Callable, Optional

# Set Keras backend to jax for optimal performance
os.environ["KERAS_BACKEND"] = "jax"

import keras
from keras import layers
import keras_hub

"""
## Loading Our AI Dream Team! 🤖

Alright, this is where the real magic begins! We're about to load up our AI model - think of this as assembling the ultimate specialist with the superpower of understanding and responding to human requests!

**What we're doing here:** We're loading the Gemma3 Instruct 1B model from KerasHub. This model is like having a brilliant conversationalist who can understand complex requests and figure out when to use tools versus when to respond directly.

**Why Gemma3?** This model is specifically designed for instruction-following and tool usage. It's like having an AI that's been trained to be helpful and actionable, not just chatty!

**The magic of KerasHub:** Instead of downloading and setting up complex model files, we just call `keras_hub.load()` and KerasHub handles all the heavy lifting for us. It's like having a personal assistant who sets up your entire workspace!

"""


def _load_model():
    """
    Load the Gemma3 Instruct 1B model from KerasHub.

    This is the "brain" of our system - the AI model that understands
    user requests and decides when to use tools.

    Returns:
        The loaded Gemma3 model ready for text generation
    """
    print("🚀 Loading Gemma3 Instruct 1B model...")
    model = keras_hub.models.Gemma3CausalLM.from_preset("gemma3_instruct_1b")
    print(f"✅ Model loaded successfully: {model.name}")
    return model


"""
## Building Our Tool Arsenal! 🛠️

Now we're getting to the really fun part! We're building our collection of tools that our AI can use to actually accomplish tasks. Think of this as creating a Swiss Army knife for your AI assistant!

**What we're building here:**

We're creating three essential tools that demonstrate different types of capabilities:
1. **Weather Tool** - Shows how to work with external data and APIs
2. **Calculator Tool** - Shows how to handle mathematical computations
3. **Search Tool** - Shows how to provide information retrieval

**Why these tools?** Each tool represents a different category of AI capabilities:
- **Data Access** (weather) - Getting real-time information
- **Computation** (calculator) - Processing and analyzing data
- **Knowledge Retrieval** (search) - Finding and organizing information

**The magic of tool design:** Each tool is designed to be simple, reliable, and focused. It's like building with LEGO blocks - each piece has a specific purpose, and together they create something amazing!

Let's build our tools and see how they work!

"""


def weather_tool(city: str) -> str:
    """
    Get weather information for a specific city.

    This tool demonstrates how MCP can access external data sources.
    In a real-world scenario, this would connect to a weather API.

    Args:
        city: The name of the city to get weather for

    Returns:
        A formatted weather report for the city
    """
    # Simulated weather data - in production, this would call a real API
    weather_data = {
        "Tokyo": "75°F, Rainy, Humidity: 82%",
        "New York": "65°F, Partly Cloudy, Humidity: 70%",
        "London": "55°F, Cloudy, Humidity: 85%",
        "Paris": "68°F, Sunny, Humidity: 65%",
        "Sydney": "72°F, Clear, Humidity: 60%",
    }

    city_normalized = city.title()
    if city_normalized in weather_data:
        return weather_data[city_normalized]
    else:
        return f"Weather data not available for {city_normalized}"


def calculator_tool(expression: str) -> str:
    """
    Calculate mathematical expressions safely.

    This tool demonstrates how MCP can handle computational tasks.
    It safely evaluates mathematical expressions while preventing code injection.

    Args:
        expression: A mathematical expression as a string (e.g., "15 + 7 - 24")

    Returns:
        The calculated result as a string
    """
    try:
        # Clean the expression to only allow safe mathematical operations
        cleaned_expr = re.sub(r"[^0-9+\-*/().\s]", "", expression)

        # Evaluate the expression safely
        result = eval(cleaned_expr)

        # Format the result nicely
        if isinstance(result, (int, float)):
            if result == int(result):
                return str(int(result))
            else:
                return f"{result:.2f}"
        else:
            return str(result)
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


def search_tool(query: str) -> str:
    """
    Search for information based on a query.

    This tool demonstrates how MCP can provide information retrieval.
    In a real-world scenario, this would connect to search engines or databases.

    Args:
        query: The search query string

    Returns:
        Relevant information based on the query
    """
    # Simulated search results - in production, this would call real search APIs
    search_responses = {
        "machine learning": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It's used in recommendation systems, image recognition, natural language processing, and many other applications.",
        "python": "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in data science, web development, machine learning, and automation.",
        "artificial intelligence": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines. It encompasses machine learning, natural language processing, computer vision, and robotics.",
        "data science": "Data science combines statistics, programming, and domain expertise to extract meaningful insights from data. It involves data collection, cleaning, analysis, and visualization.",
    }

    query_lower = query.lower()
    for key, response in search_responses.items():
        if key in query_lower:
            return response

    return f"Search results for '{query}': Information not available in our current knowledge base."


"""
## Creating Our Tool Management System! 🏗️

Now we're building the backbone of our MCP system - the tool registry and management system. Think of this as creating the control center that coordinates all our tools!

**What we're building here:** We're creating a system that:

1. **Registers tools** - Keeps track of what tools are available
2. **Manages tool metadata** - Stores descriptions and parameter information
3. **Executes tools safely** - Runs tools with proper error handling
4. **Provides tool information** - Gives the AI model context about available tools

**Why this architecture?** This design separates concerns beautifully:

- **Tool Registry** handles tool management
- **MCP Client** handles AI interaction
- **Individual Tools** handle specific functionality

**The magic of separation of concerns:** Each component has a single responsibility, making the system easy to understand, debug, and extend. It's like having a well-organized kitchen where each chef has their own station!

Let's build our tool management system!

"""


class MCPTool:
    """
    Represents a tool that can be called by the MCP system.

    This class encapsulates all the information needed to use a tool:
    - What the tool does (description)
    - What parameters it needs (function signature)
    - How to execute it (the actual function)

    Think of this as creating a detailed instruction manual for each tool!
    """

    def __init__(self, name: str, description: str, function: Callable):
        """
        Initialize a new MCP tool.

        Args:
            name: The name of the tool (e.g., "weather", "calculator")
            description: What the tool does (used by the AI to decide when to use it)
            function: The actual function that implements the tool's functionality
        """
        self.name = name
        self.description = description
        self.function = function

    def execute(self, **kwargs) -> str:
        """
        Execute the tool with the given parameters.

        Args:
            **kwargs: The parameters to pass to the tool function

        Returns:
            The result of executing the tool
        """
        try:
            return self.function(**kwargs)
        except Exception as e:
            return f"Error executing {self.name}: {str(e)}"


"""
## The Command Center: MCPToolRegistry 🎯

Now we're building the heart of our tool management system - the MCPToolRegistry! Think of this as creating the mission control center for all our AI tools.

**What this class does:** The MCPToolRegistry is like having a brilliant project manager who:

- **Keeps an organized inventory** of all available tools
- **Provides instant access** to tool information when the AI needs it
- **Coordinates tool execution** with proper error handling
- **Maintains tool metadata** so the AI knows what each tool can do

**Why this is crucial:** Without a tool registry, our AI would be like a chef without a kitchen - it might know what to cook, but it wouldn't know what tools are available or how to use them. The registry acts as the bridge between AI intelligence and tool execution.

**The magic of centralization:** By having all tools registered in one place, we can:

- Easily add new tools without changing the core system
- Provide the AI with a complete overview of available capabilities
- Handle errors consistently across all tools
- Scale the system by simply registering more tools

Think of this as the control tower at an airport - it doesn't fly the planes itself, but it coordinates everything so all flights can take off and land safely!

"""


class MCPToolRegistry:
    """
    Manages the collection of available tools in our MCP system.

    This class acts as a central registry that:
    - Keeps track of all available tools
    - Provides information about tools to the AI model
    - Executes tools when requested
    - Handles errors gracefully

    Think of this as the command center that coordinates all our tools!
    """

    def __init__(self):
        """Initialize an empty tool registry."""
        self.tools = {}

    def register_tool(self, tool: MCPTool):
        """
        Register a new tool in the registry.

        Args:
            tool: The MCPTool instance to register
        """
        self.tools[tool.name] = tool
        print(f"✅ Registered tool: {tool.name}")

    def get_tools_list(self) -> str:
        """
        Get a formatted list of all available tools.

        This creates a description that the AI model can use to understand
        what tools are available and when to use them.

        Returns:
            A formatted string describing all available tools
        """
        tools_list = []
        for name, tool in self.tools.items():
            tools_list.append(f"{name}: {tool.description}")
        return "\n".join(tools_list)

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Execute a specific tool with the given arguments.

        Args:
            tool_name: The name of the tool to execute
            arguments: The arguments to pass to the tool

        Returns:
            The result of executing the tool

        Raises:
            ValueError: If the tool is not found
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found")

        tool = self.tools[tool_name]
        return tool.execute(**arguments)


"""
## Building Our AI Communication Bridge! 🌉

Now we're creating the heart of our MCP system - the client that bridges the gap between our AI model and our tools. Think of this as building a translator that can understand both human language and machine instructions!

**What we're building here:** We're creating a system that:
1. **Understands user requests** - Processes natural language input
2. **Generates appropriate prompts** - Creates context for the AI model
3. **Parses AI responses** - Extracts tool calls from the model's output
4. **Executes tools** - Runs the requested tools and gets results
5. **Provides responses** - Gives users actionable information

**Why this architecture?** This design creates a clean separation between:

- **AI Understanding** (the model's job)
- **Tool Execution** (our system's job)
- **Response Generation** (combining AI insights with tool results)

**The magic of the bridge pattern:** It allows our AI model to focus on what it does best (understanding language) while our system handles what it does best (executing tools). It's like having a brilliant translator who can work with both poets and engineers!

Let's build our AI communication bridge!

"""


class MCPClient:
    """
    The main client that handles communication between users, AI models, and tools.

    This class orchestrates the entire MCP workflow:
    1. Takes user input and creates appropriate prompts
    2. Sends prompts to the AI model
    3. Parses the model's response for tool calls
    4. Executes requested tools
    5. Returns results to the user

    Think of this as the conductor of an orchestra, making sure everyone plays their part!
    """

    def __init__(self, model, tool_registry: MCPToolRegistry):
        """
        Initialize the MCP client.

        Args:
            model: The KerasHub model to use for understanding requests
            tool_registry: The registry of available tools
        """
        self.model = model
        self.tool_registry = tool_registry

    def _build_prompt(self, user_input: str) -> str:
        """
        Build a prompt for the AI model that includes available tools.

        This method creates the context that helps the AI model understand:
        - What tools are available
        - When to use them
        - How to format tool calls

        Args:
            user_input: The user's request

        Returns:
            A formatted prompt for the AI model
        """
        tools_list = self.tool_registry.get_tools_list()

        # Ultra-simple prompt - just the essentials
        # This minimal approach has proven most effective for encouraging tool calls
        prompt = f"""Available tools:
{tools_list}

User: {user_input}
Assistant:"""
        return prompt

    def _extract_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from the AI model's response.

        This method uses flexible parsing to handle various formats the model might generate:
        - TOOL_CALL: {...} format
        - {"tool": "name", "arguments": {...}} format
        - ```tool_code function_name(...) ``` format

        Args:
            response: The raw response from the AI model

        Returns:
            A list of parsed tool calls
        """
        tool_calls = []

        # Look for TOOL_CALL blocks with strict JSON parsing
        pattern = r"TOOL_CALL:\s*\n(\{[^}]*\})"
        matches = re.findall(pattern, response, re.DOTALL)
        for match in matches:
            try:
                json_str = match.strip()
                tool_call = json.loads(json_str)
                if "name" in tool_call and "arguments" in tool_call:
                    tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue

        # If no TOOL_CALL format found, try to parse the format the model is actually generating
        if not tool_calls:
            tool_calls = self._parse_model_tool_format(response)

        return tool_calls

    def _parse_model_tool_format(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse the format the model is actually generating: {"tool": "tool_name", "arguments": {...}}

        This method handles the JSON format that our model tends to generate,
        converting it to our standard tool call format.

        Args:
            response: The raw response from the AI model

        Returns:
            A list of parsed tool calls
        """
        tool_calls = []
        pattern = r'\{[^}]*"tool"[^}]*"arguments"[^}]*\}'
        matches = re.findall(pattern, response, re.DOTALL)

        for match in matches:
            try:
                tool_call = json.loads(match)
                if "tool" in tool_call and "arguments" in tool_call:
                    converted_call = {
                        "name": tool_call["tool"],
                        "arguments": tool_call["arguments"],
                    }
                    tool_calls.append(converted_call)
            except json.JSONDecodeError:
                continue

        # If still no tool calls found, try to parse tool_code blocks
        if not tool_calls:
            tool_calls = self._parse_tool_code_blocks(response)

        return tool_calls

    def _parse_tool_code_blocks(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse tool_code blocks that the model is generating.

        This method handles the ```tool_code function_name(...) ``` format
        that our model sometimes generates, converting it to our standard format.

        Args:
            response: The raw response from the AI model

        Returns:
            A list of parsed tool calls
        """
        tool_calls = []
        pattern = r"```tool_code\s*\n([^`]+)\n```"
        matches = re.findall(pattern, response, re.DOTALL)

        for match in matches:
            try:
                tool_call = self._parse_tool_code_call(match.strip())
                if tool_call:
                    tool_calls.append(tool_call)
            except Exception:
                continue

        return tool_calls

    def _parse_tool_code_call(self, tool_code: str) -> Dict[str, Any]:
        """
        Parse a tool_code call into a tool call structure.

        This method converts the function-call format into our standard
        tool call format with name and arguments.

        Args:
            tool_code: The tool code string (e.g., "weather.get_weather(city='Tokyo')")

        Returns:
            A parsed tool call dictionary or None if parsing fails
        """
        if "weather.get_weather" in tool_code:
            city_match = re.search(r'city="([^"]+)"', tool_code)
            if city_match:
                return {"name": "weather", "arguments": {"city": city_match.group(1)}}
        elif "calculator.add" in tool_code:
            numbers = re.findall(r"[-]?\d+", tool_code)
            if numbers:
                expression = " + ".join(numbers)
                return {"name": "calculator", "arguments": {"expression": expression}}
        elif "search." in tool_code:
            query_match = re.search(r'query="([^"]+)"', tool_code)
            if query_match:
                return {"name": "search", "arguments": {"query": query_match.group(1)}}

        return None

    def chat(self, user_input: str) -> str:
        """
        Process a user request and return a response.

        This is the main method that orchestrates the entire MCP workflow:
        1. Builds a prompt with available tools
        2. Gets a response from the AI model
        3. Extracts any tool calls from the response
        4. Executes tools and gets results
        5. Returns a formatted response to the user

        Args:
            user_input: The user's request

        Returns:
            A response that may include tool results or direct AI responses
        """
        # Build the prompt with available tools
        prompt = self._build_prompt(user_input)

        # Get response from the AI model
        response = self.model.generate(prompt, max_length=512)

        # Extract tool calls from the response
        tool_calls = self._extract_tool_calls(response)

        if tool_calls:
            # Safety check: if multiple tool calls found, execute only the first one
            if len(tool_calls) > 1:
                print(
                    f"⚠️ Multiple tool calls found, executing only the first one: {tool_calls[0]['name']}"
                )
                tool_calls = [tool_calls[0]]  # Keep only the first one

            # Execute tools with deduplication
            results = []
            seen_tools = set()

            for tool_call in tool_calls:
                tool_key = f"{tool_call['name']}_{str(tool_call['arguments'])}"
                if tool_key not in seen_tools:
                    seen_tools.add(tool_key)
                    try:
                        result = self.tool_registry.execute_tool(
                            tool_call["name"], tool_call["arguments"]
                        )
                        results.append(f"{tool_call['name']}: {result}")
                    except Exception as e:
                        results.append(f"Error in {tool_call['name']}: {str(e)}")

            # Format the final response
            if len(results) == 1:
                final_response = results[0]
            else:
                final_response = f"Here's what I found:\n\n" + "\n\n".join(results)

            return final_response
        else:
            # No tool calls found, use the model's response directly
            print("ℹ️ No tool calls found, using model response directly")
            return response


"""
## Assembling Our MCP System! 🔧

Now we're putting all the pieces together! Think of this as the moment when all the individual components come together to create something greater than the sum of its parts.

**What we're doing here:** We're creating the main function that:
1. **Sets up our tool registry** - Registers all available tools
2. **Loads our AI model** - Gets our language model ready
3. **Creates our MCP client** - Connects everything together
4. **Demonstrates the system** - Shows how everything works in action

**Why this structure?** This design creates a clean, modular system where:
- **Tool registration** is separate from tool execution
- **Model loading** is separate from client creation
- **Demonstration** is separate from system setup

**The magic of modular design:** Each piece can be developed, tested, and improved independently. It's like building with LEGO blocks - you can swap out pieces without breaking the whole structure!

Let's assemble our MCP system and see it in action!

"""


def _register_tools(tool_registry: MCPToolRegistry):
    """
    Register all available tools in the tool registry.

    This function creates and registers our three main tools:
    - Weather tool for getting weather information
    - Calculator tool for mathematical computations
    - Search tool for information retrieval

    Args:
        tool_registry: The MCPToolRegistry instance to register tools with
    """
    # Create and register the weather tool
    weather_tool_instance = MCPTool(
        name="weather", description="Get weather for a city", function=weather_tool
    )
    tool_registry.register_tool(weather_tool_instance)

    # Create and register the calculator tool
    calculator_tool_instance = MCPTool(
        name="calculator",
        description="Calculate math expressions",
        function=calculator_tool,
    )
    tool_registry.register_tool(calculator_tool_instance)

    # Create and register the search tool
    search_tool_instance = MCPTool(
        name="search", description="Search for information", function=search_tool
    )
    tool_registry.register_tool(search_tool_instance)


"""
## Complete MCP Demonstration

This function orchestrates the entire MCP system demonstration:

1. **Sets up the tool registry** - Registers all available tools (weather, calculator, search)
2. **Loads the AI model** - Gets the Gemma3 Instruct 1B model ready
3. **Creates the MCP client** - Connects everything together
4. **Runs demonstration examples** - Shows weather, calculator, and search in action
5. **Demonstrates the system** - Proves MCP works with real tool execution

Think of this as the grand finale where all the components come together to create something amazing!
"""


def main():
    print("🎯 Simple MCP with KerasHub - Working Implementation")
    print("=" * 70)

    # Set up our tool registry
    tool_registry = MCPToolRegistry()
    _register_tools(tool_registry)

    # Load our AI model
    model = _load_model()

    # Create our MCP client
    client = MCPClient(model, tool_registry)

    print("🚀 Starting MCP demonstration...")
    print("=" * 50)

    # Example 1: Weather Information
    print("Example 1: Weather Information")
    print("=" * 50)
    user_input = "What's the weather like in Tokyo?"
    print(f"🤖 User: {user_input}")

    response = client.chat(user_input)
    print(f"💬 Response: {response}")
    print()

    # Example 2: Calculator
    print("Example 2: Calculator")
    print("=" * 50)
    user_input = "Calculate 15 * 23 + 7"
    print(f"🤖 User: {user_input}")

    response = client.chat(user_input)
    print(f"💬 Response: {response}")
    print()

    # Example 3: Search
    print("Example 3: Search")
    print("=" * 50)
    user_input = "Search for information about machine learning"
    print(f"🤖 User: {user_input}")

    response = client.chat(user_input)
    print(f"💬 Response: {response}")
    print()

    print("🎉 MCP demonstration completed successfully!")


if __name__ == "__main__":
    main()

"""
## Summary

**What We Built:** A complete MCP system with KerasHub that combines AI language understanding with tool execution.

**Key Benefits:**

- **Actionable AI** - Models can actually execute functions, not just chat
- **Scalable Architecture** - Easy to add new tools and capabilities
- **Production Ready** - Robust error handling and security considerations

**Next Steps:**

- Add more tools (file operations, APIs, databases)
- Implement authentication and permissions
- Build web interfaces or integrate with external services

## Congratulations! 🎉

You've successfully built an MCP system that demonstrates the future of AI interaction - where intelligence meets action! 🚀

"""
