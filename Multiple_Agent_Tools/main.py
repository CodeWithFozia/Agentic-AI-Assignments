# Multiple Agents Tools
import os
from dotenv import load_dotenv
from agents import Agent, OpenAIChatCompletionsModel, Runner, AsyncOpenAI, function_tool
from agents.run import RunConfig

# Load API key
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in .env!")

# Setup Gemini client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(model=model, tracing_disabled=True)

# ✅ Tool 1: Math function
@function_tool
def add(a: int, b: int) -> int:
    """add two integers and return the result."""
    return a + b

# ✅ Tool 2: Weather function (mocked)
@function_tool
def get_weather(city: str) -> str:
    """Return mock weather information for the given city."""
    return f"The weather in {city} is sunny with 34⁰C."

# ✅ Main CL
def main():
    print("🛠 Welcome to Multi-Tool Agent!")
    print("Type 'exit' to quit.\n")

agent = Agent(
    name= "SmartBot",
    instructions="You are a smart assistant. Use tools to answer math and weather question.",
    model=model,
    tools=[add, get_weather] # Both tools registered
)

while True:
    user_input = input("🤖 You: ")
    if user_input.lower() in ["exit", "quit"]:
        print(" 👋 Goodby!")
        break
    result = Runner.run_sync(agent,user_input,run_config=config)
    print("🤖 Bot:",result.final_output)
    

if __name__ == "__main__":
    main()
