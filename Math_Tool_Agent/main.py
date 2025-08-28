# Math Tool Agent
import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel,function_tool
from agents.run import RunConfig


# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not foundin .env!")

# Set up Gemini-compatible client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
) 

# set up the module
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(model=model, tracing_disabled=True)

# ✅ Tool registration using decorator
@function_tool
def add (a: int, b: int) -> int:
    """Add two integers."""
    return a + b

# ✅ Main interactive function

def main():
    print("➕ Welcome to the Math Tool Agent!")

agent = Agent(
    name="MathToolBot",
    instructions="Your are a helpful math agent.Use the add tool when needed.",
    model=model,
    tools=[add]
)

while True:
    user_input = input("🔢 You:")
    if user_input.lower() in ['exit', 'quite']:
        print("👋 Goodbye!")

    result = Runner.run_sync(agent, user_input, run_config=config)
    print("🤖 Bot:", result.final_output)

if __main__ == "__main__":
    main()

