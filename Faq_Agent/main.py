import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in .env!")


external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(model=model, tracing_disabled=True)

def main():
    print("🤖 Welcome to the FAQ Bot")
    print("Type 'exit' to end the conversation.\n")

agent = Agent(
    name="FAQ Bot",
    instructions="You are a helpful FAQ bot. Answer questions about yourself.",
    model=model
)
    
while True:
    user_question = input("❓ You: ")
    if user_question.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    result = Runner.run_sync(agent, user_question, run_config=config)
    print("🤖 Bot:", result.final_output)


if __name__ == "__main__":
    main()


