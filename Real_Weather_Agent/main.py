# real weather agent
import os 
import requests
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool
from agents.run import RunConfig

# Load .env variable
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
weather_api_key = os.getenv("WEATHER_API_KEY")

if not gemini_api_key or not weather_api_key:
    raise ValueError("Missing API keys in .env file!")

# Setup Gemini
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(model=model, tracing_disabled=True)

# ✅ Real Weather Tool using API 
@function_tool
def get_weather(city: str) -> str:
    """Get real-time weather info from WeatherAPI."""
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={weather_api_key}&q={city}"
        response = requests.get(url)
        data = response.json()

        location = data['location']['name']
        temp_c = data['current']['temp_c']
        condition = data['current']['condition']['text']

        return f"The current weather in {location} is {temp_c}⁰C with {condition}."
    except Exception as e:
        return f"Sorry, I couldn't fetch the weather for {city}. please try again."
    
    # ✅ Main CLI
def main():
    print("🌏 Real Weather Info Agent")

    agent = Agent(
        name="RealWeatherBot",
        instructions="You are a weather assistant.Use the tool to get real-time weather information.",
        model=model,
        tools=[get_weather]
    )

    while True:
        user_input = input("👨💻 You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Bye!")
            break

        result = Runner.run_sync(agent, user_input, run_config=config)
        print("🤖 Bot:", result.final_output)


if __name__ == "__main__":
    main()
