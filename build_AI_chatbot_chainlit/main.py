import os
import chainlit as cl
from dotenv import load_dotenv
from openai import AsyncOpenAI


load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="👋AI Chatbot").send()

@cl.on_message
async def on_message(message: cl.Message):
    try:
        response = await client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": message.content}],
        )   
        await cl.Message(content=response.choices[0].message.content).send()
    except Exception as e:
        await cl.Message(content=f"Eror: {str(e)}").send() 

