import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

available_models =[
    'groq/compound',
    'groq/compound-pro',
    'groq/compound-x',

]
def get_llm(model:str):
    if model not in available_models:
        raise ValueError(f"Invalid model. Availabe models {available_models}")
    return ChatGroq(
        model=model,
        api_key=os.getenv("GROK_API_KEY"),
        temperature = 0.0
    )