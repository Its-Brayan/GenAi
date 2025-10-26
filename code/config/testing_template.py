from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()

template = PromptTemplate(
    input_variables = ["country"],
    template = "What is the capital of {country}"
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
for country in ["France","Kenya","USA"]:
    formatted_prompt = template.format(country=country)
    print(f"Prompt: {formatted_prompt}")
    response = client.chat.completions.create(
        messages = [
            {
                "role": "user",
                "content": formatted_prompt
            }
        ],
        model = "groq/compound"
    )
    print(f"Response: {response.choices[0].message.content}\n")
