from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import yaml
from dotenv import load_dotenv
import os
load_dotenv()
# Initialize the LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)
# Basic question
messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="What are variational autoencoders and list the top 5 applications for them?")
]

response = llm.invoke(messages)
print(response.content)
