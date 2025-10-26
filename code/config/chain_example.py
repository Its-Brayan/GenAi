from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os
from groq import Groq

load_dotenv()
prompt = PromptTemplate(
    input_variables=['topic'],
    template = "Generate 3 questions about {topic}"
)
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"),model="groq/compound")
question_chain = prompt | llm
questions = question_chain.invoke({'topic':'artificial intelligence'})
print(f"Questions: {questions.content}\n")