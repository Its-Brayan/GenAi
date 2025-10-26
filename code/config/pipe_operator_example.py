from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
# First chain generates questions
question_prompt = PromptTemplate(
    input_variables=['topic'],
    template = "Generate 3 questions about {topic}"
)

# Second chain generates answers based on questions
answer_prompt = PromptTemplate(
    input_variables=['questions'],
    template ="Answer the following questions:\n{questions}\n You response should contain the question and the answer to it."
)

llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="groq/compound")

# Output parser to convert model output to string
output_parser = StrOutputParser()

# Build the question generation chain
question_chain = question_prompt | llm | output_parser

# Build the answer generation chain
answer_chain = answer_prompt | llm | output_parser

def create_answer_input(output):
    return {'questions':output}

# Chain everything together
qa_chain = question_chain | create_answer_input | answer_chain

result = qa_chain.invoke({'topic':'artificial intelligence'})
print(f"Final QA Response: {result}\n")
