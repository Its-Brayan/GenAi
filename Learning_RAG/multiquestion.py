from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import yaml
from dotenv import load_dotenv
import os
load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    api_key = os.getenv('GROQ_API_KEY')

)
publication_content = """
Title: One Model, Five Superpowers: The Versatility of Variational Auto-Encoders

TL;DR
Variational Auto-Encoders (VAEs) are versatile deep learning models with applications in data compression, noise reduction, synthetic data generation, anomaly detection, and missing data imputation. This publication demonstrates these capabilities using the MNIST dataset, providing practical insights for AI/ML practitioners.

Introduction
Variational Auto-Encoders (VAEs) are powerful generative models that exemplify unsupervised deep learning. They use a probabilistic approach to encode data into a distribution of latent variables, enabling both data compression and the generation of new, similar data instances.
[rest of publication content... truncated for brevity]
"""

conversation = [
    SystemMessage( content = f"""
You are an Ai assistant discussing a research publication.
Base your answer only on this pubication content:

{publication_content}
"""
    )
]
conversation.append(
    HumanMessage(content=f"""
What are variational autoencoders and list the top 5 applications for them as discussed in this publication.
                 """)
)
response1 = llm.invoke(conversation)
print("Ai response 1")
print(response1.content)
print("\n" + "="*50 + "\n")

# Add AI's response to conversation history
conversation.append(AIMessage(content=response1.content))

conversation.append(HumanMessage(content = """
          How does it work in case of anomaly detection?                       
"""))
response2 = llm.invoke(conversation)
print("Ai response to question 2")
print(response2.content)