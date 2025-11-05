from langchain_classic.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationChain
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()
memory = ConversationBufferMemory(
    chat_memory = FileChatMessageHistory("chat_history_user123.json"),
    return_messages= True
)
llm = ChatGroq(
  model="llama-3.1-8b-instant",
    temperature=0.7,
    api_key = os.getenv('GROQ_API_KEY')
)
conversation = ConversationChain(
    llm = llm,
    memory = memory
)
response = conversation.predict(input="What are VAEs?")
print(response)