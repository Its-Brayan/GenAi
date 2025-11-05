from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationChain
from langchain_community.chat_message_histories import SQLChatMessageHistory
import yaml
from dotenv import load_dotenv
import os
load_dotenv()

llm = ChatGroq(
     model="llama-3.1-8b-instant",
    temperature=0.7,
    api_key = os.getenv('GROQ_API_KEY')
)

memory = ConversationBufferMemory(
    chat_memory= SQLChatMessageHistory(
        session_id = "user123_session456",
        connection_string= "sqlite:///chat_history.db"
    ),
    return_messages = True
)
conversation = ConversationChain(llm=llm,memory=memory)
response = conversation.predict(input = "Who is the president of kenya?")
print(response)