from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(model_name= "all-MiniLM-L6-v2")

texts = [
     "Vector databases enable semantic search by storing embeddings.",
    "RAG systems combine retrieval with language model generation.",
    "Embeddings capture semantic meaning in numerical form."
]
metadatas = [
      {"topic": "databases", "type": "technical"},
    {"topic": "AI", "type": "technical"},
    {"topic": "ML", "type": "technical"}
]

documents = [
    Document(page_content=text,metadata=metadatas[i])
    for i, text in enumerate(texts)
  
]
vector_store = Chroma.from_documents(documents,embeddings)

result = vector_store.similarity_search_with_score("What is a vector database?", k=2)

for doc, score in result:
    print(f"Score: {score:.3f}")
    print(f"Text:{doc.page_content}")
    print(f"metadata,{doc.metadata}")
    print("---")
