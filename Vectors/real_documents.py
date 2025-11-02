from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

def process_document_file(file_path):
    with open(file_path,'r', encoding='utf-8') as f:
       text =  f.read()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50

    )
    chunks = splitter.split_text(text)
    documents = [
         Document(
             page_content = chunk,
             metadata = {"source":file_path,"chunk_id":i}
         )
         for i,chunk in enumerate(chunks) 
    ]
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(documents,embeddings)
    results = vector_store.similarity_search_with_score("use and what does encoder consist of?", k=2)
    for doc,score in results:
     print(f"Score: {score:.3f}")
     print(f"Text:{doc.page_content}")
     print(f"metadata,{doc.metadata}")
     print("---")
    

process_document_file('/home/brayan/work/AgenticAi/GenAi/data/publication.md')