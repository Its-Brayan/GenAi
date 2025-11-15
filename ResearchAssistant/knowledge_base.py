import os
from langchain_community.document_loaders import TextLoader
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import torch


# Initialize ChromaDB
client = chromadb.PersistentClient(path="./research_db")
collection = client.get_or_create_collection(
    name = "ml_publications",
    metadata={"hnsw:space": "cosine"}
)

embedding = HuggingFaceEmbeddings(
     model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def load_research_publications(documents_path):
    """Load research publications from .txt files and return as list of strings"""
    #list to store all documents
    documents = []

    #load each .txt file in the documents folder
    for file in os.listdir(documents_path):
        if file.endswith(".txt"):
            file_path = os.path.join(documents_path,file)
            try:
                loader = TextLoader(file_path)
                loaded_docs = loader.load()
                documents.extend(loaded_docs)
                print(f"Successfully loaded {file}")
            except Exception as e:
                print(f"Error loading {file} : {str(e)}")
    print(f"Total documents loaded: {len(documents)}")
    # Extract content as strings and return
    publications = []
    for doc in documents:
        publications.append(doc.page_content)

    return publications

def chunk_research_paper(paper_content,title):
       """Break a research paper into searchable chunks"""
       text_splitter = RecursiveCharacterTextSplitter(
           chunk_size = 1000,
           chunk_overlap = 200,
           separators = ["\n\n","\n",". "," ",""]
       ) 
       chunks = text_splitter.split_text(paper_content)

       #add metadata to each chunk
       chunk_data = []
       for i, chunk in enumerate(chunks):
           chunk_data.append({
               "content":chunk,
               "title":title,
               "chunk_id":f"{title}_{i}",
           })
           return chunk_data

def embed_documents(documents : list[str] -> list[list[float]]):
    """
    Embed documents using a model
    """
    devices = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    model = HuggingFaceEmbeddings(
     model_name = "sentence-transformers/all-MiniLM-L6-v2",
     model_kwargs ={"device":devices}
    )
    embeddings = model.embed_documents(documents)
    return embeddings

def instert_publications(collection:chromadb.Collection,publications: list[str]):
       """
    Insert documents into a ChromaDB collection.

    Args:
        collection (chromadb.Collection): The collection to insert documents into
        publications (list[str]): The documents to insert

    Returns:
        None
    """
       next_id = collection.count()

       for publication in publications:
            chunked_publication = chunked_publication(publication)
            embeddings = embed_documents(chunked_publication)
            ids = list(range(next_id,next_id + len(chunked_publication)))
            ids = [f"document_{id}" for id in  ids]
            collection.add(
                 ids = ids,
                 documents = chunked_publication
                  )
            next_id += len(chunked_publication)

#intelligent retrieval
def search_request_db(query,collection,embeddings, top_k = 5):
    """Find the most relevant research chunks for a query"""
    #convert questions to vector
    query_vector = embeddings.embed_query(query)

    #search for similar content
    results = collection.query(
         query_embeddings = [query_vector],
         n_results = top_k,
         include = ["documents","metadatas","distances"]

    )
    #format results
    relevant_chunks = []
    for i, doc in enumerate(results["documents"][0]):
         relevant_chunks.append({
              "content":doc,
              "title":results["metadatas"][0][i]["title"],
              "similarity": 1 - results["distances"][0][i]
         })
    return relevant_chunks



                
            



          



    
