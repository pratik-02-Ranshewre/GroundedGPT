import shutil
import os
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
loader = TextLoader("Data/notes.txt")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size = 500 , chunk_overlap= 50)
chunks = splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if os.path.exists("Chroma_db"):
    shutil.rmtree("chroma_db")

vectorstore = Chroma.from_documents(
    documents= chunks,
    embedding= embeddings ,
    persist_directory= "chroma_db"
)
print("Setup complete: documents loaded, chunked, embedded, and stored.")
llm = ChatOllama(model = "llama3")
question = "What is the capital of France?"
relevant_chunks = vectorstore.similarity_search(question , k=3)
print(f"found {len(relevant_chunks)} relevant chunks")
context = "n/n". join([doc.page_content for doc in relevant_chunks])
prompt = f""" Answer the question using ONLY context below . If context doesn't 
context:
{context}
question:{question}
Answer:"""
response = llm.invoke(prompt)
print("\n---ANSWER---")
print(response.content)