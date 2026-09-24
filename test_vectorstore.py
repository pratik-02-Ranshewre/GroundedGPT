import shutil
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
loaders = TextLoader("Data/notes.txt")
documents = loaders.load()
splitter = RecursiveCharacterTextSplitter(chunk_size = 500 , chunk_overlap = 50)
Chunks = splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print(f"Ready : {len(Chunks)} chunks, embedding model loaded")
if os.path.exists("chroma_db"):
    shutil.rmtree("chroma_db")
    print("Cleared old chroma_db")
vectorstore = Chroma.from_documents(
    documents= Chunks,
    embedding= embeddings,
    persist_directory= "chroma_db"
)
print("vector store created and stored to chroma db")
query = " what is chunking and why does it matter ?"
results = vectorstore.similarity_search(query , k=3)
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} START ---")
    content = doc.page_content
    print("REPR:", repr(content[:200]))
    print(f"--- Result {i+1} END ---")