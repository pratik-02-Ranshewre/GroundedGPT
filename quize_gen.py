import os
import shutil
import json
import re
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

loader = TextLoader("Data/notes.txt")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size = 500 , chunk_overlap = 50)
chunks = splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
if os.path.exists("chroma_db"):
    shutil.rmtree("chroma_db")

vectorstore = Chroma.from_documents(
    documents= chunks,
    embedding= embeddings,
persist_directory="chroma_db"
)
llm = ChatOllama(model= "llama3")
print("setup complete")
topic = " Chunking"
relevant_chunks = vectorstore.similarity_search(topic , k=4)
context = "\n\n".join([doc.page_content for doc in relevant_chunks])
quize_prompt = f"""
Based ONLY on the study material below, generate exactly 2 multiple-choice questions.
Do not invent facts that aren't in the material.

study material:
{context}

Respond with ONLY valid JSON ,no other text , in this format
[
  {{"question": "...", "options": {{"a": "...", "b": "...", "c": "...", "d": "..."}}, "correct": "a", "explanation": "..."}}
]"""
response = llm.invoke(quize_prompt)
print("\n--- RAW LLM OUTPUT---")
print(response.content)
def extract_json(text):
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        return match.group()
    return text

try:
    cleaned_output = extract_json(response.content)
    quize_data = json.loads(cleaned_output)
    print(f"\nSuccessfully parsed {len(quize_data)} questions")
    print("First Question:", quize_data[0]["question"])
    print("Correct answer:", quize_data[0]["correct"])
except json.JSONDecodeError as e:
    print(f"Failed to parse json: {e}")