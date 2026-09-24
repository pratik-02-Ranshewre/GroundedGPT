from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
loader = TextLoader("Data/notes.txt")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size = 500 , chunk_overlap = 50)
Chunks = splitter.split_documents(documents)
print(f"the num of chunks :{len(Chunks)}")
print(Chunks[0])
print(Chunks[1])
print(Chunks[2])
print(Chunks[3])
