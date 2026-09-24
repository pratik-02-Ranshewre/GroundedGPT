import streamlit as st
from langchain_community.document_loaders import TextLoader , DirectoryLoader,PyMuPDFLoader,PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
import json 
import re

st.title("GroundedGPT")
mode = st.radio("choose mode :", ["Ask question", "Generate quize"])
st.write("Ask the question about your notes ")
@st.cache_resource
def setup():
    txt_loader = DirectoryLoader("Data",glob="**/*.txt", loader_cls=TextLoader)
    pdf_loader= DirectoryLoader("Data", glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    documents = txt_loader.load()+pdf_loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size = 500 , chunk_overlap= 50)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents= chunks,embedding=embeddings,persist_directory= "Chroma_db")
    llm = ChatOllama(model="llama3")
    return vectorstore,llm

vectorstore,llm = setup()
if mode == "Ask question":
  question = st.text_input("your question:")
  if question:
    relevant_chunks = vectorstore.similarity_search(question,k=3)
    context = "\n\n".join([doc.page_content for doc in relevant_chunks])
    prompt = f"""Answer thr question ONLY using the context below . If the context doesn't contain the answer, say "I don't know based on the provided notes.
Context:
{context}
Question: {question}

Answer:"""

    with st.spinner("Thinking..."):
        response = llm.invoke(prompt)

    st.write(f"**Answer:** {response.content}")
elif mode == "Generate quize":
    topic = st.text_input("Enter a topic for the quiz:")

    if topic:
        relevant_chunks = vectorstore.similarity_search(topic, k=4)
        context = "\n\n".join([doc.page_content for doc in relevant_chunks])

        quiz_prompt = f"""Based ONLY on the study material below, generate exactly 2 multiple-choice questions.
Do not invent facts that aren't in the material.

Study material:
{context}

Respond with ONLY valid JSON, no other text, in this exact format:
[
  {{"question": "...", "options": {{"a": "...", "b": "...", "c": "...", "d": "..."}}, "correct": "a", "explanation": "..."}}
]"""

        with st.spinner("Generating quiz..."):
            response = llm.invoke(quiz_prompt)

        def extract_json(text):
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                return match.group()
            return text

        try:
            cleaned_output = extract_json(response.content)
            quiz_data = json.loads(cleaned_output)

            for i, q in enumerate(quiz_data):
                st.write(f"**Q{i+1}: {q['question']}**")
                choice = st.radio("Choose one:", list(q["options"].values()), key=f"q_{i}")
                st.caption(f"Correct answer: {q['options'][q['correct']]} — {q['explanation']}")

        except json.JSONDecodeError:
            st.error("Couldn't parse the quiz. Try again.")