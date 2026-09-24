from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector = embeddings.embed_query("what is RAG?")

vector2 = embeddings.embed_query("What does RAG stand for?")
vector3 = embeddings.embed_query("whats my fav pizza topping?")
from sklearn.metrics.pairwise import cosine_similarity
similarity_related = cosine_similarity([vector] ,[vector2])
similarity_unrelated = cosine_similarity([vector],[vector3])
print("Self-similarity (should be ~1.0):", cosine_similarity([vector], [vector]))
print("Similarity (RAG vs RAG, different wording):", similarity_related)
print("Similarity (RAG vs unrelated pizza question):", similarity_unrelated)