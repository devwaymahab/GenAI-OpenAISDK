from openai import OpenAI
import numpy as np
from dotenv import load_dotenv

load_dotenv() 
client = OpenAI()

documents = [
    "FastAPI provides OAuth2 and JWT authentication.",
    "Django provides built-in authentication and user management.",
    "Python lists are ordered collections.",
    "FastAPI can handle request validation using Pydantic."
]

query = "How can I implement authentication in FastAPI?"


# Create embeddings
doc_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=documents
)

query_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=query
)

doc_vectors = [
    item.embedding
    for item in doc_response.data
]

query_vector = query_response.data[0].embedding


def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )
    
results = []

for document, vector in zip(documents, doc_vectors):
    score = cosine_similarity(query_vector, vector)
    results.append((document, score))

results.sort(key=lambda x: x[1], reverse=True)

for document, score in results:
    print(f"{score:.4f} → {document}")