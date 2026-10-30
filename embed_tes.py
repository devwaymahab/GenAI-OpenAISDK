from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

client = OpenAI()

texts = [
    "How do I reset my password?",
    "I forgot my password.What to do?",
    "What is the weather today?"
]

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)

for text, item in zip(texts, response.data):
    print(text)
    print(item.embedding[:15])
    print("Vector length:", len(item.embedding))
    print("==================================")
    
import numpy as np

vectors = [item.embedding for item in response.data]

def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )
    
similarity_1 = cosine_similarity(vectors[0], vectors[1])
similarity_2 = cosine_similarity(vectors[0], vectors[2])

print(
    "Password vs Forgot password:",
    similarity_1
)

print(
    "Password vs Weather:",
    similarity_2
)

