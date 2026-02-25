from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Force transformers to use PyTorch backend and avoid broken TensorFlow detections
os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"

"""
    Keyword Search: Finds exact matches of words in documents.
        - TF-IDF (Term Frequency-Inverse Document Frequency): Measures word importance relative to a document and a collection.
        - BM25 (Best Matching 25): A ranking algorithm that improves on TF-IDF by handling document length and term frequency saturation better.

    Meaning Search (Semantic/Vector Search): Finds documents based on their underlying concepts and intent, even if they don't share exact words.

    Embedding Models: Neural networks that transform text into high-dimensional numerical vectors (embeddings), enabling mathematical comparison of meanings.

    using two use cases - one local and one remote
"""

# LOCAL

from sentence_transformers import SentenceTransformer
import numpy as np

# local models that runs on laptops and computers
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Dogs are allowed in the office on Fridays",
    "Pets are permitted in the workplace on Fridays",
    "Canine companions are welcome on Fridays",
    "Felines are not permitted in the office",
    "Cats are prohibited from the workplace",
    "No pets allowed on Fridays"
]

embeddings = model.encode(sentences)

print(embeddings.shape)
print(embeddings[0])

sim12 = np.dot(embeddings[0], embeddings[1])
sim13 = np.dot(embeddings[0], embeddings[2])
sim14 = np.dot(embeddings[0], embeddings[3])
sim15 = np.dot(embeddings[0], embeddings[4])
sim16 = np.dot(embeddings[0], embeddings[5])

print("Similarity between sentence 1 and 2: ", sim12)
print("Similarity between sentence 1 and 3: ", sim13)
print("Similarity between sentence 1 and 4: ", sim14)
print("Similarity between sentence 1 and 5: ", sim15)
print("Similarity between sentence 1 and 6: ", sim16)


# -----------------------------------------------------------------------------

# remote with requests using hugging face api
import requests
import os 


API_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-base-en-v1.5/pipeline/feature-extraction"
headers = {
    "Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

output = query({
    "inputs": "Today is a sunny day and I will get some ice cream.",
})

print("Output: ", output)

