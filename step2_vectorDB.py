import os
# Force transformers to use PyTorch backend and avoid broken TensorFlow detections
os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"

"""
# Vector Databases & Indexing

Vector databases are specialized storage systems designed to handle high-dimensional numerical representations of data (embeddings). 
They are essential for Retrieval-Augmented Generation (RAG) to find relevant context quickly.

## Indexing Algorithms
Indexing is the process of organizing vectors to enable fast similarity searches. Common algorithms include:
- **HNSW (Hierarchical Navigable Small World):** A graph-based approach that is high-performance and widely used.
- **IVF (Inverted File Index):** Groups vectors into clusters to narrow down the search space.
- **LSH (Locality Sensitive Hashing):** Hashes similar vectors into the same "buckets" for fast lookups.

## Distance Metrics
To determine how "similar" two vectors are, we use mathematical distance metrics:
- **Cosine Similarity:** Measures the angle between two vectors (magnitude independent).
- **Euclidean Distance (L2):** Measures the straight-line distance between two points.
- **Dot Product:** Measures both magnitude and direction. done in embeddings.py

## Popular Vector Databases
- **Chroma:** Open-source, easy to use locally.
- **Weaviate:** Multi-modal, cloud and self-hosted.
- **Pinecone:** Managed, cloud-native solution.
- **Milvus:** Highly scalable, enterprise-grade.
- **FAISS:** A library by Meta for efficient similarity search.

but we will use chroma since it is easier 
"""

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# use local sentence-transformer to avoid timeouts during download
embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")


# first create collection with local embedding function
collection = client.get_or_create_collection(
    name="my_local_collection",
    embedding_function=embedding_fn
)

sentences = [
    "There are 4 major subjects being taught in the school",
    "The subjects are digital marketing, software engineering, data analytics and AI",
    "Each major subject has different sub course under them"
]

for i, statem in enumerate(sentences):
    collection.add(
        documents=[statem],
        ids=[f"statement_{i}"]
    )
    


# now lets search something
query = "How many subjects are being taught in the school?"

results = collection.query(
    query_texts=[query],
    n_results=2
)

print(results)