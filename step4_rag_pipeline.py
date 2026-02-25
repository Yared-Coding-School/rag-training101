"""
    RAG Pipeline: 
    Step 1:
        1. Documents should be prepared 
        2. Chunking them
        3. Embedding them to vectors
        4. Store them in VectorDB

    Step 2: user query time
        1. User Queries using search
        2. Retrieve relevant chunks from the vector database and augment 
        3. Generate a response using the retrieved chunks and the query

    Caching - solve the problem of repeated queries
        Query cache: 
        Embeding Cache
        Vector Cache
        LLM Response Cache 
    
    Redis - a popular in-memory data structure store, used as a database, cache, and message broker
"""

import chromadb

# Initialize ChromaDB
client = chromadb.Client()
collection = client.get_or_create_collection(name="rag_pipeline")

# Add documents
collection.add(
    documents=[
        "The capital of France is Paris.",
        "The Eiffel Tower is located in Paris.",
        "The Louvre Museum is in Paris."
    ],
    ids=["doc1", "doc2", "doc3"]
)

# User query
query = "What is the capital of France?"

# Retrieve relevant chunks
results = collection.query(
    query_texts=[query],
    n_results=2
)

# Augment the query with retrieved chunks
augmented_query = f"{query}\n\nContext:\n{results['documents'][0]}"

# Generate response (using a placeholder LLM)
response = f"Based on the retrieved information:\n{augmented_query}\n\nResponse: The capital of France is Paris."

print(response)