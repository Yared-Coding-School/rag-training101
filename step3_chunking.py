"""
    Chunking: is a method adopted when documents are too long to be processed as a whole
    
    1. Fixed Size Chunking: - 
        Simply take a fixed number like 500 char per chunk - simple and reliable - but breaks meaning
        to solve the breaking, we add ovelapping to provide context between the breakings

    2. Sentence or Paragraph Based Chunking: - 
        Split the text into sentences or a paragraph and then group them into chunks - more meaningful 
        but can be more complex to implement

    3. Recursive Chunking: - 
        It tries multiple separators to chunk the text - more flexible and can handle complex documents
""" 
import chromadb

def chunk_document(text, chunk_size=500, overlap=50):
    """
        Chunk the text into smaller chunks of fixed size with overlap
    """
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    
    return chunks


client = chromadb.Client()
collection = client.get_or_create_collection(name="chunking")

document = "..." # large document 
chunks = chunk_document(document)

# add chunks to vector database
for i, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk],
        ids=[f"chunk_{i}"]
    )

# query 
query = "What are the password requirements"

results = collection.query(
    query_texts=[query],
    n_results=3
)

print(results)