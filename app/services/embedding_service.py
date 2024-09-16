import os 
import json
from pinecone import Pinecone, ServerlessSpec
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Path to the chunk data in the notebooks folder
CHUNK_DATA_PATH = '/Users/bruce/Desktop/SCHOOL CHATBOT/notebooks/chunk_data.json'

# Pinecone API key and index name
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX_NAME = 'school-chat'

# Initialize the Pinecone instance
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if the index exists, and create it if it doesn't
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,  # 1536 is the dimension for OpenAI embeddings
        metric='cosine',  # You can use 'cosine', 'euclidean', or 'dotproduct'
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'  # Update to your region
        )
    )

# Connect to the index
index = pc.Index(PINECONE_INDEX_NAME)

# Initialize the OpenAIEmbeddings model
embedding_model = OpenAIEmbeddings()

def load_chunk_data(file_path):
    """
    Load chunk data from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file containing chunk data.
    
    Returns:
        dict: A dictionary containing the loaded chunk data.
    """
    with open(file_path, 'r') as f:
        chunks = json.load(f)
    return chunks

def generate_embeddings_for_chunks(chunks):
    """
    Generate embeddings for each chunk's summary and propositions.
    
    Args:
        chunks (dict): Dictionary of chunks where each chunk has an ID, summary, and propositions.
    
    Returns:
        list: A list of dictionaries containing chunk_id, embedding, summary, and propositions.
    """
    chunk_embeddings = []
    for chunk_id, chunk_data in chunks.items():
        # Combine summary and propositions for embedding
        text_to_embed = chunk_data['summary'] + " " + " ".join(chunk_data['propositions'])
        # Generate embedding using OpenAIEmbeddings
        embedding = embedding_model.embed_query(text_to_embed)
        # Store chunk information along with its embedding
        chunk_embeddings.append({
            'chunk_id': chunk_id,
            'embedding': embedding,
            'summary': chunk_data['summary'],
            'propositions': chunk_data['propositions'],
            'title': chunk_data['title']
        })
    return chunk_embeddings

def store_embeddings_in_pinecone(embeddings):
    """
    Store the generated embeddings in Pinecone index along with metadata.
    
    Args:
        embeddings (list): List of dictionaries containing chunk_id, embedding, summary, propositions, and title.
    """
    vectors_to_upsert = []
    for data in embeddings:
        vectors_to_upsert.append({
            'id': data['chunk_id'],  # Use chunk_id as the vector ID
            'values': data['embedding'],  # The embedding itself
            'metadata': {  # Attach the metadata (summary, propositions, title)
                'summary': data['summary'],
                'propositions': data['propositions'],
                'title': data['title']
            }
        })
    
    # Upsert the vectors with metadata into Pinecone
    index.upsert(vectors=vectors_to_upsert)

def inspect_pinecone_index(top_k=10):
    """
    Retrieve and print the first `top_k` vectors in the Pinecone index.
    
    Args:
        top_k (int): The number of vectors to retrieve from Pinecone.
    """
    response = index.query(
        vector=[0] * 1536,  # You can query with a zero vector to get all vectors in your index.
        top_k=top_k,
        include_values=True,
        include_metadata=True
    )
    
    for match in response['matches']:
        print(f"Vector ID: {match['id']}")
        print(f"Embedding: {match['values'][:5]}... (truncated for display)")
        print(f"Metadata: {match.get('metadata', 'No metadata available')}")
        print("-" * 40)

# Load chunks from the chunk_data.json file
chunks = load_chunk_data(CHUNK_DATA_PATH)

# Generate embeddings for the loaded chunks
chunk_embeddings = generate_embeddings_for_chunks(chunks)

# Store the embeddings in Pinecone
store_embeddings_in_pinecone(chunk_embeddings)

# Inspect Pinecone index to verify stored embeddings
inspect_pinecone_index(top_k=5)  # Adjust `top_k` to the number of vectors you want to inspect

print("Embeddings stored in Pinecone successfully!")
