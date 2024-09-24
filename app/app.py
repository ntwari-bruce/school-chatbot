from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from openai import OpenAI
import os
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings  # Updated import
import pinecone
import time

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}) 


# Check for required environment variables
required_env_vars = ['OPENAI_API_KEY', 'PINECONE_API_KEY', 'PINECONE_INDEX_NAME']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize Pinecone
pinecone.init(api_key=os.getenv('PINECONE_API_KEY'))
index = pinecone.Index(os.getenv('PINECONE_INDEX_NAME'))

# Initialize OpenAIEmbeddings
embedding_model = OpenAIEmbeddings()

def get_relevant_chunks(query, top_k=3):
    # Generate embedding for the query
    query_embedding = embedding_model.embed_query(query)
    
    # Query Pinecone
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    
    # Extract relevant information
    relevant_chunks = []
    
    for match in results.matches:
        # Check if metadata is present
        if match.metadata:
            relevant_chunks.append({
                'chunk_id': match.id,
                'score': match.score,
                'summary': match.metadata.get('summary', ''),
                'propositions': match.metadata.get('propositions', []),
                'title': match.metadata.get('title', '')
            })
        else:
            relevant_chunks.append({
                'chunk_id': match.id,
                'score': match.score,
                'summary': '',
                'propositions': [],
                'title': ''
            })
    
    return relevant_chunks

def make_openai_request(prompt, context):
    try:
        messages = [
            {"role": "system", "content": "You are an assistant that helps users with questions based on the provided context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        return {"error": str(e)}

@app.route('/api/chat', methods=['POST'])
@cross_origin()
def chat():
    user_input = request.json.get('question')
    
    if not user_input:
        return jsonify({"error": "No input provided"}), 400
    
    # Get relevant chunks from Pinecone
    relevant_chunks = get_relevant_chunks(user_input)
    
    # Prepare context from relevant chunks
    context = "\n".join([
        f"Title: {chunk['title']}\nSummary: {chunk['summary']}\nPropositions: {', '.join(chunk['propositions'])}"
        for chunk in relevant_chunks
    ])
    
    # Retry the request up to 3 times if there is a connection error
    retries = 3
    for attempt in range(retries):
        generated_response = make_openai_request(user_input, context)
        if isinstance(generated_response, str):
            return jsonify({"response": generated_response})
        else:
            time.sleep(2 ** attempt)  # Exponential backoff between retries
    
    # If all attempts failed
    return jsonify({"error": "Failed to connect to OpenAI API after multiple attempts"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)