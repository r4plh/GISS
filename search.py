import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Step 1: Load Processed Issues from JSON File
def load_processed_issues(json_file_path):
    with open(json_file_path, 'r') as file:
        processed_issues = json.load(file)
    return processed_issues

# Step 2: Generate Embeddings
def generate_embeddings(texts, model):
    return model.encode(texts)

# Step 3: Create FAISS Index
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance for nearest neighbor search
    index.add(embeddings)
    return index

# Step 4: Semantic Search with FAISS
def search_faiss_index(index, query_embedding, k=3):
    distances, indices = index.search(np.array([query_embedding]), k)
    return distances, indices

# Step 5: Real-Time Query Handling
def handle_user_query(json_file_path, user_query, model):
    # Load processed issues from JSON file
    processed_issues = load_processed_issues(json_file_path)
    
    # Extract texts for embedding generation
    texts = [issue['text'] for issue in processed_issues]
    
    # Generate embeddings for the text column
    embeddings = generate_embeddings(texts, model)
    
    # Create FAISS index
    index = create_faiss_index(embeddings)
    
    # Encode user query
    query_embedding = model.encode([user_query])[0]
    
    # Perform semantic search
    distances, indices = search_faiss_index(index, query_embedding)
    
    # Return relevant issues
    relevant_issues = [processed_issues[idx] for idx in indices[0]]
    return relevant_issues

# Main Function
def main():
    # Load SentenceTransformer model
    model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
    
    # Path to the JSON file containing processed issues
    json_file_path = "final_concatenated_issues.json"  # Replace with your JSON file path
    
    # Example user query
    user_query = "Is it possible to perform progressive text generation in Transformers 4.46.3 with only inputs_embeds and past_key_values ?"
    
    # Handle user query
    relevant_issues = handle_user_query(json_file_path, user_query, model)
    
    # Print relevant issues
    for issue in relevant_issues:
        print(f"Issue #{issue['number']}")
        # print(f"URL: {issue['html_url']}")
        print(f"Text: {issue['text']}")  # Print first 200 chars of text
        print("-" * 80)

if __name__ == "__main__":
    main()