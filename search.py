import json
import numpy as np
# from sentence_transformers import SentenceTransformer
import faiss


def load_processed_issues(json_file_path):
    with open(json_file_path, 'r') as file:
        processed_issues = json.load(file)
    return processed_issues


def generate_embeddings(texts, model):
    return model.encode(texts)


def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  
    index.add(embeddings)
    return index


def search_faiss_index(index, query_embedding, k=3):
    distances, indices = index.search(np.array([query_embedding]), k)
    return distances, indices


def handle_user_query(json_file_path, user_query, model):
    processed_issues = load_processed_issues(json_file_path)
    texts = [issue['text'] for issue in processed_issues]
    embeddings = generate_embeddings(texts, model)
    index = create_faiss_index(embeddings) 
    query_embedding = model.encode([user_query])[0]
    distances, indices = search_faiss_index(index, query_embedding)
    relevant_issues = [processed_issues[idx] for idx in indices[0]]
    return relevant_issues

# Test function to run
# def main():   
#     model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")    
#     json_file_path = "final_concatenated_issues.json"  
#     user_query = "Is it possible to perform progressive text generation in Transformers 4.46.3 with only inputs_embeds and past_key_values ?"
#     relevant_issues = handle_user_query(json_file_path, user_query, model)
#     for issue in relevant_issues:
#         print(f"Issue #{issue['number']}")        
#         print(f"Text: {issue['text']}")  
#         print("-" * 80)

