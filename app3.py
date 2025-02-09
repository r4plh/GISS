import os
import json
import time
import numpy as np
import requests
from flask import Flask, request, render_template, Response, stream_with_context, jsonify
from sentence_transformers import SentenceTransformer
import faiss
import logging 

# Gemini AI
import google.generativeai as genai

# Configure your Gemini API key here
genai.configure(api_key="Enter-your-gemini-api-key")

# Import your custom pipeline and search methods
from data.dataPipeline import GitHubIssuesPipeline
from search import (
    generate_embeddings, 
    create_faiss_index, 
    search_faiss_index
)

def get_total_issues_and_prs(owner: str, repo: str) -> dict:
    """
    Fetch the total number of issues and pull requests in a GitHub repository.

    Args:
    owner (str): The GitHub username of the owner of the repository.
    repo (str): The name of the repository.

    Returns:
    dict: A dictionary containing the total number of non-PR issues and pull requests.
    """
    base_url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {'Accept': 'application/vnd.github.v3+json'}

    # Get total count of pull requests (open and closed)
    prs_count = requests.get(f"{base_url}/pulls?state=all&per_page=1", headers=headers)
    total_prs = prs_count.headers.get('Link').split(',')[1].split('&page=')[1].split('>')[0] if 'Link' in prs_count.headers else 0

    # Get total count of issues (open and closed)
    issues_count = requests.get(f"{base_url}/issues?state=all&per_page=1&filter=all", headers=headers)
    total_issues = issues_count.headers.get('Link').split(',')[1].split('&page=')[1].split('>')[0] if 'Link' in issues_count.headers else 0

    return {
        'total_issues': int(total_issues),  
        'total_issues(which are not PRs)': int(total_issues) - int(total_prs),
        'total_pull_requests': int(total_prs)
    }

logging.basicConfig(level=logging.DEBUG)

def process_github_issues(owner, repo, token, user_query, k, num_issues=100):
    """Generator function to stream progress and results."""
    try:
        logging.debug("Initializing search...")
        yield "data: Initializing search...\n\n"
        time.sleep(0.5)

        # Get total issues count
        repo_stats = get_total_issues_and_prs(owner, repo)
        total_issues = repo_stats['total_issues']
        
        # Use total_issues if provided, otherwise fallback to default
        actual_num_issues = total_issues if total_issues > 0 else num_issues

        # Check if cached data exists
        cached_issues = GitHubIssueCache.load_issues(owner, repo)
        cached_embeddings = GitHubIssueCache.load_embeddings(owner, repo)
        cached_index = GitHubIssueCache.load_faiss_index(owner, repo)

        # If no cached data, fetch fresh from GitHub
        if cached_issues is None or cached_embeddings is None or cached_index is None:
            logging.debug("Fetching and processing issues from GitHub...")
            yield "data: Fetching and processing issues from GitHub...\n\n"
            time.sleep(0.5)
            pipeline = GitHubIssuesPipeline(owner, repo, token, num_issues=actual_num_issues)
            final_json_str = pipeline.run_full_pipeline()
            logging.debug("Generating embeddings...")
            yield "data: Generating embeddings...\n\n"
            time.sleep(0.5)
            processed_issues = json.loads(final_json_str)
            texts = [issue['text'] for issue in processed_issues]
            embeddings = generate_embeddings(texts, model)

            logging.debug("Creating FAISS index...")
            yield "data: Creating FAISS index...\n\n"
            time.sleep(0.5)
            index = create_faiss_index(embeddings)

            # Save to cache
            GitHubIssueCache.save_issues(owner, repo, processed_issues)
            GitHubIssueCache.save_embeddings(owner, repo, embeddings)
            GitHubIssueCache.save_faiss_index(owner, repo, index)
        else:
            logging.debug("Using cached GitHub issues data...")
            yield "data: Using cached GitHub issues data...\n\n"
            time.sleep(0.5)
            processed_issues = cached_issues
            embeddings = cached_embeddings
            index = cached_index

        logging.debug("Encoding user query and searching index...")
        yield "data: Encoding user query and searching index...\n\n"
        time.sleep(0.5)
        query_embedding = model.encode([user_query])[0]
        distances, indices = search_faiss_index(index, query_embedding, k=k)
        relevant_issues = [processed_issues[idx] for idx in indices[0]]

        logging.debug(f"Relevant issues found: {relevant_issues}")

        logging.debug("Search complete. Preparing results...")
        yield "data: Search complete. Preparing results...\n\n"
        time.sleep(0.5)

        # Prepare final results with detail, including the raw 'text' for each issue
        formatted_results = []
        for issue in relevant_issues:
            formatted_issue = {
                'number': issue['number'],
                'title': issue.get('title', 'No Title'),
                'html_url': issue.get('html_url', ''),
                'body': issue.get('body') if issue.get('body') is not None else '',
                'comments': issue.get('comments', ''),
                'text': issue.get('text', '')  # needed for AI context
            }
            formatted_results.append(formatted_issue)

        # Serialize to JSON with proper Unicode handling
        issues_json = json.dumps(formatted_results, ensure_ascii=False)
        logging.debug(f"Sending JSON data: {issues_json}")

        # Send results as JSON to the frontend
        yield f"data: {issues_json}\n\n"

    except Exception as e:
        error_message = f"Error: {str(e)}"
        logging.error(f"An exception occurred: {error_message}")
        yield f"data: {error_message}\n\n"

class GitHubIssueCache:
    CACHE_DIR = 'git-repos'
    
    @classmethod
    def _get_cache_key(cls, owner, repo):
        """Generate a unique cache key for owner and repo."""
        return f"{owner}-{repo}"
    
    @classmethod
    def _ensure_cache_dir(cls):
        """Ensure the cache directory exists."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
    
    @classmethod
    def save_issues(cls, owner, repo, issues):
        """Save issues to a JSON file."""
        cls._ensure_cache_dir()
        cache_key = cls._get_cache_key(owner, repo)
        filepath = os.path.join(cls.CACHE_DIR, f"{cache_key}_issues.json")
        
        with open(filepath, 'w') as f:
            json.dump(issues, f)
    
    @classmethod
    def save_embeddings(cls, owner, repo, embeddings):
        """Save embeddings as numpy array."""
        cls._ensure_cache_dir()
        cache_key = cls._get_cache_key(owner, repo)
        filepath = os.path.join(cls.CACHE_DIR, f"{cache_key}_embeddings.npy")
        
        np.save(filepath, embeddings)
    
    @classmethod
    def save_faiss_index(cls, owner, repo, index):
        """Save FAISS index."""
        cls._ensure_cache_dir()
        cache_key = cls._get_cache_key(owner, repo)
        filepath = os.path.join(cls.CACHE_DIR, f"{cache_key}_index.faiss")
        
        faiss.write_index(index, filepath)
    
    @classmethod
    def load_issues(cls, owner, repo):
        """Load issues from JSON file."""
        cache_key = cls._get_cache_key(owner, repo)
        filepath = os.path.join(cls.CACHE_DIR, f"{cache_key}_issues.json")
        
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return json.load(f)
        except (IOError, json.JSONDecodeError):
            pass
        return None
    
    @classmethod
    def load_embeddings(cls, owner, repo):
        """Load embeddings from numpy array."""
        cache_key = cls._get_cache_key(owner, repo)
        filepath = os.path.join(cls.CACHE_DIR, f"{cache_key}_embeddings.npy")
        
        try:
            if os.path.exists(filepath):
                return np.load(filepath)
        except (IOError, ValueError):
            pass
        return None
    
    @classmethod
    def load_faiss_index(cls, owner, repo):
        """Load FAISS index."""
        cache_key = cls._get_cache_key(owner, repo)
        filepath = os.path.join(cls.CACHE_DIR, f"{cache_key}_index.faiss")
        
        try:
            if os.path.exists(filepath):
                return faiss.read_index(filepath)
        except RuntimeError:
            pass
        return None

app = Flask(__name__)

# Pre-load the sentence-transformers model
model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    owner = request.form.get('owner')
    repo = request.form.get('repo')
    token = request.form.get('token')
    user_query = request.form.get('query')
    k = int(request.form.get('k', 3))  # Default to 3 if k is not provided

    return render_template('search_progress.html', 
                           owner=owner, 
                           repo=repo, 
                           token=token, 
                           query=user_query,
                           k=k)

@app.route('/stream_progress')
def stream_progress():
    owner = request.args.get('owner')
    repo = request.args.get('repo')
    token = request.args.get('token')
    user_query = request.args.get('query')
    k = int(request.args.get('k', 3))
    return Response(stream_with_context(process_github_issues(owner, repo, token, user_query, k)), 
                    mimetype='text/event-stream')

@app.route('/api/stats')
def get_stats():
    owner = request.args.get('owner')
    repo = request.args.get('repo')
    if owner and repo:
        try:
            stats = get_total_issues_and_prs(owner, repo)
            return jsonify(stats)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Missing parameters'}), 400

@app.route('/ai_chat', methods=['POST'])
def ai_chat():
    data = request.get_json()
    conversation = data.get('conversation', [])
    relevant_issues = data.get('relevant_issues', [])

    # Combine all relevant issue text
    context_text = "\n\n".join([issue.get('text', '') for issue in relevant_issues])

    # Build system instruction
    system_instruction = (
        "You are a conversational code query resolver expert. You have access to GitHub issues, "
        "including titles, bodies, and comments, and a conversation history between the user and "
        "yourself. When responding to user queries, determine if the provided GitHub context "
        "addresses the concerns raised. If the GitHub issues directly relate to the query, "
        "acknowledge this and use the context to formulate a helpful, concise, and accurate response. "
        "If the context does not directly answer the query, inform the user that the GitHub issues "
        "do not directly address their query and provide a solution using your own knowledge. "
        "Maintain a conversational tone throughout the interaction to ensure clear and engaging "
        "communication."
    )

    # Reconstruct the conversation
    conversation_text = ""
    for msg in conversation:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            conversation_text += f"User: {content}\n"
        elif role == "assistant":
            conversation_text += f"Assistant: {content}\n"

    # Final prompt
    full_prompt = (
        f"{system_instruction}\n\n"
        f"GitHub Context:\n{context_text}\n\n"
        f"Conversation so far:\n{conversation_text}\n"
        f"Assistant:"
    )

    try:
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        response = gemini_model.generate_content(full_prompt)
        assistant_message = response.text.strip()
        return jsonify({
            "assistant_message": assistant_message
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
