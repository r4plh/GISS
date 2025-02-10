# GitHub Issue Semantic Search Engine

Welcome! This is my personal GitHub Issue Semantic Search Engine project. The main idea here is to allow anyone to come to the website, type in the owner and repository name, and then fetch all the GitHub issues from that repo. Once the issues are pulled locally, a semantic search is performed so that users can get results based on the meaning of their query—not just simple keyword matching. This is extremely helpful for those times when you type a question in your own words, and the normal GitHub search doesn’t pull up the right issue because it’s only looking for exact keywords.

## Why This Project?

- **Semantic Search vs. Keyword Search**: The GitHub search bar is limited to keyword-based queries. Often, you might have a query like “Where is the Spanish translation of the course?” but no exact keyword in the issues matches “Spanish translation,” so you never find that relevant issue. My solution is to use embeddings and a FAISS index to provide RAG (Retrieval-Augmented Generation) style semantic search.
- **Generalized for Any Repo**: All you have to do is type in the GitHub username/organization (the “owner”) and the repository name, for instance `owner = "huggingface"` and `repo_name = "transformers"`. 
- **Caching**: If the same repo was already queried, we skip the data-fetch step and go directly to the search. That means faster results the second time around!
- **Query Resolver Bot**: Once you see the top issues, there’s also a chatbot powered by the Gemini 1.5 flash model from Google PaLM. It’s given context about the repo issues and your previous query, so it can answer follow-up questions in a conversational manner.

## How It Works

### 1. Getting Issues from GitHub

1. **User Input**: Owner name, repo name, an optional GitHub token (if needed), the user’s query, and how many results (`k`) to return.
2. **Fetch & Filter**: 
   - Pull up to `num_issues` issues from the GitHub API.  
   - Filter out pull requests because the GitHub REST API lumps them into “issues.”
   - Keep only real issues with relevant text.
3. **Extract Comments**: Gather all comments for each issue (if any). We remove short comments (fewer than 15 words) so that only more substantial, informative comments remain.
4. **Concatenate Text**: Create a single `text` field that is a combination of title + body + comment.  

### 2. Generating Embeddings

- We use `sentence-transformers/multi-qa-mpnet-base-dot-v1` to encode each `text` into an embedding vector.
- These embeddings are stored in a NumPy array (`embeddings.npy`), and a FAISS index is created and saved (`_index.faiss`).

### 3. Semantic Search

- When you type a query, it gets encoded into an embedding using the same model.
- We perform a similarity search against the FAISS index to get the top `k` most relevant issues.
- We show you the results, including their titles, bodies, and a direct link (HTML URL) to each issue on GitHub.

### 4. Query Resolver Bot

- Alongside the search results, we provide a chatbot. 
- This bot gets:
  1. The most recent user query.
  2. The top `k` relevant GitHub issues returned from the semantic search.
- The bot is then instructed (via a system prompt) to do the following:
  - Check if the GitHub context addresses the user’s query.
  - If yes, reference that context to answer in a concise, accurate way.
  - If not, politely say that the issues don’t fully answer the question and use its own knowledge base.
- This is powered by the Gemini 1.5 flash model

  ```
  "You are a conversational code query resolver expert. You have access to GitHub issues, 
  including titles, bodies, and comments, and a conversation history between the user and 
  yourself. When responding to user queries, determine if the provided GitHub context 
  addresses the concerns raised. If the GitHub issues directly relate to the query, 
  acknowledge this and use the context to formulate a helpful, concise, and accurate response. 
  If the context does not directly answer the query, inform the user that the GitHub issues 
  do not directly address their query and provide a solution using your own knowledge. 
  Maintain a conversational tone throughout the interaction to ensure clear and engaging 
  communication."
  ```

## Project Structure

Here’s a quick overview of how the project code is laid out:

```
.
├── app.py
├── search.py
├── data
│   └── dataPipeline.py
├── templates
│   ├── index.html
│   ├── search_progress.html
│
├── git-repos
    └── (cached JSON, embeddings.npy, and FAISS files go here)
```

- **app.py**: Main Flask application. This is where routes are defined, user input is handled, and streaming search results happen.
- **data/dataPipeline.py**: Contains a class called `GitHubIssuesPipeline` that deals with fetching issues, filtering out PRs, exploding comments, and more.
- **git-repos**: A local caching directory where JSON, embeddings, and FAISS index files are stored to avoid re-fetching the same repo repeatedly.

## Installation & Running Locally

1. **Clone the repository** and `cd` into it.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Make sure you have `faiss-cpu`, `sentence-transformers`, `flask`, `pandas`, `numpy`, `tqdm`, `requests`, `datasets`, and `google-generativeai` installed.
   
3. **Set Up Your Credentials**:
   - If you plan to fetch public repo issues without authentication, you might get rate-limited. So you can pass a `GITHUB_TOKEN` as well.  
   - For the Gemini 1.5 flash model, configure your Google API key:  
     ```python
     genai.configure(api_key="YOUR_GOOGLE_PALM_API_KEY")
     ```
4. **Run the Application**:
   ```bash
   python app.py
   ```
   By default, Flask starts on [http://127.0.0.1:5000/](http://127.0.0.1:5000/).

5. **Open the Website** in your browser at [http://127.0.0.1:5000/](http://127.0.0.1:5000/). You’ll see a form where you can enter the owner, repo name, your GitHub token (if needed), your query, and how many results (`k`) you want to see.

## Usage Flow

1. **Enter Repository Details**: In the form on the homepage (`index.html`), type the `owner`, `repo`, `token` (optional, but recommended for higher rate limits), your `query`, and how many results to return.
   
2. **Either**:
   - **New Repo**: If this is the first time searching a repo, the app will:
     - Fetch issues and comments.
     - Filter out PRs.
     - Build and save embeddings.
     - Create and save a FAISS index.
     - Return top `k` results relevant to your query.
   - **Already Cached**: If the repo was searched previously, it just loads data from the cache in `git-repos/`, saving you time.
   
3. **View Results**: You’ll see the top `k` issues with:
   - **Title**  
   - **Body**  
   - **Direct link (html_url)**  

4. **Additional Features**:
   - **Search Again**: On the same page, you can type a new query (maybe change `k` too) to refine or ask a different question on the same repo.
   - **Query Resolver Bot**: Right below, there’s a chatbot that has context of:
     - The current query you asked.
     - The top `k` issues returned.  
     This bot references them to answer your follow-up questions. If the issues themselves don’t solve your query, it will say so and try to help with its own knowledge.

## Key Code Snippets

### Core Search Logic

```python
@app.route('/search', methods=['POST'])
def search():
    owner = request.form.get('owner')
    repo = request.form.get('repo')
    token = request.form.get('token')
    user_query = request.form.get('query')
    k = int(request.form.get('k', 3))  # Default to 3 if none provided

    return render_template('search_progress.html', 
                           owner=owner, 
                           repo=repo, 
                           token=token, 
                           query=user_query,
                           k=k)
```

Here we grab the owner, repo, token, the user’s query, and k from the HTML form, then render a progress page that streams search updates using `stream_progress`.

### The Pipeline

```python
class GitHubIssuesPipeline:
    def __init__(self, owner: str, repo: str, github_token: str, num_issues: int = 1000):
        self.owner = owner
        self.repo = repo
        self.github_token = github_token
        self.num_issues = num_issues

    def run_full_pipeline(self) -> str:
        # 1) Fetch issues
        issues_data = self.fetch_issues()
        # 2) Filter out pull requests
        filtered_issues = self.filter_out_pull_requests(issues_data)
        # 3) Select relevant fields
        selected_issues = self.select_issue_fields(filtered_issues)
        # 4) Enrich with GitHub comments
        enriched_issues = self.enrich_issues_with_comments(selected_issues)
        # 5) Explode/filter comments
        processed_comments_issues = self.process_comments(enriched_issues)
        # 6) Concatenate text fields
        final_concatenated_data = self.concatenate_text_fields(processed_comments_issues)
        return final_concatenated_data
```

It’s step-by-step: first fetch, then filter out PRs, then only keep relevant fields, pull and attach comments, explode them out, filter short ones, and finally create that single `text` field.

### One Sample Row

After processing, here’s an example JSON entry (a single “row”) in the final data:

```json
{
    "html_url": "https://github.com/huggingface/datasets/issues/7371",
    "title": "500 Server error with pushing a dataset",
    "comments": "EDIT: seems to be all good now. I'll add a comment if the error happens again ...",
    "body": "### Describe the bug\n\nSuddenly, I started getting this error message saying ...",
    "number": 7371,
    "text": "500 Server error with pushing a dataset\n### Describe the bug ... (title + body + comments combined)"
}
```

This `text` field is what we embed using `sentence-transformers/multi-qa-mpnet-base-dot-v1`.

## Requirements & Notes

- **GitHub API Rate Limits**: Without a token, you quickly hit rate limits. So it’s best to provide a personal token.
- **FAISS**: We are using FAISS for vector similarity search. Make sure it’s installed properly.
- **Google PaLM API Key**: For the chatbot to work, you need to configure your Google PaLM API key in `genai.configure(api_key="YOUR_KEY")`.
- **Local Environment**: Right now, this is not deployed anywhere (just my local environment). If you want to deploy, you’ll have to set up a server environment with this code.

## Final Thoughts

That’s pretty much the whole flow. **Everything here is free** (though obviously you need your own API keys for GitHub and Google PaLM if you want to use them). Feel free to modify the code to suit your needs!

If you have any questions or find any bugs, just open an issue or a pull request. Hope this helps you in exploring GitHub issues in a more semantic, user-friendly way!




