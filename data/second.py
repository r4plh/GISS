# github_issues_pipeline.py

import os
import math
import time
import json
import requests
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset, Dataset

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

    # # Get total count of pull requests (open and closed)
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

def fetch_issues(owner: str, repo: str, github_token: str, num_issues: int = 10_000) -> str:
    """
    Fetch all issues from a specified GitHub repository (up to num_issues) and
    return them as a JSON string (in memory).
    
    Args:
        owner (str): The username of the repository's owner.
        repo (str): The name of the repository.
        github_token (str): GitHub personal access token for authentication.
        num_issues (int): Maximum number of issues to fetch.
        
    Returns:
        str: A JSON string representing the list of issues.
    """
    all_issues = []
    per_page = 100  # Number of issues per page (max possible)
    num_pages = math.ceil(num_issues / per_page)
    base_url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    headers = {"Authorization": f"token {github_token}"}
    request_count = 0

    for page in tqdm(range(1, num_pages + 1), desc="Fetching issues"):
        # Check if near rate limit
        if request_count >= 5000:
            print("Reached GitHub rate limit. Sleeping for one hour...")
            time.sleep(60 * 60)  # Wait for rate-limit reset
            request_count = 0

        query = f"?page={page}&per_page={per_page}&state=all"
        response = requests.get(f"{base_url}{query}", headers=headers)
        request_count += 1

        # If we've already exhausted all issues, break
        if not response.json() or len(response.json()) == 0:
            break

        all_issues.extend(response.json())

        # If we fetched fewer than per_page issues, we likely reached the end
        if len(response.json()) < per_page:
            break

    # Convert list of dicts to JSON string
    json_data = json.dumps(all_issues, indent=4)
    print(f"Fetched {len(all_issues)} issues total from {owner}/{repo}.")
    return json_data


def filter_out_pull_requests(issues_json_str: str) -> str:
    """
    Filters out pull requests from the provided issues JSON string.
    Returns a JSON string of only 'issues' (no pull_request entries).
    
    Args:
        issues_json_str (str): JSON string of issues data.
    
    Returns:
        str: JSON string containing only issues (excluding PRs).
    """
    issues = json.loads(issues_json_str)

    filtered_issues = []
    for issue in issues:
        # Some repos might not have 'pull_request' at all; use get()
        # If the key does not exist or its value is None => it's a normal issue
        if issue.get("pull_request") is None:
            filtered_issues.append(issue)

    print(f"Filtered down to {len(filtered_issues)} issues (no pull requests).")
    return json.dumps(filtered_issues, indent=4)


def select_issue_fields(issues_json_str: str) -> str:
    """
    Reads a JSON string of issues, retains only the relevant fields,
    and returns a JSON string with selected fields.
    
    Args:
        issues_json_str (str): JSON string of GitHub issues.
    
    Returns:
        str: A JSON string with only selected fields for each issue.
    """
    issues = json.loads(issues_json_str)

    selected_issues = []
    for issue in issues:
        selected_data = {
            "html_url": issue.get("html_url", ""),
            "title": issue.get("title", ""),
            "comments": issue.get("comments", 0),
            "body": issue.get("body", ""),
            "number": issue.get("number", 0),
        }
        selected_issues.append(selected_data)

    print(f"Selected fields for {len(selected_issues)} issues.")
    return json.dumps(selected_issues, indent=4)


def enrich_issues_with_comments(owner: str, repo: str, issues_json_str: str, github_token: str) -> str:
    """
    For each issue in the JSON string, fetch its comments from GitHub,
    attach them to the issue object, and return the enriched JSON string.
    
    Args:
        owner (str): GitHub repository owner.
        repo (str): GitHub repository name.
        issues_json_str (str): JSON string of filtered issues.
        github_token (str): GitHub personal access token for API requests.
    
    Returns:
        str: JSON string with each issue enriched with a list of its comment bodies.
    """
    headers = {"Authorization": f"token {github_token}"}
    issues = json.loads(issues_json_str)

    def get_comments(issue_number):
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return [comment["body"] for comment in response.json()]
        else:
            return []

    for issue in tqdm(issues, desc="Enriching issues with comments"):
        issue_number = issue.get("number", None)
        if issue_number is not None:
            issue["comments"] = get_comments(issue_number)
        else:
            # If somehow there's no number, keep comments empty
            issue["comments"] = []

    print(f"Enriched {len(issues)} issues with their comments.")
    return json.dumps(issues, indent=4)


def process_comments(issues_json_str: str) -> str:
    """
    Takes a JSON string of issues with comments, filters out issues
    that have zero comments, explodes them so each comment is in its own row,
    and removes comments shorter than 15 words. Returns a JSON string
    of the processed data.

    Args:
        issues_json_str (str): JSON string of issues with comments.

    Returns:
        str: JSON string of the processed data (each row has a single comment).
    """
    # Convert JSON string to a Pandas DataFrame via Dataset
    issues_list = json.loads(issues_json_str)
    df = pd.DataFrame(issues_list)
    dataset = Dataset.from_pandas(df)


    dataset = dataset.filter(lambda x: len(x["comments"]) > 0)


    dataset.set_format("pandas")
    exploded_df = dataset[:].explode("comments", ignore_index=True)


    exploded_df = exploded_df[exploded_df["comments"].str.split().str.len() >= 15]


    processed_data = exploded_df.to_dict(orient="records")

    print(f"After exploding and filtering comments, we have {len(processed_data)} rows.")
    return json.dumps(processed_data, indent=4)


def concatenate_text_fields(issues_json_str: str) -> str:
    """
    Concatenates the 'title', 'body', and 'comments' fields into a single 'text' field
    (one per row). Returns a JSON string of the final data.

    Args:
        issues_json_str (str): JSON string of processed issues data.

    Returns:
        str: JSON string with a 'text' field that contains the concatenated text.
    """
    # Convert JSON string to a Pandas DataFrame via Dataset
    issues_list = json.loads(issues_json_str)
    df = pd.DataFrame(issues_list)
    dataset = Dataset.from_pandas(df)

    def concatenate_text(examples):
        title = examples["title"] if examples.get("title") else ""
        body = examples["body"] if examples.get("body") else ""
        comments = examples["comments"]
        # If 'comments' is a string, use it directly;
        # if it's a list, join into one string
        if isinstance(comments, list):
            comments = " ".join(comments)
        if comments is None:
            comments = ""
        # Concatenate
        concatenated_text = f"{title}\n{body}\n{comments}"
        return {"text": concatenated_text}

    # Map over the dataset
    transformed_dataset = dataset.map(concatenate_text)

    # Convert the transformed dataset to a list of dicts
    final_data = transformed_dataset.to_pandas().to_dict(orient="records")
    print(f"Final data has {len(final_data)} rows after concatenation.")
    return json.dumps(final_data, indent=4)


def run_full_pipeline(owner: str, repo: str, github_token: str, num_issues: int = 1000) -> str:
    """
    Runs the entire pipeline in memory:
      1) Fetch issues
      2) Filter out pull requests
      3) Select relevant fields
      4) Enrich with actual GitHub comments
      5) Explode and filter comments
      6) Concatenate text fields (title, body, comments)
    Returns the final JSON string.
    """

    # 1) Fetch issues
    issues_data = fetch_issues(owner, repo, github_token, num_issues=num_issues)

    # 2) Filter out pull requests
    filtered_issues = filter_out_pull_requests(issues_data)

    # 3) Select relevant fields
    selected_issues = select_issue_fields(filtered_issues)

    # 4) Enrich with actual comments
    enriched_issues = enrich_issues_with_comments(owner, repo, selected_issues, github_token)

    # 5) Explode and filter comments
    processed_comments_issues = process_comments(enriched_issues)

    # 6) Concatenate text fields (title, body, comments)
    final_concatenated_data = concatenate_text_fields(processed_comments_issues)

    # The pipeline returns the final JSON string
    print("Pipeline complete. Returning final concatenated data.")
    return final_concatenated_data


if __name__ == "__main__":
    """
    Example usage:
      python github_issues_pipeline.py
    Make sure to set your GITHUB_TOKEN or pass it in. This is just an example:
    """
    # Example input
    OWNER = "huggingface"
    REPO = "transformers"
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN","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")
    NUM_ISSUES = 200  # for demonstration, can be bigger

    # Run the pipeline
    final_data_str = run_full_pipeline(OWNER, REPO, GITHUB_TOKEN, num_issues=NUM_ISSUES)

    # Optionally, you can save or process final_data_str further
    with open("final_concatenated_issues.json", "w", encoding="utf-8") as f:
        f.write(final_data_str)
