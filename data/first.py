import requests
import time
import math
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from datasets import load_dataset, Features, Value, ClassLabel
from dotenv import load_dotenv
import os

load_dotenv() 

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
headers = {"Authorization": f"token {GITHUB_TOKEN}"}

def fetch_issues(owner: str, repo: str, num_issues: int = 10_000, issues_path: Path = Path(".")) -> Path:
    """
    Fetches all issues from a specified GitHub repository and saves them as a JSONL file in the specified directory.
    Manages the GitHub API rate limit by pausing after 5000 API requests.

    Args:
        owner (str): The username of the repository's owner.
        repo (str): The name of the repository.
        num_issues (int): Maximum number of issues to fetch.
        issues_path (Path): The directory path where the issues file will be saved.

    Returns:
        Path: The path to the saved dataset file.
    """
    if not issues_path.is_dir():
        issues_path.mkdir(exist_ok=True)

    all_issues = []
    per_page = 100  # Number of issues to return per page
    num_pages = math.ceil(num_issues / per_page)
    base_url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    headers = {"Authorization": f"token {os.getenv('GITHUB_TOKEN')}"}
    request_count = 0

    for page in tqdm(range(1, num_pages + 1)):
        print(request_count)
        if request_count >= 5000:
            print(f"Reached GitHub rate limit. Sleeping for one hour...")
            time.sleep(60 * 60)  # Wait for rate limit reset
            request_count = 0  # Reset request count after sleeping

        query = f"?page={page}&per_page={per_page}&state=all"
        issues = requests.get(f"{base_url}{query}", headers=headers)
        print(issues)
        request_count += 1

        all_issues.extend(issues.json())
        print(len(all_issues))


    dataset_path = issues_path / f"{repo}-issues.json"
    df = pd.DataFrame.from_records(all_issues)
    df.to_json(dataset_path, orient="records" , indent=4)
    print(f"Downloaded all the issues for {repo}! Dataset stored at {dataset_path}")

    # Convert DataFrame to JSON string
    json_data = df.to_json(orient="records" , indent=4)
    return json_data  # Return JSON string

# fetch_issues("huggingface","datasets")

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

def process_issues_data(json_data: str) -> str:
    """
    Processes a JSON string containing GitHub issues, filtering out pull requests,
    and returns a new JSON string with issues that are not pull requests.

    Args:
        json_data (str): A JSON string containing GitHub issues data.

    Returns:
        str: A JSON string containing filtered GitHub issues that are not pull requests.
    """
    # Convert the JSON string to a list of dictionaries
    issues = json.loads(json_data)
    
    # Initialize a list to hold filtered issues
    filtered_issues = []
    
    # Initialize a counter to track number of issues filtered
    count = 0

    # Iterate over each issue in the original data
    for issue in issues:
        # Check if 'pull_request' key is not present in the issue
        if 'pull_request' not in issue:
            filtered_issues.append(issue)
            count += 1

    print(f'Filtered {count} issues that are not pull requests.')

    # Convert the filtered issues back to a JSON string
    output_json = json.dumps(filtered_issues, indent=None, separators=(',', ':'))

    # Save the output JSON to a file in the current directory
    with open('filtered_fields_issues.jsonl', 'w') as file:
        file.write(output_json)     
    
    return output_json

def select_issue_fields(json_data: str) -> str:
    """
    Filters the JSON string of GitHub issues to select specific fields needed for a search engine.

    Args:
        json_data (str): A JSON string containing GitHub issues data.

    Returns:
        str: A JSON string with only the selected fields of each issue.
    """
    # Convert the JSON string to a list of dictionaries
    issues = json.loads(json_data)

    # List to store the selected fields from each issue
    selected_issues = []

    # Iterate over each issue in the list
    for issue in issues:
        # Extract only the selected fields
        selected_data = {
            'html_url': issue.get('html_url', ''),
            'title': issue.get('title', ''),
            'comments': issue.get('comments', 0),
            'body': issue.get('body', ''),
            'number': issue.get('number', 0)
        }
        selected_issues.append(selected_data)

    # Save the output JSON to a file in the current directory
    with open('selected_fields_issues.jsonl', 'w') as file:
        file.write(output_json)        

    # Convert the list of selected fields back to a JSON string
    output_json = json.dumps(selected_issues, indent=None, separators=(',', ':'))
    
    return output_json

def process_issues_with_comments(json_data: str, github_token: str) -> Dataset:
    """
    Processes GitHub issues data by fetching comments, filtering and selecting data for further analysis.

    Args:
        json_data (str): A JSON string containing GitHub issues data.
        github_token (str): GitHub personal access token for API requests.

    Returns:
        Dataset: The final Hugging Face dataset with processed and filtered issues.
    """
    # Setup for API requests
    headers = {"Authorization": f"token {github_token}"}

    # Save the input JSON data to a file to use with load_dataset
    input_file = 'selected_fields_issues.jsonl'
    with open(input_file, 'w') as file:
        file.write(json_data)

    # Load the dataset from the saved JSONL file
    issues_dataset = load_dataset("json", data_files=input_file, split="train")

    # Function to fetch comments for each issue using the GitHub API
    def get_comments(issue_number):
        url = f"https://api.github.com/repos/huggingface/datasets/issues/{issue_number}/comments"
        response = requests.get(url, headers=headers)
        return [r["body"] for r in response.json()]

    # Map the function to fetch comments
    issues_dataset = issues_dataset.map(
        lambda x: {"comments": get_comments(x["number"])}
    )

    # Filter issues to keep only those with comments and then explode on comments
    issues_dataset = issues_dataset.filter(lambda x: len(x["comments"]) > 0)
    issues_dataset.set_format("pandas")
    df = issues_dataset[:]
    comments_df = df.explode("comments", ignore_index=True)

    # Convert back to dataset and map a new column for comment length
    comments_dataset = Dataset.from_pandas(comments_df)
    comments_dataset = comments_dataset.map(
        lambda x: {"comment_length": len(x["comments"].split())}
    )

    # Filter out short comments
    comments_dataset = comments_dataset.filter(lambda x: x["comment_length"] > 15)

    # Save the final dataset
    final_output_file = 'final_issues_with_comments.jsonl'
    comments_dataset.to_json(final_output_file)
    print(f"Final dataset saved at {final_output_file}")

    return comments_dataset

def concatenate_and_save_text_dataset(dataset: Dataset, output_path: str) -> Dataset:
    """
    Concatenates the title, body, and comments of GitHub issues into a single text field, saves the modified dataset to a local file, and returns the modified dataset.

    Args:
        dataset (Dataset): A Hugging Face dataset containing entries with 'title', 'body', and 'comments' fields.
        output_path (str): The file path where the dataset should be saved.

    Returns:
        Dataset: The dataset with an additional field 'text' that combines the title, body, and comments.
    """
    def concatenate_text(examples):
        # Ensure that title and body are non-empty strings if they are None
        title = examples["title"] if examples["title"] is not None else ""
        body = examples["body"] if examples["body"] is not None else ""
        # Comments could be a list, convert it to a single string
        comments = ' '.join(examples["comments"]) if isinstance(examples["comments"], list) else examples["comments"]
        # Concatenate title, body, and comments with newlines between them
        concatenated_text = title + " \n " + body + " \n " + comments
        return {"text": concatenated_text}

    # Apply the concatenate_text function to each entry in the dataset
    transformed_dataset = dataset.map(concatenate_text)

    # Save the transformed dataset to a JSONL file specified by output_path
    transformed_dataset.to_json(output_path)
    print(f"Dataset saved at {output_path}")

    return transformed_dataset

# https://api.github.com/repos/huggingface/datasets/issues?page=1&per_page=100&state=all
# https://api.github.com/repos/huggingface/datasets/issues?page=1&per_page=100&state=all

def process_issues_data(input_path: Path, output_path: Path = Path("filtered_issues.json")) -> None:
    """
    Reads a JSON file containing GitHub issues, filters out pull requests, and saves a new JSON file
    with issues that are not pull requests at the specified output path.

    Args:
        input_path (Path): The file path of the JSON file containing GitHub issues data.
        output_path (Path): The file path where the filtered issues will be saved.

    Returns:
        None: The function writes output directly to a file and returns nothing.
    """
    # Read the JSON data from the file
    with open(input_path, 'r') as file:
        issues = json.load(file)

    # print(type(issues),type(issues[0]))
    # print(issues[0])
    
    # Initialize a list to hold filtered issues
    filtered_issues = []
    
    # Initialize a counter to track number of issues filtered
    count = 0

    # Iterate over each issue in the original data
    for issue in issues:
        # Check if 'pull_request' key is not present in the issue
        if issue['pull_request']== None:
            filtered_issues.append(issue)
            count += 1

    print(f'Filtered {count} issues that are not pull requests.')

    # Write the filtered issues back to a new JSON file
    with open(output_path, 'w') as file:
        json.dump(filtered_issues, file, indent=4)

    output_json = json.dumps(filtered_issues , indent=4)
    print(f"Filtered issues saved to {output_path}")
    return filtered_issues

# process_issues_data("/Users/0xr4plh/Documents/Machine Learning/GitHub-Issues-Semantic-Search/datasets-issues.json")

def select_issue_fields(file_path: str) -> str:
    """
    Reads a JSON file of GitHub issues, filters to select specific fields needed for a search engine,
    and saves the filtered data into a new JSON file.

    Args:
        file_path (str): The file path of the JSON file containing GitHub issues data.

    Returns:
        str: A JSON string with only the selected fields of each issue.
    """
    # Read the JSON file
    with open(file_path, 'r') as file:
        issues = json.load(file)

    # List to store the selected fields from each issue
    selected_issues = []

    # Iterate over each issue in the list
    for issue in issues:
        # Extract only the selected fields
        selected_data = {
            'html_url': issue.get('html_url', ''),
            'title': issue.get('title', ''),
            'comments': issue.get('comments', 0),
            'body': issue.get('body', ''),
            'number': issue.get('number', 0)
        }
        selected_issues.append(selected_data)

    # Convert the list of selected fields back to a JSON string
    output_json = json.dumps(selected_issues , indent=4)

    # Save the output JSON to a file in the current directory
    with open('selected_fields_issues.json', 'w') as file:
        file.write(output_json)

    return output_json

# select_issue_fields("filtered_issues.json")

def process_issues(owner: str, repo: str, input_json_path: str, github_token: str) -> None:
    """
    Processes GitHub issues by fetching comments for each issue, enriching the data,
    and saving the processed issues back into a JSON file.

    Args:
        owner (str): GitHub repository owner.
        repo (str): GitHub repository name.
        input_json_path (str): Path to the input JSON file containing initial issues data.
        github_token (str): GitHub personal access token for API requests.
    """
    headers = {"Authorization": f"token {github_token}"}

    # Load issues from the input JSON file
    with open(input_json_path, 'r') as file:
        issues = json.load(file)

    # Function to fetch comments for each issue using the GitHub API
    def get_comments(issue_number):
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return [comment["body"] for comment in response.json()]
        else:
            return []

    # Enrich each issue with comments, showing progress with tqdm
    for issue in tqdm(issues, desc="Processing issues"):
        issue['comments'] = get_comments(issue['number'])

    # Save the enriched issues back to a new JSON file
    output_path = 'enriched_issues.json'
    with open(output_path, 'w') as file:
        json.dump(issues, file, indent=4)

    print(f"Enriched issues saved at {output_path}")

    json_data = json.dumps(issues , indent = 4)

    return json_data


# process_issues("huggingface","datasets", "selected_fields_issues.json","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")

def process_comments(input_json_path: str) -> str:
    """
    Processes a JSON file with GitHub issues data by filtering out issues with no comments,
    exploding the comments into individual rows, filtering comments shorter than 15 words,
    and saving the processed data back into a JSON file.

    Args:
        input_json_path (str): Path to the JSON file containing issues data.

    Returns:
        str: Path to the output JSON file containing processed data.
    """
    # Load the dataset from the JSON file
    dataset = load_dataset("json", data_files=input_json_path, split="train")

    # Filter out entries with empty comment lists
    dataset = dataset.filter(lambda x: len(x["comments"]) > 0)

    # Explode the dataset based on the comments
    dataset.set_format("pandas")
    df = dataset[:]
    comments_df = df.explode("comments", ignore_index=True)

    # Filter out comments with less than 15 words
    comments_df = comments_df[comments_df["comments"].str.split().str.len() >= 15]

    # Convert the DataFrame back to a Dataset to utilize Dataset methods if needed
    processed_dataset = Dataset.from_pandas(comments_df)

    # Save the final dataset to a JSON file
    final_output_file = "processed_comments.json"
    with open(final_output_file, "w", encoding="utf-8") as file:
        json.dump(comments_df.to_dict(orient="records"), file, indent=4)

    print(f"Processed dataset saved at {final_output_file}")
    return final_output_file


# process_comments("enriched_issues.json")

def concatenate_and_save_issues(input_json_path: str):
    """
    Processes a JSON file to concatenate the title, body, and comments of GitHub issues,
    then saves the transformed data to a predefined JSON file.

    Args:
        input_json_path (str): Path to the JSON file containing issues data.
    """
    # Load the dataset from the JSON file
    dataset = load_dataset("json", data_files=input_json_path, split="train")

    def concatenate_text(examples):
        # Ensure that title and body are non-empty strings if they are None
        title = examples["title"] if examples["title"] is not None else ""
        body = examples["body"] if examples["body"] is not None else ""
        # Comments could be a list, convert it to a single string
        comments = ' '.join(examples["comments"]) if isinstance(examples["comments"], list) else examples["comments"]
        # Concatenate title, body, and comments with newlines between them
        concatenated_text = f"{title} \n {body} \n {comments}"
        return {"text": concatenated_text}

    # Apply the concatenate_text function to each entry in the dataset
    transformed_dataset = dataset.map(concatenate_text)

    # Convert the transformed dataset to pandas DataFrame for easier JSON handling
    json_data = transformed_dataset.to_pandas().to_dict(orient='records')

    # Define the output path within the function
    output_path = "concatenated_issues.json"

    # Check if the directory for the output path exists, and create it if it does not
    output_directory = os.path.dirname(output_path)
    if output_directory:  # If the directory part is non-empty, create it
        os.makedirs(output_directory, exist_ok=True)

    # Manually save the transformed dataset to a JSON file specified by output_path
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4)

    print(f"Dataset saved at {output_path}")

    return transformed_dataset

# This call would execute the function using your specified input JSON path.
concatenate_and_save_issues("processed_comments.json")