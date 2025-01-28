import requests

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

# import requests
# import os

# GITHUB_TOKEN = "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"

# def get_total_issues_and_prs(owner: str, repo: str, token = GITHUB_TOKEN) -> dict:
#     """
#     Fetch the total number of issues and pull requests in a GitHub repository.

#     Args:
#         owner (str): The GitHub username of the owner of the repository.
#         repo (str): The name of the repository.
#         token (str, optional): GitHub personal access token. If not provided, 
#                              will check GITHUB_TOKEN environment variable.

#     Returns:
#         dict: A dictionary containing the total number of non-PR issues and pull requests.
#     """
#     # Get token from environment variable if not provided
#     if not token:
#         token = os.getenv('GITHUB_TOKEN')
#         if not token:
#             print("Warning: No GitHub token provided. Rate limits will be restricted.")

#     base_url = f"https://api.github.com/repos/{owner}/{repo}"
#     headers = {
#         'Accept': 'application/vnd.github.v3+json',
#         'Authorization': f'token {token}' if token else None
#     }
#     # Remove None values from headers
#     headers = {k: v for k, v in headers.items() if v is not None}

#     try:
#         # First check rate limit
#         rate_limit_response = requests.get('https://api.github.com/rate_limit', headers=headers)
#         rate_limit_data = rate_limit_response.json()
        
#         remaining = rate_limit_data['resources']['core']['remaining']
        
#         if remaining == 0:
#             print("Rate limit exceeded")
#             return {
#                 'total_issues': 0,
#                 'total_issues(which are not PRs)': 0,
#                 'total_pull_requests': 0,
#                 'error': 'Rate limit exceeded'
#             }

#         # Get repository details
#         repo_response = requests.get(base_url, headers=headers)
#         repo_response.raise_for_status()

#         # Get total PRs (both open and closed)
#         prs_response = requests.get(
#             f"{base_url}/pulls?state=all&per_page=1",
#             headers=headers
#         )
#         prs_response.raise_for_status()
#         total_prs = int(prs_response.headers.get('X-Total-Count', 0))

#         # Get total issues including PRs (both open and closed)
#         issues_response = requests.get(
#             f"{base_url}/issues?state=all&per_page=1",
#             headers=headers
#         )
#         issues_response.raise_for_status()
#         total_issues = int(issues_response.headers.get('X-Total-Count', 0))

#         return {
#             'total_issues': total_issues,
#             'total_issues(which are not PRs)': total_issues - total_prs,
#             'total_pull_requests': total_prs,
#             'rate_limit_remaining': remaining
#         }

#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching data: {e}")
#         return {
#             'total_issues': 0,
#             'total_issues(which are not PRs)': 0,
#             'total_pull_requests': 0,
#             'error': str(e)
#         }

print(get_total_issues_and_prs("huggingface", "datasets"))