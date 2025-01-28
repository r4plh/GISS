import math
import time
import requests
import json
from tqdm import tqdm
import pandas as pd
from datasets import Dataset
from datetime import datetime

class GitHubIssuesPipeline:
    def __init__(self, owner: str, repo: str, github_token: str, num_issues: int = 1000):
        """
        Initialize pipeline parameters.
        
        Args:
            owner (str): GitHub repository owner/user.
            repo (str): GitHub repository name.
            github_token (str): GitHub personal access token for authentication.
            num_issues (int): Maximum number of issues to fetch.
        """
        self.owner = owner
        self.repo = repo
        self.github_token = github_token
        self.num_issues = num_issues
        self.start_time = None

    def _log_progress(self, message: str, count: int = None, total: int = None):
        """Helper method to log progress with timestamps"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if count is not None and total is not None:
            return f"[{timestamp}] {message} ({count}/{total})"
        return f"[{timestamp}] {message}"

    def fetch_issues(self) -> str:
        """
        Fetch all issues (up to num_issues) from the repo and return them as a JSON string.
        """
        self.start_time = time.time()
        all_issues = []
        per_page = 100
        num_pages = math.ceil(self.num_issues / per_page)
        base_url = f"https://api.github.com/repos/{self.owner}/{self.repo}/issues"
        headers = {"Authorization": f"token {self.github_token}"}
        request_count = 0
        
        print(self._log_progress(f"Starting to fetch issues from {self.owner}/{self.repo}"))
        progress_bar = tqdm(total=self.num_issues, desc="Fetching issues")

        for page in range(1, num_pages + 1):
            if request_count >= 5000:
                print(self._log_progress("Reached GitHub rate limit. Sleeping for one hour..."))
                time.sleep(60 * 60)
                request_count = 0

            query = f"?page={page}&per_page={per_page}&state=all"
            response = requests.get(f"{base_url}{query}", headers=headers)
            request_count += 1

            if not response.json() or len(response.json()) == 0:
                break

            new_issues = response.json()
            all_issues.extend(new_issues)
            progress_bar.update(len(new_issues))

            if len(new_issues) < per_page:
                break

        progress_bar.close()
        elapsed = time.time() - self.start_time
        json_data = json.dumps(all_issues, indent=4)
        print(self._log_progress(f"Fetched {len(all_issues)} issues in {elapsed:.2f} seconds"))
        return json_data

    def filter_out_pull_requests(self, issues_json_str: str) -> str:
        """
        Filter out pull requests from the provided issues JSON string.
        """
        start = time.time()
        issues = json.loads(issues_json_str)
        total = len(issues)
        
        print(self._log_progress("Starting PR filtering", 0, total))
        progress_bar = tqdm(total=total, desc="Filtering PRs")
        
        filtered_issues = []
        for issue in issues:
            if issue.get("pull_request") is None:
                filtered_issues.append(issue)
            progress_bar.update(1)
        
        progress_bar.close()
        elapsed = time.time() - start
        print(self._log_progress(f"Filtered to {len(filtered_issues)} issues (removed {total - len(filtered_issues)} PRs) in {elapsed:.2f} seconds"))
        return json.dumps(filtered_issues, indent=4)

    def select_issue_fields(self, issues_json_str: str) -> str:
        """
        Retain only relevant fields from each issue (html_url, title, comments, body, number).
        """
        start = time.time()
        issues = json.loads(issues_json_str)
        total = len(issues)
        
        print(self._log_progress("Starting field selection", 0, total))
        progress_bar = tqdm(total=total, desc="Selecting fields")
        
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
            progress_bar.update(1)
        
        progress_bar.close()
        elapsed = time.time() - start
        print(self._log_progress(f"Selected fields for {len(selected_issues)} issues in {elapsed:.2f} seconds"))
        return json.dumps(selected_issues, indent=4)

    def enrich_issues_with_comments(self, issues_json_str: str) -> str:
        """
        Fetch and attach comments for each issue. The 'comments' field becomes a list of comment bodies.
        """
        start = time.time()
        headers = {"Authorization": f"token {self.github_token}"}
        issues = json.loads(issues_json_str)
        total = len(issues)
        
        print(self._log_progress("Starting comment enrichment", 0, total))
        progress_bar = tqdm(total=total, desc="Enriching with comments")
        comments_fetched = 0

        def get_comments(issue_number):
            nonlocal comments_fetched
            url = f"https://api.github.com/repos/{self.owner}/{self.repo}/issues/{issue_number}/comments"
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                comments = [comment["body"] for comment in response.json()]
                comments_fetched += len(comments)
                return comments
            return []

        for issue in issues:
            issue_number = issue.get("number")
            if issue_number is not None:
                issue["comments"] = get_comments(issue_number)
            else:
                issue["comments"] = []
            progress_bar.update(1)

        progress_bar.close()
        elapsed = time.time() - start
        print(self._log_progress(f"Enriched {total} issues with {comments_fetched} comments in {elapsed:.2f} seconds"))
        return json.dumps(issues, indent=4)

    def process_comments(self, issues_json_str: str) -> str:
        """
        1) Filter out issues with zero comments.
        2) Explode each comment into its own row.
        3) Filter comments shorter than 15 words.
        """
        start = time.time()
        issues_list = json.loads(issues_json_str)
        print(self._log_progress("Processing comments"))
        
        df = pd.DataFrame(issues_list)
        dataset = Dataset.from_pandas(df)
        initial_count = len(dataset)

        # Keep only issues that have at least 1 comment
        dataset = dataset.filter(lambda x: len(x["comments"]) > 0)
        with_comments_count = len(dataset)

        # Convert to pandas and explode
        dataset.set_format("pandas")
        exploded_df = dataset[:].explode("comments", ignore_index=True)
        exploded_count = len(exploded_df)

        # Filter comments < 15 words
        exploded_df = exploded_df[exploded_df["comments"].str.split().str.len() >= 15]
        final_count = len(exploded_df)

        processed_data = exploded_df.to_dict(orient="records")
        elapsed = time.time() - start
        print(self._log_progress(
            f"Comment processing complete:\n"
            f"  - Initial issues: {initial_count}\n"
            f"  - Issues with comments: {with_comments_count}\n"
            f"  - Total comments: {exploded_count}\n"
            f"  - Comments ≥15 words: {final_count}\n"
            f"  - Time taken: {elapsed:.2f} seconds"
        ))
        return json.dumps(processed_data, indent=4)

    def concatenate_text_fields(self, issues_json_str: str) -> str:
        """
        Concatenate 'title', 'body', and 'comments' into a single 'text' field.
        """
        start = time.time()
        issues_list = json.loads(issues_json_str)
        total = len(issues_list)
        
        print(self._log_progress("Starting text concatenation", 0, total))
        progress_bar = tqdm(total=total, desc="Concatenating fields")
        
        df = pd.DataFrame(issues_list)
        dataset = Dataset.from_pandas(df)

        def concatenate_text(examples):
            title = examples.get("title", "")
            body = examples.get("body", "")
            comments = examples["comments"]
            if isinstance(comments, list):
                comments = " ".join(comments)
            comments = comments if comments else ""
            concatenated_text = f"{title}\n{body}\n{comments}"
            return {"text": concatenated_text}

        transformed_dataset = dataset.map(concatenate_text)
        final_data = transformed_dataset.to_pandas().to_dict(orient="records")
        
        progress_bar.update(total)  # Complete the progress bar
        progress_bar.close()
        
        elapsed = time.time() - start
        print(self._log_progress(f"Concatenation complete: {len(final_data)} records processed in {elapsed:.2f} seconds"))
        return json.dumps(final_data, indent=4)

    def run_full_pipeline(self) -> str:
        """
        Runs the entire pipeline and returns the final JSON string, suitable for embedding.
        """
        pipeline_start = time.time()
        print(self._log_progress("Starting full pipeline"))

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

        # 6) Concatenate text fields (title, body, comments)
        final_concatenated_data = self.concatenate_text_fields(processed_comments_issues)

        total_elapsed = time.time() - pipeline_start
        print(self._log_progress(f"Pipeline complete. Total time: {total_elapsed:.2f} seconds"))
        return final_concatenated_data

# Example usage in a main script:
if __name__ == "__main__":
    owner = "huggingface"
    repo = "datasets"
    token = "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"
    pipeline = GitHubIssuesPipeline(owner, repo, token, num_issues=100)

    final_json_str = pipeline.run_full_pipeline()
    with open("final_output.json", "w", encoding="utf-8") as f: f.write(final_json_str)
    # `final_json_str` now contains the concatenated text for each row.
    # Next steps: embed `final_json_str` text column using your favorite embedding library or model.
