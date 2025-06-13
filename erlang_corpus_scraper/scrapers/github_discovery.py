"""
GitHub repository discovery for Erlang corpus scraper.
Discovers high-quality Erlang repositories using GitHub API.
"""

import requests
import time
import json
import logging
from typing import List, Dict, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

# Import our config (assumes config.py is in parent directory)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    GITHUB_API_BASE, GITHUB_TOKEN, REQUESTS_PER_HOUR,
    REPO_DISCOVERY, SEED_REPOSITORIES, GITHUB_SEARCH_QUERIES,
    PROCESSING_LIMITS, ERROR_HANDLING, get_output_path
)

@dataclass
class RepositoryInfo:
    """Repository information structure."""
    name: str
    full_name: str
    description: str
    stars: int
    forks: int
    size_kb: int
    language: str
    languages: Dict[str, int]
    created_at: str
    updated_at: str
    clone_url: str
    html_url: str
    archived: bool
    has_wiki: bool
    has_issues: bool
    erlang_percentage: float
    quality_score: float

class GitHubAPIError(Exception):
    """Custom exception for GitHub API errors."""
    pass

class GitHubDiscovery:
    """Discovers Erlang repositories using GitHub API."""
    
    def __init__(self):
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
        
        # Set up authentication if token is available
        if GITHUB_TOKEN:
            self.session.headers.update({
                "Authorization": f"token {GITHUB_TOKEN}",
                "Accept": "application/vnd.github.v3+json"
            })
            self.logger.info("GitHub token configured")
        else:
            self.logger.warning("No GitHub token - rate limits will be restrictive")
            
        # Rate limiting
        self.requests_made = 0
        self.hour_start = time.time()
        
    def _rate_limit_check(self):
        """Check and enforce rate limiting."""
        current_time = time.time()
        
        # Reset counter every hour
        if current_time - self.hour_start > 3600:
            self.requests_made = 0
            self.hour_start = current_time
            
        # Check if we're approaching rate limit
        if self.requests_made >= REQUESTS_PER_HOUR * 0.9:  # 90% of limit
            sleep_time = 3600 - (current_time - self.hour_start)
            if sleep_time > 0:
                self.logger.warning(f"Rate limit approaching, sleeping for {sleep_time:.0f} seconds")
                time.sleep(sleep_time)
                self.requests_made = 0
                self.hour_start = time.time()
    
    def _make_request(self, url: str, params: Optional[Dict] = None) -> Dict:
        """Make rate-limited request to GitHub API."""
        self._rate_limit_check()
        
        for attempt in range(ERROR_HANDLING["max_retries"]):
            try:
                response = self.session.get(url, params=params)
                self.requests_made += 1
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 403:
                    # Rate limit exceeded
                    reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                    sleep_time = max(reset_time - int(time.time()), 60)
                    self.logger.warning(f"Rate limit exceeded, sleeping for {sleep_time} seconds")
                    time.sleep(sleep_time)
                    continue
                elif response.status_code == 404:
                    self.logger.warning(f"Repository not found: {url}")
                    return {}
                else:
                    response.raise_for_status()
                    
            except requests.RequestException as e:
                self.logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < ERROR_HANDLING["max_retries"] - 1:
                    time.sleep(ERROR_HANDLING["retry_delay_seconds"] * (attempt + 1))
                else:
                    raise GitHubAPIError(f"Failed to fetch {url} after {ERROR_HANDLING['max_retries']} attempts")
                    
        return {}
    
    def get_repository_info(self, repo_full_name: str) -> Optional[RepositoryInfo]:
        """Get detailed information about a repository."""
        self.logger.info(f"Fetching repository info: {repo_full_name}")
        
        # Get basic repository info
        repo_url = f"{GITHUB_API_BASE}/repos/{repo_full_name}"
        repo_data = self._make_request(repo_url)
        
        if not repo_data:
            return None
            
        # Get language breakdown
        languages_url = f"{GITHUB_API_BASE}/repos/{repo_full_name}/languages"
        languages_data = self._make_request(languages_url)
        
        # Calculate Erlang percentage
        total_bytes = sum(languages_data.values()) if languages_data else 0
        erlang_bytes = languages_data.get("Erlang", 0)
        erlang_percentage = (erlang_bytes / total_bytes) if total_bytes > 0 else 0
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(repo_data, languages_data)
        
        try:
            return RepositoryInfo(
                name=repo_data["name"],
                full_name=repo_data["full_name"],
                description=repo_data.get("description", "") or "",
                stars=repo_data["stargazers_count"],
                forks=repo_data["forks_count"],
                size_kb=repo_data["size"],
                language=repo_data.get("language", ""),
                languages=languages_data or {},
                created_at=repo_data["created_at"],
                updated_at=repo_data["updated_at"],
                clone_url=repo_data["clone_url"],
                html_url=repo_data["html_url"],
                archived=repo_data.get("archived", False),
                has_wiki=repo_data.get("has_wiki", False),
                has_issues=repo_data.get("has_issues", False),
                erlang_percentage=erlang_percentage,
                quality_score=quality_score
            )
        except KeyError as e:
            self.logger.error(f"Missing required field in repository data: {e}")
            return None
    
    def _calculate_quality_score(self, repo_data: Dict, languages_data: Dict) -> float:
        """Calculate a quality score for the repository."""
        score = 0.0
        
        # Stars (normalized to 0-40 points, log scale)
        stars = repo_data["stargazers_count"]
        if stars > 0:
            score += min(40, 10 * (stars ** 0.5) / 10)
        
        # Recent activity (0-20 points)
        updated_at = datetime.fromisoformat(repo_data["updated_at"].replace('Z', '+00:00'))
        days_since_update = (datetime.now(updated_at.tzinfo) - updated_at).days
        if days_since_update < 30:
            score += 20
        elif days_since_update < 90:
            score += 15
        elif days_since_update < 365:
            score += 10
        
        # Erlang percentage (0-20 points)
        total_bytes = sum(languages_data.values()) if languages_data else 0
        erlang_bytes = languages_data.get("Erlang", 0)
        if total_bytes > 0:
            erlang_pct = erlang_bytes / total_bytes
            score += 20 * erlang_pct
        
        # Repository features (0-10 points)
        if repo_data.get("has_wiki"):
            score += 3
        if repo_data.get("has_issues"):
            score += 3
        if repo_data.get("description"):
            score += 4
        
        # Size bonus/penalty (0-10 points)
        size_kb = repo_data["size"]
        if 100 <= size_kb <= 50000:  # Sweet spot
            score += 10
        elif size_kb < 100:
            score += 2  # Too small
        # Large repos don't get penalty but no bonus
        
        return min(100, score)
    
    def _meets_quality_criteria(self, repo_info: RepositoryInfo) -> bool:
        """Check if repository meets our quality criteria."""
        criteria = REPO_DISCOVERY
        
        # Basic criteria
        if repo_info.stars < criteria["min_stars"]:
            return False
            
        if repo_info.size_kb < criteria["min_size_kb"]:
            return False
            
        if repo_info.size_kb > criteria["max_size_mb"] * 1024:
            return False
            
        if repo_info.erlang_percentage < criteria["min_erlang_percentage"]:
            return False
            
        if criteria["exclude_forks"] and repo_info.forks > repo_info.stars * 2:
            # Likely a fork if it has way more forks than stars
            return False
            
        if repo_info.archived and not criteria["include_archived"]:
            return False
            
        # Recent activity check
        updated_at = datetime.fromisoformat(repo_info.updated_at.replace('Z', '+00:00'))
        months_ago = datetime.now(updated_at.tzinfo) - timedelta(days=criteria["recent_activity_months"] * 30)
        if updated_at < months_ago:
            return False
            
        return True
    
    def search_repositories(self, query: str, max_results: int = 100) -> List[str]:
        """Search for repositories using GitHub search API."""
        self.logger.info(f"Searching repositories: {query}")
        
        repo_names = []
        page = 1
        per_page = min(100, max_results)  # GitHub max is 100 per page
        
        while len(repo_names) < max_results:
            search_url = f"{GITHUB_API_BASE}/search/repositories"
            params = {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "page": page,
                "per_page": per_page
            }
            
            try:
                data = self._make_request(search_url, params)
                
                if not data or "items" not in data:
                    break
                    
                items = data["items"]
                if not items:
                    break
                    
                for item in items:
                    repo_names.append(item["full_name"])
                    if len(repo_names) >= max_results:
                        break
                        
                # Check if we have more pages
                if len(items) < per_page:
                    break
                    
                page += 1
                
            except GitHubAPIError as e:
                self.logger.error(f"Search failed: {e}")
                break
                
        self.logger.info(f"Found {len(repo_names)} repositories for query: {query}")
        return repo_names
    
    def discover_all_repositories(self) -> List[RepositoryInfo]:
        """Discover all repositories using seed list and search queries."""
        self.logger.info("Starting repository discovery")
        
        all_repo_names: Set[str] = set()
        discovered_repos: List[RepositoryInfo] = []
        
        # Add seed repositories
        all_repo_names.update(SEED_REPOSITORIES)
        self.logger.info(f"Added {len(SEED_REPOSITORIES)} seed repositories")
        
        # Search for additional repositories
        for query in GITHUB_SEARCH_QUERIES:
            try:
                search_results = self.search_repositories(
                    query, 
                    REPO_DISCOVERY["max_repos_per_search"]
                )
                all_repo_names.update(search_results)
                time.sleep(1)  # Brief pause between searches
            except Exception as e:
                self.logger.error(f"Search query failed '{query}': {e}")
                continue
        
        self.logger.info(f"Total unique repositories to check: {len(all_repo_names)}")
        
        # Get detailed info and filter
        for i, repo_name in enumerate(all_repo_names):
            if len(discovered_repos) >= PROCESSING_LIMITS["max_repositories"]:
                self.logger.info(f"Reached maximum repository limit: {PROCESSING_LIMITS['max_repositories']}")
                break
                
            try:
                repo_info = self.get_repository_info(repo_name)
                if repo_info and self._meets_quality_criteria(repo_info):
                    discovered_repos.append(repo_info)
                    self.logger.info(f"✓ Added {repo_name} (quality score: {repo_info.quality_score:.1f})")
                elif repo_info:
                    self.logger.debug(f"✗ Filtered out {repo_name} (quality score: {repo_info.quality_score:.1f})")
                    
            except Exception as e:
                self.logger.error(f"Failed to process {repo_name}: {e}")
                continue
                
            # Progress update
            if (i + 1) % 10 == 0:
                self.logger.info(f"Processed {i + 1}/{len(all_repo_names)} repositories, "
                               f"discovered {len(discovered_repos)} quality repos")
        
        # Sort by quality score
        discovered_repos.sort(key=lambda r: r.quality_score, reverse=True)
        
        self.logger.info(f"Repository discovery complete: {len(discovered_repos)} repositories")
        return discovered_repos
    
    def save_repositories(self, repositories: List[RepositoryInfo], filename: str = None):
        """Save discovered repositories to JSON file."""
        if filename is None:
            filename = get_output_path("repositories.json")
            
        repo_data = [asdict(repo) for repo in repositories]
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                "discovery_date": datetime.now().isoformat(),
                "total_repositories": len(repositories),
                "repositories": repo_data
            }, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Saved {len(repositories)} repositories to {filename}")

def main():
    """Test the discovery functionality."""
    logging.basicConfig(level=logging.INFO)
    
    discovery = GitHubDiscovery()
    
    # Test with a few repositories first
    test_repos = SEED_REPOSITORIES[:5]
    print(f"Testing with {len(test_repos)} repositories...")

    discovered_repos=[]
    
    for repo_name in test_repos:
        repo_info = discovery.get_repository_info(repo_name)
        if repo_info:
            print(f"✓ {repo_name}: {repo_info.stars} stars, "
                  f"{repo_info.erlang_percentage:.1%} Erlang, "
                  f"quality score: {repo_info.quality_score:.1f}")
            if repo_info and discovery._meets_quality_criteria(repo_info):
                    discovered_repos.append(repo_info)

        else:
            print(f"✗ Failed to get info for {repo_name}")

    discovery.save_repositories(discovered_repos, "./__test-output.json")
            
if __name__ == "__main__":
    main()
