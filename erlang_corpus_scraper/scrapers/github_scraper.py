"""
GitHub repository discovery for Erlang corpus scraper.
Discovers high-quality Erlang repositories using GitHub API.

Note: This is the exact same functionality as GitHubDiscovery but renamed to GitHubScraper
to match the updated main.py imports. All original functionality is preserved.
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
    GITHUB_CONFIG, DISCOVERY_CONFIG, OUTPUT_CONFIG, 
    get_output_path
)

@dataclass
class RepositoryInfo:
    """Repository information structure - identical to original."""
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

class GitHubScraper:
    """Discovers Erlang repositories using GitHub API.
    
    This is identical to the original GitHubDiscovery class, just renamed for consistency.
    All functionality and behavior is preserved exactly.
    """

    def __init__(self, token: Optional[str] = None, max_repos: Optional[int] = None, 
                 min_stars: Optional[int] = None):
        """Initialize GitHub scraper.
        
        Args:
            token: GitHub API token (will use env GITHUB_TOKEN if not provided)
            max_repos: Maximum repositories to discover
            min_stars: Minimum star count filter
        """
        self.logger = logging.getLogger(__name__)
        
        # API configuration
        self.base_url = GITHUB_CONFIG['base_url']
        self.token = token or os.getenv('GITHUB_TOKEN')
        self.headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': GITHUB_CONFIG['user_agent']
        }
        
        if self.token:
            self.headers['Authorization'] = f'token {self.token}'
            self.logger.info("GitHub API token configured")
        else:
            self.logger.warning("No GitHub token - rate limits will be severe")

        # Discovery parameters  
        self.max_repos = max_repos or DISCOVERY_CONFIG['max_total_repos']
        self.min_stars = min_stars or DISCOVERY_CONFIG['min_stars']
        self.min_size = DISCOVERY_CONFIG['min_size']
        self.max_size = DISCOVERY_CONFIG['max_size']
        
        # Rate limiting
        self.requests_per_hour = GITHUB_CONFIG['rate_limit_per_hour']
        self.last_request_time = 0
        self.request_count = 0
        
        # Quality criteria
        self.exclude_forks = DISCOVERY_CONFIG['exclude_forks']
        self.exclude_archived = DISCOVERY_CONFIG['exclude_archived']
        
        self.logger.info(f"GitHub scraper initialized: max_repos={self.max_repos}, min_stars={self.min_stars}")

    def _make_request(self, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make rate-limited GitHub API request."""
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_request_time < 3600 / self.requests_per_hour:
            sleep_time = (3600 / self.requests_per_hour) - (current_time - self.last_request_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

        try:
            response = requests.get(
                url, 
                headers=self.headers, 
                params=params or {},
                timeout=GITHUB_CONFIG['request_timeout']
            )
            
            self.last_request_time = time.time()
            self.request_count += 1
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 403:
                # Rate limit hit
                reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                current_time = int(time.time())
                wait_time = max(0, reset_time - current_time)
                
                self.logger.warning(f"Rate limit hit. Waiting {wait_time} seconds...")
                time.sleep(wait_time + 1)
                return self._make_request(url, params)  # Retry
            else:
                self.logger.error(f"GitHub API error {response.status_code}: {response.text}")
                return None
                
        except requests.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            return None

    def search_repositories(self, query: str, per_page: int = 100, max_pages: int = 10) -> List[Dict]:
        """Search repositories with given query."""
        repositories = []
        page = 1
        
        self.logger.info(f"Searching repositories: {query}")
        
        while page <= max_pages and len(repositories) < self.max_repos:
            url = f"{self.base_url}/search/repositories"
            params = {
                'q': query,
                'sort': 'stars',
                'order': 'desc',
                'per_page': per_page,
                'page': page
            }
            
            response_data = self._make_request(url, params)
            if not response_data:
                break
                
            items = response_data.get('items', [])
            if not items:
                break
                
            repositories.extend(items)
            
            # Check if we have more pages
            total_count = response_data.get('total_count', 0)
            if len(repositories) >= total_count:
                break
                
            page += 1
            
        self.logger.info(f"Found {len(repositories)} repositories for query: {query}")
        return repositories

    def get_repository_info(self, repo_name: str) -> Optional[RepositoryInfo]:
        """Get detailed information for a single repository."""
        # Get basic repo info
        url = f"{self.base_url}/repos/{repo_name}"
        repo_data = self._make_request(url)
        
        if not repo_data:
            return None
            
        # Get language breakdown
        languages_url = f"{url}/languages"
        languages_data = self._make_request(languages_url) or {}
        
        # Calculate Erlang percentage
        total_bytes = sum(languages_data.values())
        erlang_bytes = languages_data.get('Erlang', 0)
        erlang_percentage = erlang_bytes / total_bytes if total_bytes > 0 else 0
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(repo_data, erlang_percentage)
        
        return RepositoryInfo(
            name=repo_data['name'],
            full_name=repo_data['full_name'],
            description=repo_data.get('description', ''),
            stars=repo_data['stargazers_count'],
            forks=repo_data['forks_count'],
            size_kb=repo_data['size'],
            language=repo_data.get('language', ''),
            languages=languages_data,
            created_at=repo_data['created_at'],
            updated_at=repo_data['updated_at'],
            clone_url=repo_data['clone_url'],
            html_url=repo_data['html_url'],
            archived=repo_data.get('archived', False),
            has_wiki=repo_data.get('has_wiki', False),
            has_issues=repo_data.get('has_issues', False),
            erlang_percentage=erlang_percentage,
            quality_score=quality_score
        )

    def _calculate_quality_score(self, repo_data: Dict, erlang_percentage: float) -> float:
        """Calculate repository quality score."""
        score = 0.0
        
        # Star rating (0-40 points)
        stars = repo_data['stargazers_count']
        star_score = min(40, stars * 2)  # 1 star = 2 points, max 40
        score += star_score
        
        # Erlang percentage (0-30 points)
        erlang_score = erlang_percentage * 30
        score += erlang_score
        
        # Activity score (0-20 points)
        updated_at = datetime.fromisoformat(repo_data['updated_at'].replace('Z', '+00:00'))
        days_since_update = (datetime.now(updated_at.tzinfo) - updated_at).days
        
        if days_since_update <= 30:
            activity_score = 20
        elif days_since_update <= 90:
            activity_score = 15
        elif days_since_update <= 365:
            activity_score = 10
        else:
            activity_score = 5
        score += activity_score
        
        # Size score (0-10 points)
        size_kb = repo_data['size']
        if self.min_size <= size_kb <= self.max_size:
            # Optimal size range
            if size_kb <= 1000:
                size_score = 10
            elif size_kb <= 10000:
                size_score = 8
            else:
                size_score = 6
        else:
            size_score = 2
        score += size_score
        
        return score

    def _meets_quality_criteria(self, repo: RepositoryInfo) -> bool:
        """Check if repository meets our quality criteria."""
        # Basic filters
        if repo.stars < self.min_stars:
            return False
            
        if repo.size_kb < self.min_size or repo.size_kb > self.max_size:
            return False
            
        if self.exclude_archived and repo.archived:
            return False
            
        # Erlang percentage check
        if repo.erlang_percentage < 0.5:  # At least 50% Erlang
            return False
            
        # Quality score threshold
        if repo.quality_score < 30:  # Minimum quality threshold
            return False
            
        return True

    def discover_repositories(self, max_repos: Optional[int] = None) -> List[RepositoryInfo]:
        """Discover repositories using search queries."""
        max_repos = max_repos or self.max_repos
        discovered_repos = []
        seen_repos = set()
        
        self.logger.info(f"Starting repository discovery (target: {max_repos})")
        
        # Use search queries from config
        queries = DISCOVERY_CONFIG['search_queries']
        
        for query in queries:
            if len(discovered_repos) >= max_repos:
                break
                
            self.logger.info(f"Processing query: {query}")
            
            # Search repositories
            search_results = self.search_repositories(
                query, 
                per_page=100,
                max_pages=DISCOVERY_CONFIG['max_repos_per_query'] // 100 + 1
            )
            
            for repo_data in search_results:
                if len(discovered_repos) >= max_repos:
                    break
                    
                repo_name = repo_data['full_name']
                
                # Skip duplicates
                if repo_name in seen_repos:
                    continue
                seen_repos.add(repo_name)
                
                # Get detailed info
                repo_info = self.get_repository_info(repo_name)
                
                if repo_info and self._meets_quality_criteria(repo_info):
                    discovered_repos.append(repo_info)
                    self.logger.info(f"✓ Added {repo_name}: {repo_info.stars} stars, "
                                   f"{repo_info.erlang_percentage:.1%} Erlang, "
                                   f"score: {repo_info.quality_score:.1f}")
                
        self.logger.info(f"Discovery complete: {len(discovered_repos)} repositories found")
        return discovered_repos

    def save_repositories(self, repositories: List[RepositoryInfo], 
                         filename: Optional[str] = None):
        """Save repositories to JSON file."""
        if filename is None:
            filename = get_output_path(OUTPUT_CONFIG["repos_file"])

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

    # Test using the new GitHubScraper class (same functionality as GitHubDiscovery)
    scraper = GitHubScraper()

    # Use a simple search to test
    repositories = scraper.discover_repositories(max_repos=5)
    
    if repositories:
        print(f"Successfully discovered {len(repositories)} repositories:")
        for repo in repositories:
            print(f"  - {repo.full_name}: {repo.stars} stars, "
                  f"{repo.erlang_percentage:.1%} Erlang, "
                  f"quality score: {repo.quality_score:.1f}")
        
        # Save test results
        scraper.save_repositories(repositories, "./test_repositories.json")
    else:
        print("No repositories discovered")

if __name__ == "__main__":
    main()
