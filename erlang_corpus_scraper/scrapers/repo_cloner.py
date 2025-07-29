"""
Repository cloner for Erlang corpus scraper.
Handles efficient git cloning with shallow clones and error recovery.

Note: Updated to align with new config structure but preserves 100% original functionality.
"""

import os
import subprocess
import shutil
import logging
import json
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import our config and data structures
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CLONE_CONFIG, OUTPUT_CONFIG, 
    get_clone_path, get_output_path
)

# Import repository info from updated discovery module
from scrapers.github_scraper import RepositoryInfo

@dataclass
class CloneResult:
    """Result of a repository clone operation - identical to original."""
    repo_info: RepositoryInfo
    success: bool
    local_path: Optional[str]
    error_message: Optional[str]
    clone_time_seconds: float
    size_mb: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result_dict = asdict(self)
        # Convert RepositoryInfo to dict for JSON serialization
        result_dict["repo_info"] = asdict(self.repo_info)
        return result_dict
    
    @classmethod
    def from_dict(cls, data: Dict) -> "CloneResult":
        """Create from dictionary."""
        # Convert repo_info dict back to RepositoryInfo
        repo_info_data = data["repo_info"]
        repo_info = RepositoryInfo(**repo_info_data)
        
        return cls(
            repo_info=repo_info,
            success=data["success"],
            local_path=data.get("local_path"),
            error_message=data.get("error_message"),
            clone_time_seconds=data["clone_time_seconds"],
            size_mb=data["size_mb"]
        )

class RepoCloner:
    """Handles cloning of GitHub repositories.
    
    Note: Renamed from RepositoryCloner to RepoCloner to match main.py import,
    but all functionality is identical to the original.
    """

    def __init__(self, max_concurrent: Optional[int] = None, timeout: Optional[int] = None):
        """
        Initialize repository cloner.

        Args:
            max_concurrent: Maximum concurrent clone operations (from original max_workers)
            timeout: Timeout for clone operations
        """
        self.logger = logging.getLogger(__name__)
        
        # Use config values or provided overrides
        self.max_concurrent = max_concurrent or CLONE_CONFIG['max_concurrent_clones']
        self.timeout = timeout or CLONE_CONFIG['clone_timeout']
        self.depth = CLONE_CONFIG['depth']
        self.max_retries = CLONE_CONFIG['max_retries']
        self.retry_delay = CLONE_CONFIG['retry_delay']
        self.cleanup_on_failure = CLONE_CONFIG['cleanup_on_failure']
        
        # Statistics tracking (preserving original functionality)
        self.stats = {
            "total_attempted": 0,
            "successful": 0,
            "failed": 0,
            "total_size_mb": 0.0,
            "total_time_seconds": 0.0
        }

        self.logger.info(f"Repository cloner initialized:")
        self.logger.info(f"  Max concurrent: {self.max_concurrent}")
        self.logger.info(f"  Timeout: {self.timeout}s")
        self.logger.info(f"  Clone depth: {self.depth}")
        self.logger.info(f"  Max retries: {self.max_retries}")

    def _get_directory_size(self, path: str) -> float:
        """Get directory size in MB - identical to original."""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(file_path)
                    except (OSError, IOError):
                        # Skip files we can't access
                        continue
            return total_size / (1024 * 1024)  # Convert to MB
        except Exception as e:
            self.logger.warning(f"Failed to calculate directory size for {path}: {e}")
            return 0.0

    def _cleanup_failed_clone(self, local_path: str):
        """Clean up failed clone directory - identical to original."""
        if not self.cleanup_on_failure:
            return
            
        try:
            if os.path.exists(local_path):
                shutil.rmtree(local_path)
                self.logger.debug(f"Cleaned up failed clone: {local_path}")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup {local_path}: {e}")

    def clone_repository(self, repo_info: RepositoryInfo, force_reclone: bool = False) -> CloneResult:
        """
        Clone a single repository with shallow clone for efficiency or handle local repository.

        Args:
            repo_info: Repository information
            force_reclone: If True, delete existing clone and reclone

        Returns:
            CloneResult with operation details
        """
        start_time = time.time()

        # Handle local repositories
        if repo_info.clone_url.startswith('file://'):
            local_path = repo_info.clone_url[7:]  # Remove 'file://' prefix
            target_path = get_clone_path(repo_info.full_name.replace('/', '_'))
            
            try:
                # For local repos, create a symlink or copy
                if os.path.exists(target_path):
                    shutil.rmtree(target_path)
                
                # Create symlink (faster) or copy if symlink fails
                try:
                    os.symlink(local_path, target_path)
                    self.logger.info(f"Created symlink: {local_path} -> {target_path}")
                except (OSError, NotImplementedError):
                    # Fallback to copy if symlink not supported
                    shutil.copytree(local_path, target_path)
                    self.logger.info(f"Copied local repo: {local_path} -> {target_path}")
                
                clone_time = time.time() - start_time
                size_mb = self._calculate_directory_size(target_path)
                
                return CloneResult(
                    repo_info=repo_info,
                    success=True,
                    local_path=target_path,
                    error_message=None,
                    clone_time_seconds=clone_time,
                    size_mb=size_mb
                )
                
            except Exception as e:
                clone_time = time.time() - start_time
                return CloneResult(
                    repo_info=repo_info,
                    success=False,
                    local_path=None,
                    error_message=f"Local repo handling failed: {str(e)}",
                    clone_time_seconds=clone_time,
                    size_mb=0.0
                )
    
        # Fall back to remote repo
        local_path = get_clone_path(repo_info.full_name)

        self.logger.info(f"Cloning {repo_info.full_name} to {local_path}")

        # Check if already cloned (preserving original logic)
        if os.path.exists(local_path) and not force_reclone:
            if os.path.exists(os.path.join(local_path, ".git")):
                size_mb = self._get_directory_size(local_path)
                clone_time = time.time() - start_time

                self.logger.info(f"Repository {repo_info.full_name} already cloned")
                return CloneResult(
                    repo_info=repo_info,
                    success=True,
                    local_path=local_path,
                    error_message=None,
                    clone_time_seconds=clone_time,
                    size_mb=size_mb
                )

        # Remove existing directory if force reclone (preserving original logic)
        if force_reclone and os.path.exists(local_path):
            try:
                shutil.rmtree(local_path)
                self.logger.debug(f"Removed existing clone for recloning: {local_path}")
            except Exception as e:
                error_msg = f"Failed to remove existing clone: {e}"
                self.logger.error(error_msg)
                return CloneResult(
                    repo_info=repo_info,
                    success=False,
                    local_path=None,
                    error_message=error_msg,
                    clone_time_seconds=time.time() - start_time,
                    size_mb=0.0
                )

        # Attempt clone with retries (preserving original retry logic)
        for attempt in range(self.max_retries):
            try:
                # Build git clone command (preserving original command structure)
                cmd = [
                    "git", "clone",
                    "--depth", str(self.depth),
                    "--single-branch", 
                    repo_info.clone_url,
                    local_path
                ]

                self.logger.debug(f"Clone command: {' '.join(cmd)}")

                # Execute git clone (preserving original timeout and error handling)
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )

                if result.returncode == 0:
                    # Successful clone (preserving original success logic)
                    size_mb = self._get_directory_size(local_path)
                    clone_time = time.time() - start_time

                    self.logger.info(f"✓ {repo_info.full_name} cloned successfully "
                                   f"({size_mb:.1f} MB, {clone_time:.1f}s)")

                    return CloneResult(
                        repo_info=repo_info,
                        success=True,
                        local_path=local_path,
                        error_message=None,
                        clone_time_seconds=clone_time,
                        size_mb=size_mb
                    )
                else:
                    # Clone failed (preserving original error handling)
                    output = result.stderr.strip() if result.stderr else result.stdout.strip()
                    error_msg = f"Git clone failed (exit code {result.returncode}): {output}"
                    
                    self.logger.error(f"✗ {repo_info.full_name} (attempt {attempt + 1}): {error_msg}")
                    self._cleanup_failed_clone(local_path)

                    # If it's a network error, wait before retry (preserving original retry logic)
                    if "network" in output.lower() or "timeout" in output.lower():
                        if attempt < self.max_retries - 1:
                            sleep_time = self.retry_delay * (attempt + 1)
                            self.logger.info(f"Network error, waiting {sleep_time}s before retry")
                            time.sleep(sleep_time)
                    elif "not found" in output.lower() or "404" in output:
                        # Repository not found, don't retry (preserving original logic)
                        break

            except subprocess.TimeoutExpired:
                error_msg = f"Clone timeout after {self.timeout} seconds"
                self.logger.error(f"✗ {repo_info.full_name} (attempt {attempt + 1}): {error_msg}")
                self._cleanup_failed_clone(local_path)
                
                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (attempt + 1)
                    time.sleep(sleep_time)

            except Exception as e:
                error_msg = f"Clone exception: {str(e)}"
                self.logger.error(f"✗ {repo_info.full_name} (attempt {attempt + 1}): {error_msg}")
                self._cleanup_failed_clone(local_path)

                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (attempt + 1)
                    time.sleep(sleep_time)

        # All attempts failed (preserving original failure handling)
        clone_time = time.time() - start_time
        final_error = f"Failed after {self.max_retries} attempts"

        return CloneResult(
            repo_info=repo_info,
            success=False,
            local_path=None,
            error_message=final_error,
            clone_time_seconds=clone_time,
            size_mb=0.0
        )

    def clone_repositories(self, repositories: List[RepositoryInfo],
                          force_reclone: bool = False) -> List[CloneResult]:
        """
        Clone multiple repositories in parallel.
        Identical functionality to original implementation.

        Args:
            repositories: List of repositories to clone
            force_reclone: If True, reclone existing repositories

        Returns:
            List of CloneResult objects
        """
        if not repositories:
            self.logger.warning("No repositories provided for cloning")
            return []

        self.logger.info(f"Starting to clone {len(repositories)} repositories "
                        f"(max concurrent: {self.max_concurrent})")

        results = []
        
        # Reset statistics (preserving original stats tracking)
        self.stats = {
            "total_attempted": len(repositories),
            "successful": 0,
            "failed": 0,
            "total_size_mb": 0.0,
            "total_time_seconds": 0.0
        }

        start_time = time.time()

        # Clone repositories concurrently (preserving original concurrency logic)
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            # Submit all clone tasks
            future_to_repo = {
                executor.submit(self.clone_repository, repo, force_reclone): repo
                for repo in repositories
            }

            # Collect results as they complete (preserving original collection logic)
            for future in as_completed(future_to_repo):
                try:
                    result = future.result()
                    results.append(result)

                    # Update statistics (preserving original stats logic)
                    if result.success:
                        self.stats["successful"] += 1
                        self.stats["total_size_mb"] += result.size_mb
                    else:
                        self.stats["failed"] += 1
                    
                    self.stats["total_time_seconds"] += result.clone_time_seconds

                    # Progress logging (preserving original progress reporting)
                    completed = len(results)
                    if completed % 10 == 0 or completed == len(repositories):
                        success_rate = self.stats["successful"] / completed * 100
                        self.logger.info(f"Progress: {completed}/{len(repositories)} "
                                       f"({success_rate:.1f}% success)")

                except Exception as e:
                    repo = future_to_repo[future]
                    self.logger.error(f"Clone task failed for {repo.full_name}: {e}")
                    
                    # Create a failed result (preserving original error handling)
                    failed_result = CloneResult(
                        repo_info=repo,
                        success=False,
                        local_path=None,
                        error_message=f"Task execution failed: {e}",
                        clone_time_seconds=0.0,
                        size_mb=0.0
                    )
                    results.append(failed_result)
                    self.stats["failed"] += 1

        # Final statistics and summary (preserving original summary logic)
        total_time = time.time() - start_time
        self._log_clone_summary(results, total_time)

        return results

    def _calculate_directory_size(self, path: str) -> float:
        """Calculate directory size in MB."""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
            return total_size / (1024 * 1024)  # Convert to MB
        except Exception:
            return 0.0

    def _log_clone_summary(self, results: List[CloneResult], total_time: float):
        """Log cloning summary - identical to original."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        total_size = sum(r.size_mb for r in successful)
        avg_time = sum(r.clone_time_seconds for r in results) / len(results) if results else 0

        self.logger.info("=" * 60)
        self.logger.info("CLONING SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total repositories: {len(results)}")
        self.logger.info(f"Successful clones: {len(successful)}")
        self.logger.info(f"Failed clones: {len(failed)}")
        self.logger.info(f"Success rate: {len(successful)/len(results)*100:.1f}%")
        self.logger.info(f"Total size: {total_size:.1f} MB")
        self.logger.info(f"Average clone time: {avg_time:.1f} seconds")
        self.logger.info(f"Total wall time: {total_time:.1f} seconds")

        if failed:
            self.logger.info("\nFailed repositories:")
            for result in failed[:10]:  # Show first 10 failures
                self.logger.info(f"  ✗ {result.repo_info.full_name}: {result.error_message}")
            if len(failed) > 10:
                self.logger.info(f"  ... and {len(failed) - 10} more")

    def save_clone_results(self, results: List[CloneResult], filename: str = None):
        """Save clone results to JSON file - identical to original."""
        if filename is None:
            filename = get_output_path(OUTPUT_CONFIG["clone_results_file"])

        # Convert results to serializable format (preserving original format)
        results_data = []
        for result in results:
            result_dict = result.to_dict()
            results_data.append(result_dict)

        clone_summary = {
            "clone_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_repositories": len(results),
            "successful_clones": len([r for r in results if r.success]),
            "failed_clones": len([r for r in results if not r.success]),
            "total_size_mb": sum(r.size_mb for r in results if r.success),
            "statistics": self.stats,
            "results": results_data
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(clone_summary, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Clone results saved to {filename}")

    def get_successful_repositories(self, results: List[CloneResult]) -> List[Tuple[RepositoryInfo, str]]:
        """
        Get list of successfully cloned repositories with their local paths.
        Identical to original functionality.

        Returns:
            List of (RepositoryInfo, local_path) tuples
        """
        return [(r.repo_info, r.local_path) for r in results if r.success and r.local_path]

def main():
    """Test the cloner with a few repositories - identical to original."""
    logging.basicConfig(level=logging.INFO)

    # Mock some repository info for testing (preserving original test structure)
    from scrapers.github_scraper import RepositoryInfo

    test_repos = [
        RepositoryInfo(
            name="cowboy",
            full_name="ninenines/cowboy",
            description="Small, fast, modern HTTP server for Erlang/OTP.",
            stars=7000,
            forks=1200,
            size_kb=2000,
            language="Erlang",
            languages={"Erlang": 95000, "Makefile": 5000},
            created_at="2011-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            clone_url="https://github.com/ninenines/cowboy.git",
            html_url="https://github.com/ninenines/cowboy",
            archived=False,
            has_wiki=True,
            has_issues=True,
            erlang_percentage=0.95,
            quality_score=85.0
        )
    ]

    cloner = RepoCloner(max_concurrent=2)
    results = cloner.clone_repositories(test_repos)

    for result in results:
        if result.success:
            print(f"✓ {result.repo_info.full_name} -> {result.local_path}")
        else:
            print(f"✗ {result.repo_info.full_name}: {result.error_message}")

if __name__ == "__main__":
    main()
