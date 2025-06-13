"""
Repository cloner for Erlang corpus scraper.
Handles efficient git cloning with shallow clones and error recovery.
"""

import os
import subprocess
import shutil
import logging
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Import our config and data structures
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    OUTPUT_CONFIG, PROCESSING_LIMITS, ERROR_HANDLING, 
    get_clone_path, get_output_path
)

# Import repository info from discovery module
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scrapers'))
from github_discovery import RepositoryInfo

@dataclass
class CloneResult:
    """Result of a repository clone operation."""
    repo_info: RepositoryInfo
    success: bool
    local_path: Optional[str]
    error_message: Optional[str]
    clone_time_seconds: float
    size_mb: float

class RepositoryCloner:
    """Handles cloning of GitHub repositories."""
    
    def __init__(self, max_workers: int = None):
        self.logger = logging.getLogger(__name__)
        self.max_workers = max_workers or PROCESSING_LIMITS["parallel_clone_workers"]
        
        # Ensure clone directory exists
        clone_dir = OUTPUT_CONFIG["clone_directory"]
        os.makedirs(clone_dir, exist_ok=True)
        self.logger.info(f"Clone directory: {clone_dir}")
        
        # Track cloning statistics
        self.stats = {
            "total_attempted": 0,
            "successful_clones": 0,
            "failed_clones": 0,
            "total_size_mb": 0.0,
            "total_time_seconds": 0.0
        }
    
    def _run_git_command(self, cmd: List[str], cwd: str = None, timeout: int = 300) -> Tuple[bool, str]:
        """
        Run a git command with timeout and error handling.
        
        Args:
            cmd: Git command as list of strings
            cwd: Working directory
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (success, output/error_message)
        """
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                return True, result.stdout.strip()
            else:
                return False, result.stderr.strip()
                
        except subprocess.TimeoutExpired:
            return False, f"Git command timed out after {timeout} seconds"
        except Exception as e:
            return False, f"Git command failed: {str(e)}"
    
    def _get_directory_size(self, path: str) -> float:
        """Get directory size in MB."""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except (OSError, IOError):
                        # Skip files we can't read
                        continue
            return total_size / (1024 * 1024)  # Convert to MB
        except Exception:
            return 0.0
    
    def _cleanup_failed_clone(self, local_path: str):
        """Clean up a failed clone directory."""
        try:
            if os.path.exists(local_path):
                shutil.rmtree(local_path)
                self.logger.debug(f"Cleaned up failed clone: {local_path}")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup {local_path}: {e}")
    
    def clone_repository(self, repo_info: RepositoryInfo, force_reclone: bool = False) -> CloneResult:
        """
        Clone a single repository with shallow clone for efficiency.
        
        Args:
            repo_info: Repository information
            force_reclone: If True, delete existing clone and reclone
            
        Returns:
            CloneResult with operation details
        """
        start_time = time.time()
        local_path = get_clone_path(repo_info.full_name)
        
        self.logger.info(f"Cloning {repo_info.full_name} to {local_path}")
        
        # Check if already cloned
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
        
        # Remove existing directory if force reclone
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
        
        # Attempt clone with retries
        for attempt in range(ERROR_HANDLING["max_retries"]):
            try:
                # Use shallow clone for efficiency (only latest commit)
                git_cmd = [
                    "git", "clone",
                    "--depth", "1",  # Shallow clone
                    "--single-branch",  # Only default branch
                    "--no-tags",  # Skip tags
                    repo_info.clone_url,
                    local_path
                ]
                
                success, output = self._run_git_command(git_cmd, timeout=600)  # 10 minute timeout
                
                if success:
                    # Verify clone was successful
                    if os.path.exists(os.path.join(local_path, ".git")):
                        size_mb = self._get_directory_size(local_path)
                        clone_time = time.time() - start_time
                        
                        self.logger.info(f"✓ Successfully cloned {repo_info.full_name} "
                                       f"({size_mb:.1f} MB in {clone_time:.1f}s)")
                        
                        return CloneResult(
                            repo_info=repo_info,
                            success=True,
                            local_path=local_path,
                            error_message=None,
                            clone_time_seconds=clone_time,
                            size_mb=size_mb
                        )
                    else:
                        error_msg = "Clone completed but .git directory not found"
                        self.logger.error(f"✗ {repo_info.full_name}: {error_msg}")
                        self._cleanup_failed_clone(local_path)
                else:
                    error_msg = f"Git clone failed: {output}"
                    self.logger.warning(f"✗ {repo_info.full_name} (attempt {attempt + 1}): {error_msg}")
                    self._cleanup_failed_clone(local_path)
                    
                    # If it's a network error, wait before retry
                    if "network" in output.lower() or "timeout" in output.lower():
                        if attempt < ERROR_HANDLING["max_retries"] - 1:
                            sleep_time = ERROR_HANDLING["retry_delay_seconds"] * (attempt + 1)
                            self.logger.info(f"Network error, waiting {sleep_time}s before retry")
                            time.sleep(sleep_time)
                    elif "not found" in output.lower() or "404" in output:
                        # Repository not found, don't retry
                        break
                        
            except Exception as e:
                error_msg = f"Clone exception: {str(e)}"
                self.logger.error(f"✗ {repo_info.full_name} (attempt {attempt + 1}): {error_msg}")
                self._cleanup_failed_clone(local_path)
                
                if attempt < ERROR_HANDLING["max_retries"] - 1:
                    sleep_time = ERROR_HANDLING["retry_delay_seconds"] * (attempt + 1)
                    time.sleep(sleep_time)
        
        # All attempts failed
        clone_time = time.time() - start_time
        final_error = f"Failed after {ERROR_HANDLING['max_retries']} attempts"
        
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
        
        Args:
            repositories: List of repositories to clone
            force_reclone: If True, reclone even if already exists
            
        Returns:
            List of CloneResult objects
        """
        self.logger.info(f"Starting to clone {len(repositories)} repositories "
                        f"with {self.max_workers} parallel workers")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all clone tasks
            future_to_repo = {
                executor.submit(self.clone_repository, repo, force_reclone): repo
                for repo in repositories
            }
            
            # Process completed tasks
            for future in as_completed(future_to_repo):
                repo = future_to_repo[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Update statistics
                    self.stats["total_attempted"] += 1
                    if result.success:
                        self.stats["successful_clones"] += 1
                        self.stats["total_size_mb"] += result.size_mb
                    else:
                        self.stats["failed_clones"] += 1
                    self.stats["total_time_seconds"] += result.clone_time_seconds
                    
                    # Progress logging
                    completed = self.stats["total_attempted"]
                    if completed % 10 == 0 or completed == len(repositories):
                        success_rate = self.stats["successful_clones"] / completed * 100
                        self.logger.info(f"Progress: {completed}/{len(repositories)} "
                                       f"({success_rate:.1f}% success rate)")
                        
                except Exception as e:
                    self.logger.error(f"Clone task failed for {repo.full_name}: {e}")
                    # Create failed result
                    results.append(CloneResult(
                        repo_info=repo,
                        success=False,
                        local_path=None,
                        error_message=str(e),
                        clone_time_seconds=0.0,
                        size_mb=0.0
                    ))
        
        # Sort results by success, then by repository name
        results.sort(key=lambda r: (not r.success, r.repo_info.full_name))
        
        # Log final statistics
        self._log_final_stats(results)
        
        return results
    
    def _log_final_stats(self, results: List[CloneResult]):
        """Log final cloning statistics."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        total_size = sum(r.size_mb for r in successful)
        total_time = sum(r.clone_time_seconds for r in results)
        avg_time = total_time / len(results) if results else 0
        
        self.logger.info("=" * 60)
        self.logger.info("CLONING SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total repositories: {len(results)}")
        self.logger.info(f"Successful clones: {len(successful)}")
        self.logger.info(f"Failed clones: {len(failed)}")
        self.logger.info(f"Success rate: {len(successful)/len(results)*100:.1f}%")
        self.logger.info(f"Total size: {total_size:.1f} MB")
        self.logger.info(f"Average clone time: {avg_time:.1f} seconds")
        
        if failed:
            self.logger.info("\nFailed repositories:")
            for result in failed[:10]:  # Show first 10 failures
                self.logger.info(f"  ✗ {result.repo_info.full_name}: {result.error_message}")
            if len(failed) > 10:
                self.logger.info(f"  ... and {len(failed) - 10} more")
    
    def save_clone_results(self, results: List[CloneResult], filename: str = None):
        """Save clone results to JSON file."""
        if filename is None:
            filename = get_output_path("clone_results.json")
        
        # Convert results to serializable format
        results_data = []
        for result in results:
            result_dict = asdict(result)
            # Convert RepositoryInfo to dict
            result_dict["repo_info"] = asdict(result.repo_info)
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
        
        Returns:
            List of (RepositoryInfo, local_path) tuples
        """
        return [(r.repo_info, r.local_path) for r in results if r.success and r.local_path]

def main():
    """Test the cloner with a few repositories."""
    logging.basicConfig(level=logging.INFO)
    
    # Mock some repository info for testing
    from github_discovery import RepositoryInfo
    
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
    
    cloner = RepositoryCloner(max_workers=2)
    results = cloner.clone_repositories(test_repos)
    
    for result in results:
        if result.success:
            print(f"✓ {result.repo_info.full_name} -> {result.local_path}")
        else:
            print(f"✗ {result.repo_info.full_name}: {result.error_message}")

if __name__ == "__main__":
    main()
