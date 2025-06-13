#!/usr/bin/env python3
"""
Main orchestration script for Erlang corpus scraper.
Coordinates repository discovery, cloning, and preparation for function extraction.
"""

import argparse
import logging
import json
import os
import sys
from datetime import datetime
from typing import List, Optional

# Import our modules
from config import (
    LOGGING_CONFIG, OUTPUT_CONFIG, PROCESSING_LIMITS,
    get_output_path, validate_config
)
from scrapers.github_discovery import GitHubDiscovery, RepositoryInfo
from scrapers.repo_cloner import RepositoryCloner, CloneResult
from utils.rate_limiter import create_github_rate_limiter

def setup_logging(log_level: str = "INFO", log_to_file: bool = True):
    """Set up logging configuration."""
    log_format = LOGGING_CONFIG["format"]
    level = getattr(logging, log_level.upper())
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[]
    )
    
    logger = logging.getLogger()
    
    # Console handler
    if LOGGING_CONFIG["console_output"]:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(log_format)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file and LOGGING_CONFIG["file_output"]:
        log_file = get_output_path(OUTPUT_CONFIG["log_file"])
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    return logger

def load_repositories_from_file(filename: str) -> Optional[List[RepositoryInfo]]:
    """Load previously discovered repositories from JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        repositories = []
        for repo_data in data.get("repositories", []):
            repositories.append(RepositoryInfo(**repo_data))
            
        return repositories
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logging.getLogger(__name__).warning(f"Could not load repositories from {filename}: {e}")
        return None

def save_checkpoint(stage: str, data: dict):
    """Save checkpoint for resumability."""
    checkpoint_file = get_output_path(OUTPUT_CONFIG["checkpoint_file"])
    
    checkpoint = {
        "timestamp": datetime.now().isoformat(),
        "stage": stage,
        "data": data
    }
    
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, indent=2)
    
    logging.getLogger(__name__).info(f"Checkpoint saved: {stage}")

def load_checkpoint() -> Optional[dict]:
    """Load checkpoint for resuming."""
    checkpoint_file = get_output_path(OUTPUT_CONFIG["checkpoint_file"])
    
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def discover_repositories(args) -> List[RepositoryInfo]:
    """Discover repositories using GitHub API."""
    logger = logging.getLogger(__name__)
    
    # Check if we should load from existing file
    repo_file = get_output_path(OUTPUT_CONFIG["repositories_file"])
    if not args.force_discovery and os.path.exists(repo_file):
        logger.info(f"Loading existing repositories from {repo_file}")
        repositories = load_repositories_from_file(repo_file)
        if repositories:
            logger.info(f"Loaded {len(repositories)} repositories from file")
            return repositories
        else:
            logger.warning("Failed to load repositories from file, discovering new ones")
    
    # Discover new repositories
    logger.info("Starting repository discovery")
    discovery = GitHubDiscovery()
    
    try:
        repositories = discovery.discover_all_repositories()
        
        # Save discovered repositories
        discovery.save_repositories(repositories, repo_file)
        
        # Save checkpoint
        save_checkpoint("discovery_complete", {
            "repositories_found": len(repositories),
            "repositories_file": repo_file
        })
        
        return repositories
        
    except Exception as e:
        logger.error(f"Repository discovery failed: {e}")
        raise

def clone_repositories(repositories: List[RepositoryInfo], args) -> List[CloneResult]:
    """Clone discovered repositories."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting to clone {len(repositories)} repositories")
    
    # Limit repositories if specified
    if args.max_repos and len(repositories) > args.max_repos:
        logger.info(f"Limiting to {args.max_repos} repositories (sorted by quality)")
        repositories = sorted(repositories, key=lambda r: r.quality_score, reverse=True)[:args.max_repos]
    
    cloner = RepositoryCloner(max_workers=args.clone_workers)
    
    try:
        results = cloner.clone_repositories(repositories, force_reclone=args.force_reclone)
        
        # Save clone results
        results_file = get_output_path("clone_results.json")
        cloner.save_clone_results(results, results_file)
        
        # Save checkpoint
        successful_clones = len([r for r in results if r.success])
        save_checkpoint("cloning_complete", {
            "total_repositories": len(results),
            "successful_clones": successful_clones,
            "failed_clones": len(results) - successful_clones,
            "clone_results_file": results_file
        })
        
        return results
        
    except Exception as e:
        logger.error(f"Repository cloning failed: {e}")
        raise

def generate_corpus_stats(repositories: List[RepositoryInfo], clone_results: List[CloneResult]):
    """Generate and save corpus statistics."""
    logger = logging.getLogger(__name__)
    
    successful_clones = [r for r in clone_results if r.success]
    
    stats = {
        "generation_date": datetime.now().isoformat(),
        "discovery_stats": {
            "total_repositories_discovered": len(repositories),
            "average_quality_score": sum(r.quality_score for r in repositories) / len(repositories),
            "top_quality_score": max(r.quality_score for r in repositories),
            "languages_distribution": {},
            "stars_distribution": {
                "min": min(r.stars for r in repositories),
                "max": max(r.stars for r in repositories),
                "average": sum(r.stars for r in repositories) / len(repositories)
            }
        },
        "clone_stats": {
            "total_attempted": len(clone_results),
            "successful_clones": len(successful_clones),
            "failed_clones": len(clone_results) - len(successful_clones),
            "success_rate": len(successful_clones) / len(clone_results),
            "total_size_mb": sum(r.size_mb for r in successful_clones),
            "average_size_mb": sum(r.size_mb for r in successful_clones) / len(successful_clones) if successful_clones else 0,
            "total_clone_time": sum(r.clone_time_seconds for r in clone_results),
        },
        "repository_list": [
            {
                "full_name": repo.full_name,
                "stars": repo.stars,
                "quality_score": repo.quality_score,
                "erlang_percentage": repo.erlang_percentage,
                "cloned_successfully": any(r.success and r.repo_info.full_name == repo.full_name for r in clone_results)
            }
            for repo in sorted(repositories, key=lambda r: r.quality_score, reverse=True)
        ]
    }
    
    # Calculate language distribution
    for repo in repositories:
        main_lang = repo.language or "Unknown"
        stats["discovery_stats"]["languages_distribution"][main_lang] = \
            stats["discovery_stats"]["languages_distribution"].get(main_lang, 0) + 1
    
    # Save stats
    stats_file = get_output_path(OUTPUT_CONFIG["stats_file"])
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Corpus statistics saved to {stats_file}")
    
    # Log summary
    logger.info("=" * 60)
    logger.info("CORPUS GENERATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Repositories discovered: {len(repositories)}")
    logger.info(f"Repositories cloned: {len(successful_clones)}/{len(clone_results)}")
    logger.info(f"Success rate: {len(successful_clones)/len(clone_results)*100:.1f}%")
    logger.info(f"Total corpus size: {sum(r.size_mb for r in successful_clones):.1f} MB")
    logger.info(f"Average repository quality: {stats['discovery_stats']['average_quality_score']:.1f}/100")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Erlang Corpus Scraper - Discover and clone Erlang repositories for ML training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --discover --clone                    # Full pipeline
  python main.py --discover-only                       # Just discover repositories
  python main.py --clone-only                          # Clone from existing discovery
  python main.py --discover --clone --max-repos 50     # Limit to 50 repositories
  python main.py --force-discovery --force-reclone     # Force refresh everything
        """
    )
    
    # Mode selection
    parser.add_argument("--discover", action="store_true",
                       help="Discover repositories via GitHub API")
    parser.add_argument("--clone", action="store_true", 
                       help="Clone discovered repositories")
    parser.add_argument("--discover-only", action="store_true",
                       help="Only discover repositories (don't clone)")
    parser.add_argument("--clone-only", action="store_true",
                       help="Only clone repositories (use existing discovery)")
    
    # Force options
    parser.add_argument("--force-discovery", action="store_true",
                       help="Force rediscovery even if repositories.json exists")
    parser.add_argument("--force-reclone", action="store_true",
                       help="Force recloning even if repository already exists")
    
    # Limits and controls
    parser.add_argument("--max-repos", type=int, metavar="N",
                       help="Maximum number of repositories to process")
    parser.add_argument("--clone-workers", type=int, 
                       default=PROCESSING_LIMITS["parallel_clone_workers"],
                       help="Number of parallel clone workers")
    
    # Logging
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    parser.add_argument("--no-file-log", action="store_true",
                       help="Disable logging to file")
    
    # Resume functionality
    parser.add_argument("--resume", action="store_true",
                       help="Resume from last checkpoint")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.discover, args.clone, args.discover_only, args.clone_only, args.resume]):
        parser.error("Must specify at least one action: --discover, --clone, --discover-only, --clone-only, or --resume")
    
    if args.discover_only and args.clone:
        parser.error("Cannot use --discover-only with --clone")
    
    if args.clone_only and args.discover:
        parser.error("Cannot use --clone-only with --discover")
    
    # Set up logging
    logger = setup_logging(args.log_level, not args.no_file_log)
    
    # Validate configuration
    if not validate_config():
        logger.error("Configuration validation failed")
        return 1
    
    logger.info("=" * 60)
    logger.info("ERLANG CORPUS SCRAPER STARTING")
    logger.info("=" * 60)
    logger.info(f"Command line: {' '.join(sys.argv)}")
    logger.info(f"Start time: {datetime.now().isoformat()}")
    
    try:
        repositories = []
        clone_results = []
        
        # Handle resume functionality
        if args.resume:
            checkpoint = load_checkpoint()
            if checkpoint:
                logger.info(f"Resuming from checkpoint: {checkpoint['stage']}")
                # TODO: Implement resume logic based on checkpoint stage
            else:
                logger.warning("No checkpoint found, starting from beginning")
        
        # Discovery phase
        if args.discover or args.discover_only:
            logger.info("Phase 1: Repository Discovery")
            repositories = discover_repositories(args)
            logger.info(f"Discovery complete: {len(repositories)} repositories found")
        
        # Clone phase
        if args.clone or args.clone_only:
            logger.info("Phase 2: Repository Cloning")
            
            # Load repositories if we didn't discover them in this run
            if not repositories:
                repo_file = get_output_path(OUTPUT_CONFIG["repositories_file"])
                repositories = load_repositories_from_file(repo_file)
                if not repositories:
                    logger.error("No repositories found. Run discovery first.")
                    return 1
            
            clone_results = clone_repositories(repositories, args)
            successful_clones = len([r for r in clone_results if r.success])
            logger.info(f"Cloning complete: {successful_clones}/{len(clone_results)} repositories cloned")
        
        # Generate final statistics
        if repositories and (clone_results or args.discover_only):
            generate_corpus_stats(repositories, clone_results)
        
        logger.info("=" * 60)
        logger.info("SCRAPER COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        # Next steps message
        if clone_results:
            successful_repos = len([r for r in clone_results if r.success])
            logger.info(f"Next step: Run function extraction on {successful_repos} cloned repositories")
            logger.info("Command: python parsers/function_extractor.py")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Scraper interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Scraper failed with error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
