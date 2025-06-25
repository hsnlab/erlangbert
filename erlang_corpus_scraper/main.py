#!/usr/bin/env python3
"""
Main orchestration script for Erlang corpus scraper.
Coordinates repository discovery, cloning, and function extraction for GraphCodeBERT training.
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
    LOGGING_CONFIG, OUTPUT_CONFIG, PROCESSING_LIMITS, PARSER_CONFIG,
    get_output_path, get_clone_path, validate_config
)
from scrapers.github_discovery import GitHubDiscovery, RepositoryInfo
from scrapers.repo_cloner import RepositoryCloner, CloneResult
from parsers.function_extractor import FunctionExtractor, ErlangFunction
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

def load_clone_results_from_file(filename: str) -> Optional[List[CloneResult]]:
    """Load previously cloned repository results from JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        clone_results = []
        for result_data in data.get("clone_results", []):
            # Reconstruct CloneResult from saved data
            repo_info = RepositoryInfo(**result_data["repo_info"])
            clone_result = CloneResult(
                repo_info=repo_info,
                success=result_data["success"],
                error_message=result_data.get("error_message"),
                local_path=result_data.get("local_path"),
                clone_time=result_data.get("clone_time", 0.0)
            )
            clone_results.append(clone_result)
            
        return clone_results
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logging.getLogger(__name__).warning(f"Could not load clone results from {filename}: {e}")
        return None

def save_checkpoint(stage: str, data: dict):
    """Save checkpoint for resumability."""
    checkpoint_file = get_output_path(OUTPUT_CONFIG["checkpoint_file"])
    
    checkpoint = {
        "stage": stage,
        "timestamp": datetime.now().isoformat(),
        "data": data
    }
    
    try:
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to save checkpoint: {e}")

def load_checkpoint() -> Optional[dict]:
    """Load checkpoint for resumability."""
    checkpoint_file = get_output_path(OUTPUT_CONFIG["checkpoint_file"])
    
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return None

def discover_repositories(args) -> List[RepositoryInfo]:
    """Phase 1: Discover Erlang repositories."""
    logger = logging.getLogger(__name__)
    logger.info("Starting repository discovery...")
    
    # Check if we should skip discovery
    repo_file = get_output_path(OUTPUT_CONFIG["repositories_file"])
    if os.path.exists(repo_file) and not args.force_discovery:
        logger.info(f"Repository file exists: {repo_file}")
        repositories = load_repositories_from_file(repo_file)
        if repositories:
            logger.info(f"Loaded {len(repositories)} repositories from file")
            return repositories[:args.max_repos] if args.max_repos else repositories
    
    # Create rate limiter and discovery service
    rate_limiter = create_github_rate_limiter()
    discovery = GitHubDiscovery(rate_limiter=rate_limiter)
    
    try:
        # Discover repositories
        repositories = discovery.discover_repositories()
        
        # Apply limits
        if args.max_repos:
            repositories = repositories[:args.max_repos]
        
        # Save results
        save_data = {
            "discovery_date": datetime.now().isoformat(),
            "total_discovered": len(repositories),
            "repositories": [repo.__dict__ for repo in repositories]
        }
        
        with open(repo_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"Saved {len(repositories)} repositories to {repo_file}")
        save_checkpoint("discovery_complete", {"repository_count": len(repositories)})
        
        return repositories
        
    except Exception as e:
        logger.error(f"Repository discovery failed: {e}")
        raise

def clone_repositories(repositories: List[RepositoryInfo], args) -> List[CloneResult]:
    """Phase 2: Clone discovered repositories."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting to clone {len(repositories)} repositories...")
    
    # Check if we should skip cloning
    clone_file = get_output_path("clone_results.json")
    if os.path.exists(clone_file) and not args.force_reclone:
        logger.info(f"Clone results file exists: {clone_file}")
        clone_results = load_clone_results_from_file(clone_file)
        if clone_results:
            successful = len([r for r in clone_results if r.success])
            logger.info(f"Loaded {len(clone_results)} clone results ({successful} successful)")
            return clone_results
    
    # Create cloner
    clone_base_dir = OUTPUT_CONFIG["clone_directory"]
    cloner = RepositoryCloner(
        base_directory=clone_base_dir,
        max_workers=args.clone_workers,
        force_reclone=args.force_reclone
    )
    
    try:
        # Clone repositories
        clone_results = cloner.clone_repositories(repositories)
        
        # Save results
        save_data = {
            "clone_date": datetime.now().isoformat(),
            "total_attempted": len(clone_results),
            "successful_clones": len([r for r in clone_results if r.success]),
            "clone_results": []
        }
        
        for result in clone_results:
            result_data = {
                "repo_info": result.repo_info.__dict__,
                "success": result.success,
                "error_message": result.error_message,
                "local_path": result.local_path,
                "clone_time": result.clone_time
            }
            save_data["clone_results"].append(result_data)
        
        with open(clone_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2)
        
        successful = len([r for r in clone_results if r.success])
        logger.info(f"Saved clone results: {successful}/{len(clone_results)} successful")
        save_checkpoint("cloning_complete", {"successful_clones": successful})
        
        return clone_results
        
    except Exception as e:
        logger.error(f"Repository cloning failed: {e}")
        raise

def extract_functions(clone_results: List[CloneResult], args) -> List[ErlangFunction]:
    """Phase 3: Extract functions from cloned repositories."""
    logger = logging.getLogger(__name__)
    
    # Filter successful clones
    successful_clones = [r for r in clone_results if r.success and r.local_path]
    logger.info(f"Starting function extraction from {len(successful_clones)} repositories...")
    
    # Check if we should skip extraction
    functions_file = get_output_path(OUTPUT_CONFIG["functions_file"])
    if os.path.exists(functions_file) and not args.force_extraction:
        logger.info(f"Functions file exists: {functions_file}")
        if not args.force_extraction:
            logger.info("Use --force-extraction to regenerate")
            return []
    
    # Create extractor
    extractor = FunctionExtractor()
    all_functions = []
    
    try:
        for i, clone_result in enumerate(successful_clones):
            logger.info(f"Processing repository {i+1}/{len(successful_clones)}: {clone_result.repo_info.full_name}")
            
            try:
                repo_functions = extractor.extract_from_repository(
                    clone_result.local_path,
                    clone_result.repo_info.full_name
                )
                all_functions.extend(repo_functions)
                
                # Apply per-repository limit
                if len(repo_functions) > PROCESSING_LIMITS["max_functions_per_repo"]:
                    logger.warning(f"Repository {clone_result.repo_info.full_name} has {len(repo_functions)} functions, limiting to {PROCESSING_LIMITS['max_functions_per_repo']}")
                    repo_functions = repo_functions[:PROCESSING_LIMITS["max_functions_per_repo"]]
                
                # Apply total limit
                if len(all_functions) >= PROCESSING_LIMITS["max_total_functions"]:
                    logger.info(f"Reached maximum total functions: {PROCESSING_LIMITS['max_total_functions']}")
                    break
                    
            except Exception as e:
                logger.warning(f"Failed to extract from {clone_result.repo_info.full_name}: {e}")
                continue
        
        # Save functions to JSONL format
        logger.info(f"Saving {len(all_functions)} functions to {functions_file}")
        with open(functions_file, 'w', encoding='utf-8') as f:
            for func in all_functions:
                # Convert to dict and save as JSONL
                func_dict = {
                    "idx": func.idx,
                    "name": func.name,
                    "arity": func.arity,
                    "file_path": func.file_path,
                    "line_start": func.line_start,
                    "line_end": func.line_end,
                    "code": func.code,
                    "code_tokens": func.code_tokens,
                    "docstring": func.docstring,
                    "type_spec": func.type_spec,
                    "clauses": func.clauses,
                    "has_guards": func.has_guards,
                    "has_patterns": func.has_patterns,
                    "is_exported": func.is_exported,
                    "score": func.score,
                    "score_breakdown": func.score_breakdown,
                    "repo_name": func.repo_name
                }
                f.write(json.dumps(func_dict) + '\\n')
        
        # Save extraction statistics
        stats_file = get_output_path(OUTPUT_CONFIG["stats_file"])
        extraction_stats = {
            "extraction_date": datetime.now().isoformat(),
            "total_repositories": len(successful_clones),
            "total_functions": len(all_functions),
            "average_functions_per_repo": len(all_functions) / len(successful_clones) if successful_clones else 0,
            "score_distribution": {
                "high_quality": len([f for f in all_functions if f.score >= 50]),
                "medium_quality": len([f for f in all_functions if 25 <= f.score < 50]),
                "low_quality": len([f for f in all_functions if f.score < 25])
            },
            "feature_distribution": {
                "with_guards": len([f for f in all_functions if f.has_guards]),
                "with_patterns": len([f for f in all_functions if f.has_patterns]),
                "with_docs": len([f for f in all_functions if f.docstring]),
                "exported": len([f for f in all_functions if f.is_exported])
            }
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(extraction_stats, f, indent=2)
        
        logger.info(f"Saved extraction statistics to {stats_file}")
        save_checkpoint("extraction_complete", {"function_count": len(all_functions)})
        
        return all_functions
        
    except Exception as e:
        logger.error(f"Function extraction failed: {e}")
        raise

def generate_corpus_summary(repositories: List[RepositoryInfo], clone_results: List[CloneResult], functions: List[ErlangFunction]):
    """Generate final corpus summary."""
    logger = logging.getLogger(__name__)
    
    summary = {
        "corpus_generation_date": datetime.now().isoformat(),
        "pipeline_summary": {
            "repositories_discovered": len(repositories),
            "repositories_cloned": len([r for r in clone_results if r.success]),
            "functions_extracted": len(functions)
        },
        "quality_metrics": {
            "avg_score": sum(f.score for f in functions) / len(functions) if functions else 0,
            "high_quality_functions": len([f for f in functions if f.score >= 50]),
            "functions_with_docs": len([f for f in functions if f.docstring]),
            "functions_with_guards": len([f for f in functions if f.has_guards])
        }
    }
    
    summary_file = get_output_path("corpus_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Generated corpus summary: {summary_file}")

def main():
    """Main pipeline orchestration."""
    parser = argparse.ArgumentParser(
        description="Erlang Corpus Scraper - Discover, clone, and extract functions from Erlang repositories for GraphCodeBERT training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --discover --clone --extract                     # Full pipeline
  python main.py --discover-only --max-repos 10                   # Just discover 10 repositories
  python main.py --clone-only                                     # Clone from existing discovery
  python main.py --extract-only                                   # Extract from existing clones
  python main.py --discover --clone --extract --max-repos 50      # Limited pipeline
  python main.py --force-discovery --force-reclone --force-extraction  # Force refresh everything
        """
    )
    
    # Phase selection
    parser.add_argument("--discover", action="store_true",
                       help="Discover repositories via GitHub API")
    parser.add_argument("--clone", action="store_true", 
                       help="Clone discovered repositories")
    parser.add_argument("--extract", action="store_true",
                       help="Extract functions from cloned repositories")
    
    # Phase-only options
    parser.add_argument("--discover-only", action="store_true",
                       help="Only discover repositories (don't clone or extract)")
    parser.add_argument("--clone-only", action="store_true",
                       help="Only clone repositories (use existing discovery)")
    parser.add_argument("--extract-only", action="store_true",
                       help="Only extract functions (use existing clones)")
    
    # Force options
    parser.add_argument("--force-discovery", action="store_true",
                       help="Force rediscovery even if repositories.json exists")
    parser.add_argument("--force-reclone", action="store_true",
                       help="Force recloning even if repository already exists")
    parser.add_argument("--force-extraction", action="store_true",
                       help="Force re-extraction even if functions.jsonl exists")
    
    # Limits and controls
    parser.add_argument("--max-repos", type=int, metavar="N",
                       help="Maximum number of repositories to process")
    parser.add_argument("--clone-workers", type=int, 
                       default=PROCESSING_LIMITS["parallel_clone_workers"],
                       help="Number of parallel clone workers")
    parser.add_argument("--min-score", type=int,
                       default=PARSER_CONFIG.get("min_score", 10),
                       help="Minimum function quality score threshold")
    
    # Logging
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    parser.add_argument("--no-file-log", action="store_true",
                       help="Disable logging to file")
    
    # Resume functionality
    parser.add_argument("--resume", action="store_true",
                       help="Resume from last checkpoint")
    
    # Test mode
    parser.add_argument("--test", action="store_true",
                       help="Run in test mode with minimal data")

    args = parser.parse_args()

    # Set up logging
    logger = setup_logging(args.log_level, not args.no_file_log)
    
    # Update configuration from command line
    if args.min_score is not None:
        PARSER_CONFIG["min_score"] = args.min_score
    
    if args.test:
        # Test mode - use minimal limits
        args.max_repos = args.max_repos or 3
        PROCESSING_LIMITS["max_functions_per_repo"] = 50
        PROCESSING_LIMITS["max_total_functions"] = 200
        logger.info("Running in test mode with reduced limits")
       
    # Validate arguments
    phase_args = [args.discover, args.clone, args.extract, args.discover_only, args.clone_only, args.extract_only, args.resume]
    if not any(phase_args):
        parser.error("Must specify at least one phase: --discover, --clone, --extract, or their -only variants")
    
    exclusive_pairs = [
        (args.discover_only, args.clone, "Cannot use --discover-only with --clone"),
        (args.discover_only, args.extract, "Cannot use --discover-only with --extract"),
        (args.clone_only, args.discover, "Cannot use --clone-only with --discover"),
        (args.extract_only, args.discover, "Cannot use --extract-only with --discover"),
        (args.extract_only, args.clone, "Cannot use --extract-only with --clone")
    ]
    
    for arg1, arg2, message in exclusive_pairs:
        if arg1 and arg2:
            parser.error(message)

    # Validate configuration
    if not validate_config():
        logger.error("Configuration validation failed")
        return 1
    
    logger.info("=" * 60)
    logger.info("ERLANG CORPUS SCRAPER STARTING")
    logger.info("=" * 60)
    logger.info(f"Command line: {' '.join(sys.argv)}")
    logger.info(f"Start time: {datetime.now().isoformat()}")
    logger.info(f"Configuration: max_repos={args.max_repos}, min_score={PARSER_CONFIG['min_score']}")
    
    try:
        repositories = []
        clone_results = []
        functions = []
        
        # Handle resume functionality
        if args.resume:
            checkpoint = load_checkpoint()
            if checkpoint:
                logger.info(f"Resuming from checkpoint: {checkpoint['stage']}")
                # TODO: Implement specific resume logic based on checkpoint stage
            else:
                logger.warning("No checkpoint found, starting from beginning")
        
        # Phase 1: Discovery
        if args.discover or args.discover_only:
            logger.info("Phase 1: Repository Discovery")
            repositories = discover_repositories(args)
            logger.info(f"Discovery complete: {len(repositories)} repositories found")
        
        # Phase 2: Cloning
        if args.clone or args.clone_only:
            logger.info("Phase 2: Repository Cloning")
            
            # Load repositories if we didn't discover them in this run
            if not repositories:
                repo_file = get_output_path(OUTPUT_CONFIG["repositories_file"])
                repositories = load_repositories_from_file(repo_file)
                if not repositories:
                    logger.error("No repositories found. Run discovery first.")
                    return 1
                if args.max_repos:
                    repositories = repositories[:args.max_repos]
            
            clone_results = clone_repositories(repositories, args)
            successful_clones = len([r for r in clone_results if r.success])
            logger.info(f"Cloning complete: {successful_clones}/{len(clone_results)} repositories cloned")
        
        # Phase 3: Function Extraction
        if args.extract or args.extract_only:
            logger.info("Phase 3: Function Extraction")
            
            # Load clone results if we didn't clone in this run
            if not clone_results:
                clone_file = get_output_path("clone_results.json")
                clone_results = load_clone_results_from_file(clone_file)
                if not clone_results:
                    logger.error("No clone results found. Run cloning first.")
                    return 1
            
            functions = extract_functions(clone_results, args)
            logger.info(f"Function extraction complete: {len(functions)} functions extracted")
        
        # Generate final summary
        if repositories and clone_results and functions:
            generate_corpus_summary(repositories, clone_results, functions)
        
        logger.info("=" * 60)
        logger.info("ERLANG CORPUS SCRAPER COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        # Next steps message
        if functions:
            logger.info(f"Corpus ready: {len(functions)} functions in {get_output_path(OUTPUT_CONFIG['functions_file'])}")
            logger.info("Next step: Use functions.jsonl for GraphCodeBERT training")
        elif clone_results:
            successful_repos = len([r for r in clone_results if r.success])
            logger.info(f"Next step: Run function extraction on {successful_repos} cloned repositories")
            logger.info("Command: python main.py --extract-only")
        elif repositories:
            logger.info(f"Next step: Clone {len(repositories)} discovered repositories")
            logger.info("Command: python main.py --clone-only")
        
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
