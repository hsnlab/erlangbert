#!/usr/bin/env python3
"""
Main entry point for Erlang corpus scraper and GraphCodeBERT data preparation.
Enhanced to support GraphCodeBERT data transformation.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import asdict

# Import modules
from config import (
    OUTPUT_CONFIG, GITHUB_CONFIG, DISCOVERY_CONFIG, CLONE_CONFIG, PARSER_CONFIG, FUNCTION_SCORING,
    GRAPHCODEBERT_CONFIG, LOGGING_CONFIG, get_output_path, get_graphcodebert_output_path
)
from scrapers.github_scraper import GitHubScraper, RepositoryInfo
from scrapers.repo_cloner import RepoCloner, CloneResult
from parsers.function_extractor import FunctionExtractor, ErlangFunction
from transformer.graphcodebert_transformer import GraphCodeBERTTransformer

# Setup logging
def setup_logging(debug: bool = False, log_file: Optional[str] = None):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else getattr(logging, LOGGING_CONFIG['level'])
    format_str = LOGGING_CONFIG['format']
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file or LOGGING_CONFIG['file_logging']:
        log_file = log_file or get_output_path(LOGGING_CONFIG['log_file'])
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_str))
        handlers.append(file_handler)
        
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=handlers,
        force=True
    )

logger = logging.getLogger(__name__)

def discover_repositories(args: argparse.Namespace) -> List[RepositoryInfo]:
    """Discover Erlang repositories on GitHub."""
    logger.info("=" * 60)
    logger.info("STEP 1: DISCOVERING ERLANG REPOSITORIES")
    logger.info("=" * 60)

    if args.use_repos:
        logger.info(f"Using specified repositories: {args.use_repos}")
        
        from scrapers.github_scraper import RepositoryInfo
        
        repositories = []
        for repo_name in args.use_repos:
            # Create proper RepositoryInfo objects
            repo_info = RepositoryInfo(
                name=repo_name.split('/')[-1],
                full_name=repo_name,
                description=f'Repository {repo_name}',
                stars=0,
                forks=0,
                size_kb=1000,
                language='Erlang',
                languages={'Erlang': 100000},
                created_at='2020-01-01T00:00:00Z',
                updated_at='2024-01-01T00:00:00Z',
                clone_url=f'https://github.com/{repo_name}.git',
                html_url=f'https://github.com/{repo_name}',
                archived=False,
                has_wiki=False,
                has_issues=True,
                erlang_percentage=1.0,
                quality_score=50.0
            )
            repositories.append(repo_info)
            
        # Convert to dicts for JSON serialization
        output_file = get_output_path(OUTPUT_CONFIG['repos_file'])
        repo_dicts = [asdict(repo) for repo in repositories]
        
        with open(output_file, 'w') as f:
            json.dump({
                "discovery_date": datetime.now().isoformat(),
                "total_repositories": len(repositories),
                "repositories": repo_dicts,
                "source": "command_line_--use-repos"
            }, f, indent=2)
            
        return repositories

    scraper = GccitHubScraper(
        token=args.github_token,
        max_repos=args.max_repos,
        min_stars=args.min_stars
    )
    
    repositories = scraper.discover_repositories()
    
    if repositories:
        output_file = get_output_path(OUTPUT_CONFIG['repos_file'])
        with open(output_file, 'w') as f:
            json.dump(repositories, f, indent=2)
            logger.info(f"Saved {len(repositories)} repositories to {output_file}")
            
    return repositories

def clone_repositories(repositories: List[Dict[str, Any]], args: argparse.Namespace) -> List[CloneResult]:
    """Clone discovered repositories."""
    logger.info("=" * 60)
    logger.info("STEP 2: CLONING REPOSITORIES")
    logger.info("=" * 60)
    
    cloner = RepoCloner(
        max_concurrent=args.max_concurrent_clones,
        timeout=args.clone_timeout
    )
    
    clone_results = cloner.clone_repositories(repositories)
    
    if clone_results:
        # Save results
        output_file = get_output_path(OUTPUT_CONFIG['clone_results_file'])
        results_data = [result.to_dict() for result in clone_results]
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
            
        # Log summary
        successful = len([r for r in clone_results if r.success])
        logger.info(f"Cloning complete: {successful}/{len(clone_results)} successful")
        
    return clone_results

def extract_functions(clone_results: List[CloneResult], args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Extract functions from cloned repositories."""
    logger.info("=" * 60)
    logger.info("STEP 3: EXTRACTING FUNCTIONS")
    logger.info("=" * 60)
    
    extractor = FunctionExtractor(
        max_workers=args.max_parser_workers,
        min_score=args.min_function_score
    )
    
    functions = extractor.extract_from_clone_results(clone_results)
    
    if functions:
        output_file = get_output_path(OUTPUT_CONFIG['functions_file'])
        with open(output_file, 'w', encoding='utf-8') as f:
            for func in functions:
                # Use asdict to convert the dataclass to a dictionary
                func_dict = asdict(func)
                json.dump(func_dict, f, ensure_ascii=False)
                f.write('\n')

        logger.info(f"Saved {len(functions)} functions to {output_file}")
        
    return functions

def transform_to_graphcodebert(args: argparse.Namespace) -> bool:
    """Transform functions to GraphCodeBERT format."""
    logger.info("=" * 60)
    logger.info("STEP 4: TRANSFORMING TO GRAPHCODEBERT FORMAT")
    logger.info("=" * 60)
    
    # Get input file
    functions_file = args.functions_file or get_output_path(OUTPUT_CONFIG['functions_file'])
    
    if not os.path.exists(functions_file):
        logger.error(f"Functions file not found: {functions_file}")
        return False
    
    # Get output directory
    output_dir = args.graphcodebert_output or get_graphcodebert_output_path('')
    
    # Create transformer
    transformer = GraphCodeBERTTransformer(config=GRAPHCODEBERT_CONFIG)
    
    # Perform transformation
    success = transformer.transform(
        functions_file=functions_file,
        output_dir=output_dir,
        split_data=not args.no_split
    )
    
    if success:
        logger.info(f"GraphCodeBERT data saved to: {output_dir}")
        if not args.no_split:
            logger.info("Data split into train/valid/test sets")
            logger.info("Files created:")
            for split_file in ['train.jsonl', 'valid.jsonl', 'test.jsonl', 'dataset_stats.json']:
                file_path = os.path.join(output_dir, split_file)
                if os.path.exists(file_path):
                    logger.info(f"  - {file_path}")
        else:
            logger.info("Files created:")
            for file_name in ['all_examples.jsonl', 'dataset_stats.json']:
                file_path = os.path.join(output_dir, file_name)
                if os.path.exists(file_path):
                    logger.info(f"  - {file_path}")
                    
    return success

def generate_corpus_summary(repositories: List[RepositoryInfo], 
                          clone_results: List[CloneResult],
                          functions: List[ErlangFunction]):
    """Generate summary of the corpus creation process."""
    logger.info("=" * 60)
    logger.info("GENERATING CORPUS SUMMARY")
    logger.info("=" * 60)
    
    successful_clones = [r for r in clone_results if r.success]
    failed_clones = [r for r in clone_results if not r.success]
    
    # Calculate function statistics - FIXED: Use attribute access
    total_functions = len(functions)
    exported_functions = len([f for f in functions if f.is_exported])
    documented_functions = len([f for f in functions if f.docstring])
    
    # Repository statistics
    repo_stats = {}
    for func in functions:
        repo_name = func.repo_name  # FIXED: Use attribute access
        if repo_name not in repo_stats:
            repo_stats[repo_name] = 0
        repo_stats[repo_name] += 1
    
    # Score distribution - FIXED: Use attribute access
    scores = [f.score for f in functions]
    score_stats = {
        'min': min(scores) if scores else 0,
        'max': max(scores) if scores else 0,
        'avg': sum(scores) / len(scores) if scores else 0,
        'median': sorted(scores)[len(scores) // 2] if scores else 0
    }
    
    from datetime import datetime  # Add missing import
    
    summary = {
        'generation_timestamp': datetime.now().isoformat(),
        'repositories': {
            'discovered': len(repositories),
            'cloned_successfully': len(successful_clones),
            'clone_failures': len(failed_clones),
            'top_repos_by_functions': sorted(repo_stats.items(), key=lambda x: x[1], reverse=True)[:10]
        },
        'functions': {
            'total_extracted': total_functions,
            'exported_functions': exported_functions,
            'documented_functions': documented_functions,
            'score_statistics': score_stats,
            'avg_functions_per_repo': total_functions / len(successful_clones) if successful_clones else 0
        },
        'corpus_quality': {
            'export_ratio': exported_functions / total_functions if total_functions else 0,
            'documentation_ratio': documented_functions / total_functions if total_functions else 0,
            'avg_score': score_stats['avg']
        }
    }
    
    # Save summary
    output_file = get_output_path(OUTPUT_CONFIG['corpus_summary_file'])
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Corpus summary saved to: {output_file}")
    logger.info(f"Total functions extracted: {total_functions}")
    logger.info(f"Average score: {score_stats['avg']:.2f}")

def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Erlang corpus scraper and GraphCodeBERT data preparation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline: discover -> clone -> extract -> transform
  python main.py --github-token YOUR_TOKEN
  
  # Run only specific steps
  python main.py --clone-only
  python main.py --extract-only
  python main.py --transform-only
  
  # Transform existing functions.jsonl to GraphCodeBERT format
  python main.py --transform-only --functions-file my_functions.jsonl
  
  # Transform without splitting data
  python main.py --transform-only --no-split
        """
    )
    
    # Main operation modes
    parser.add_argument('--discover-only', action='store_true',
                        help='Only discover repositories (skip cloning and extraction)')
    parser.add_argument('--clone-only', action='store_true',
                        help='Only clone repositories (requires existing repositories.json)')
    parser.add_argument('--extract-only', action='store_true',
                        help='Only extract functions (requires cloned repositories)')
    parser.add_argument('--transform-only', action='store_true',
                        help='Only transform to GraphCodeBERT format (requires functions.jsonl)')
    
    # GitHub API settings
    parser.add_argument('--github-token', type=str,
                        help='GitHub API token (or set GITHUB_TOKEN environment variable)')
    parser.add_argument('--max-repos', type=int, default=DISCOVERY_CONFIG['max_total_repos'],
                        help='Maximum number of repositories to discover')
    parser.add_argument('--min-stars', type=int, default=DISCOVERY_CONFIG['min_stars'],
                        help='Minimum star count for repositories')
    parser.add_argument('--use-repos', nargs='+', metavar='REPO',
                        help='Use specific repositories instead of discovery (e.g., ninenines/cowboy erlang/otp)')
    
    # Cloning settings
    parser.add_argument('--max-concurrent-clones', type=int, default=CLONE_CONFIG['max_concurrent_clones'],
                        help='Maximum concurrent clone operations')
    parser.add_argument('--clone-timeout', type=int, default=CLONE_CONFIG['clone_timeout'],
                        help='Timeout for each clone operation (seconds)')
    
    # Function extraction settings
    parser.add_argument('--max-parser-workers', type=int, default=PARSER_CONFIG['max_concurrent_parsers'],
                        help='Maximum concurrent parser workers')
    parser.add_argument('--min-function-score', type=float, default=FUNCTION_SCORING['min_score'],
                        help='Minimum score for functions to include')
    
    # GraphCodeBERT transformation settings
    parser.add_argument('--functions-file', type=str,
                        help='Input functions.jsonl file (default: output/functions.jsonl)')
    parser.add_argument('--graphcodebert-output', type=str,
                        help='Output directory for GraphCodeBERT data (default: output/graphcodebert_data)')
    parser.add_argument('--no-split', action='store_true',
                        help='Do not split data into train/val/test sets')
    parser.add_argument('--max-code-length', type=int, default=GRAPHCODEBERT_CONFIG['max_code_length'],
                        help='Maximum code token length for GraphCodeBERT')
    parser.add_argument('--max-dfg-length', type=int, default=GRAPHCODEBERT_CONFIG['max_dfg_length'],
                        help='Maximum data flow graph edge count')
    
    # Logging and debugging
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--log-file', type=str,
                        help='Custom log file path')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')
    
    return parser

def validate_args(args: argparse.Namespace) -> bool:
    """Validate command line arguments."""
    errors = []

    if not any([args.clone_only, args.extract_only, args.transform_only]):
        if not args.github_token and not os.getenv('GITHUB_TOKEN') and not args.use_repos:
            # Just warn about rate limits, don't block execution
            logger = logging.getLogger(__name__)
            logger.warning("No GitHub token provided - API rate limits will be restrictive (60 requests/hour)")
            logger.warning("For better performance, set GITHUB_TOKEN environment variable or use --github-token")
    
    if args.use_repos:
        # Validate repository format
        for repo in args.use_repos:
            if '/' not in repo:
                Errors.append(f"Invalid repository format '{repo}'. Use format 'owner/repo' (e.g., ninenines/cowboy)")
                
        # --use-repos can work without GitHub token for cloning/extraction only
        if any([args.clone_only, args.extract_only, args.transform_only]):
            pass  # No token needed for these operations
        
    # Check for existing files when needed
    if args.clone_only:
        repos_file = get_output_path(OUTPUT_CONFIG['repos_file'])
        if not os.path.exists(repos_file):
            errors.append(f"Repository file not found: {repos_file}. Run discovery first.")
            
    if args.extract_only:
        clone_results_file = get_output_path(OUTPUT_CONFIG['clone_results_file'])
        if not os.path.exists(clone_results_file):
            errors.append(f"Clone results file not found: {clone_results_file}. Run cloning first.")
            
    if args.transform_only:
        functions_file = args.functions_file or get_output_path(OUTPUT_CONFIG['functions_file'])
        if not os.path.exists(functions_file):
            errors.append(f"Functions file not found: {functions_file}. Run extraction first.")
            
    # Validate numeric arguments
    if args.max_repos <= 0:
        errors.append("max-repos must be positive")
        
    if args.min_stars < 0:
        errors.append("min-stars cannot be negative")
        
    if args.max_concurrent_clones <= 0:
        errors.append("max-concurrent-clones must be positive")
        
    if args.clone_timeout <= 0:
        errors.append("clone-timeout must be positive")
        
    if errors:
        logger.error("Argument validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    return True

# Helper functions for loading data from files:
def load_repositories_from_file(repo_file: str) -> List[RepositoryInfo]:
    """Load repositories from JSON file."""
    try:
        with open(repo_file, 'r') as f:
            repo_data = json.load(f)
        
        from scrapers.github_scraper import RepositoryInfo
        repositories = []
        
        # Handle different formats
        if isinstance(repo_data, list):
            # Direct list format
            repo_list = repo_data
        elif isinstance(repo_data, dict):
            # Dictionary format with repositories key
            repo_list = repo_data.get('repositories', [])
        else:
            logger.error(f"Unexpected repo data format: {type(repo_data)}")
            return []
        
        for repo_dict in repo_list:
            try:
                repo_info = RepositoryInfo(**repo_dict)
                repositories.append(repo_info)
            except Exception as e:
                logger.warning(f"Failed to load repository {repo_dict.get('full_name', 'unknown')}: {e}")
        
        logger.info(f"Loaded {len(repositories)} repositories from {repo_file}")
        return repositories
        
    except Exception as e:
        logger.error(f"Failed to load repositories from {repo_file}: {e}")
        return []

def load_clone_results_from_file(clone_file: str) -> List[CloneResult]:
    """Load clone results from JSON file - handle multiple formats."""
    try:
        with open(clone_file, 'r') as f:
            clone_data = json.load(f)
        
        from scrapers.repo_cloner import CloneResult
        clone_results = []
        
        # Handle different JSON formats
        if isinstance(clone_data, list):
            # Direct list of results (legacy format)
            results_list = clone_data
        elif isinstance(clone_data, dict):
            # Dictionary with metadata - try different key names
            results_list = (clone_data.get('results') or 
                          clone_data.get('clone_results') or 
                          clone_data.get('clone_data', []))
        else:
            logger.error(f"Unexpected clone data format: {type(clone_data)}")
            return []
        
        # Convert each result to CloneResult object
        for result_dict in results_list:
            try:
                clone_result = CloneResult.from_dict(result_dict)
                clone_results.append(clone_result)
            except Exception as e:
                logger.warning(f"Failed to load clone result: {e}")
                continue
        
        logger.info(f"Loaded {len(clone_results)} clone results from {clone_file}")
        return clone_results
        
    except Exception as e:
        logger.error(f"Failed to load clone results from {clone_file}: {e}")
        return []

def main() -> int:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging first
    setup_logging(debug=args.debug, log_file=args.log_file)
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    logger.info("Starting Erlang corpus scraper and GraphCodeBERT preparation")
    logger.info(f"Arguments: {vars(args)}")
    
    # Validate arguments
    if not validate_args(args):
        return 1
    
    try:
        # Set GitHub token
        if args.github_token:
            os.environ['GITHUB_TOKEN'] = args.github_token
        
        # Update GraphCodeBERT config with command line overrides
        if args.max_code_length:
            GRAPHCODEBERT_CONFIG['max_code_length'] = args.max_code_length
        if args.max_dfg_length:
            GRAPHCODEBERT_CONFIG['max_dfg_length'] = args.max_dfg_length
        
        # Initialize variables - THIS IS THE KEY FIX
        repositories = []
        clone_results = []
        functions = []
        
        # Execute pipeline steps based on arguments
        if args.transform_only:
            # Only transform to GraphCodeBERT format
            success = transform_to_graphcodebert(args)
            if not success:
                logger.error("GraphCodeBERT transformation failed")
                return 1
        
        elif args.extract_only:
            # Only extract functions
            clone_results_file = get_output_path(OUTPUT_CONFIG['clone_results_file'])
            if not os.path.exists(clone_results_file):
                logger.error(f"Clone results file not found: {clone_results_file}")
                return 1
            
            # Load clone results from file
            clone_results = load_clone_results_from_file(clone_results_file)
            if not clone_results:
                logger.error("No valid clone results loaded")
                return 1
            
            functions = extract_functions(clone_results, args)
            if not functions:
                logger.error("Function extraction failed")
                return 1

        elif args.clone_only:
            # Only clone repositories - NEED TO LOAD REPOSITORIES
            repos_file = get_output_path(OUTPUT_CONFIG['repos_file'])
            if not os.path.exists(repos_file):
                logger.error(f"Repository file not found: {repos_file}")
                return 1
            
            # Load repositories from file
            with open(repos_file, 'r') as f:
                repo_data = json.load(f)
            
            # Convert to RepositoryInfo objects
            from scrapers.github_scraper import RepositoryInfo
            repositories = []
            for repo_dict in repo_data.get('repositories', []):
                try:
                    repo_info = RepositoryInfo(**repo_dict)
                    repositories.append(repo_info)
                except Exception as e:
                    logger.warning(f"Failed to load repository: {e}")
            
            if not repositories:
                logger.error("No repositories found. Run discovery first.")
                return 1
            
            clone_results = clone_repositories(repositories, args)
            if not any(r.success for r in clone_results):
                logger.error("All repository clones failed")
                return 1
        
        elif args.discover_only:
            # Only discover repositories
            repositories = discover_repositories(args)
            if not repositories:
                logger.error("Repository discovery failed")
                return 1
        
        else:
            # Full pipeline
            repositories = discover_repositories(args)
            if not repositories:
                logger.error("Repository discovery failed, aborting pipeline")
                return 1
            
            clone_results = clone_repositories(repositories, args)
            if not any(r.success for r in clone_results):
                logger.error("All repository clones failed, aborting pipeline")
                return 1
            
            functions = extract_functions(clone_results, args)
            if not functions:
                logger.error("Function extraction failed, aborting pipeline")
                return 1
            
            # Transform to GraphCodeBERT format
            logger.info("Proceeding with GraphCodeBERT transformation...")
            success = transform_to_graphcodebert(args)
            if not success:
                logger.warning("GraphCodeBERT transformation failed, but function extraction succeeded")
        
        # Generate final summary (if we have data)
        if repositories and clone_results and functions:
            generate_corpus_summary(repositories, clone_results, functions)
        
        logger.info("=" * 60)
        logger.info("ERLANG CORPUS SCRAPER COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        # Next steps message
        if args.transform_only:
            logger.info("GraphCodeBERT data transformation complete")
            output_dir = args.graphcodebert_output or get_graphcodebert_output_path('')
            logger.info(f"Ready for training with data in: {output_dir}")
        elif functions:
            logger.info(f"Corpus ready: {len(functions)} functions in {get_output_path(OUTPUT_CONFIG['functions_file'])}")
            logger.info("Next step: Transform to GraphCodeBERT format")
            logger.info("Command: python main.py --transform-only")
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
