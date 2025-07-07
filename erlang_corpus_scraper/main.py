#!/usr/bin/env python3
"""
Main entry point for Erlang corpus scraper and GraphCodeBERT data preparation.
Updated to use enhanced function extractor directly with dataset (no transformer layer).
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import modules
from config import (
    OUTPUT_CONFIG, GITHUB_CONFIG, DISCOVERY_CONFIG, CLONE_CONFIG, PARSER_CONFIG, FUNCTION_SCORING,
    GRAPHCODEBERT_CONFIG, LOGGING_CONFIG, get_output_path, get_graphcodebert_output_path
)
from scrapers.github_scraper import GitHubScraper, RepositoryInfo
from scrapers.repo_cloner import RepoCloner, CloneResult
from parsers.function_extractor import FunctionExtractor, ErlangFunction

# Training components (optional)
try:
    from train.dataset import create_split_dataloaders, split_and_save_functions
    from train.train import GraphCodeBERTTrainer
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False
    print("PEFT/LoRA not available - falling back to full fine-tuning")

# Setup logging
def setup_logging(debug: bool = False, log_file: Optional[str] = None):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    log_format = LOGGING_CONFIG['format']
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    elif not debug:
        # Default log file for non-debug runs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_log = log_dir / f"scraper_{timestamp}.log"
        handlers.append(logging.FileHandler(default_log))
    
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers
    )
    
    # Suppress noisy loggers
    for noisy_logger in ['urllib3', 'requests', 'git']:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

def create_argument_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Erlang corpus scraper and GraphCodeBERT data preparation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline: discover -> clone -> extract -> prepare training data
  python main.py --github-token YOUR_TOKEN
  
  # Run only specific steps
  python main.py --clone-only
  python main.py --extract-only
  python main.py --prepare-only
  python main.py --train-only
  
  # Prepare training data from existing functions.jsonl
  python main.py --prepare-only --functions-file my_functions.jsonl
  
  # Train with custom files and LoRA
  python main.py --train-only --train-file my_train.jsonl --use-lora
        """
    )
    
    # Main operation modes
    parser.add_argument('--discover-only', action='store_true',
                        help='Only discover repositories (skip cloning and extraction)')
    parser.add_argument('--clone-only', action='store_true',
                        help='Only clone repositories (requires existing repositories.json)')
    parser.add_argument('--extract-only', action='store_true',
                        help='Only extract functions (requires cloned repositories)')
    parser.add_argument('--prepare-only', action='store_true',
                        help='Only prepare training data (split functions into train/val/test)')
    parser.add_argument('--train-only', action='store_true',
                        help='Only run GraphCodeBERT training (requires training data)')
    
    # GitHub API settings
    parser.add_argument('--github-token', type=str,
                        help='GitHub API token for higher rate limits')
    parser.add_argument('--max-repos', type=int, default=DISCOVERY_CONFIG['max_total_repos'],
                        help='Maximum number of repositories to discover')
    parser.add_argument('--min-stars', type=int, default=DISCOVERY_CONFIG['min_stars'],
                        help='Minimum GitHub stars for repository inclusion')
    parser.add_argument('--use-repos', nargs='+',
                        help='Use specific repositories (e.g., --use-repos ninenines/cowboy)')
    
    # Cloning settings
    parser.add_argument('--max-concurrent-clones', type=int, default=CLONE_CONFIG['max_concurrent_clones'],
                        help='Maximum concurrent clone operations')
    parser.add_argument('--clone-timeout', type=int, default=CLONE_CONFIG['clone_timeout'],
                        help='Timeout for clone operations (seconds)')
    
    # Function extraction settings
    parser.add_argument('--max-parser-workers', type=int, default=PARSER_CONFIG['max_concurrent_parsers'],
                        help='Maximum concurrent parser workers')
    parser.add_argument('--min-function-score', type=float, default=FUNCTION_SCORING['min_score'],
                        help='Minimum function quality score')
    
    # Data files
    parser.add_argument('--functions-file', type=str,
                        help='Path to functions JSONL file (for prepare-only)')
    parser.add_argument('--graphcodebert-output', type=str,
                        help='Output directory for GraphCodeBERT training data')
    parser.add_argument('--no-split', action='store_true',
                        help='Don\'t split data into train/val/test sets')
    
    # GraphCodeBERT settings
    parser.add_argument('--max-code-length', type=int,
                        help='Maximum code sequence length')
    parser.add_argument('--max-dfg-length', type=int,
                        help='Maximum DFG edges')
    
    # Training settings (if available)
    if TRAINING_AVAILABLE:
        parser.add_argument('--train-file', type=str,
                            help='Path to training JSONL file')
        parser.add_argument('--val-file', type=str,
                            help='Path to validation JSONL file')
        parser.add_argument('--model-output-dir', type=str,
                            help='Directory to save trained model')
        parser.add_argument('--use-lora', action='store_true',
                            help='Use LoRA adaptation instead of full fine-tuning')
    
    # Logging and debugging
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--log-file', type=str,
                        help='Log file path')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce logging output')
    
    return parser

def validate_args(args):
    """Validate command line arguments."""
    # Check for mutually exclusive modes
    modes = [args.discover_only, args.clone_only, args.extract_only, 
             args.prepare_only, args.train_only]
    active_modes = sum(modes)
    
    if active_modes > 1:
        logger.error("Only one operation mode can be specified at a time")
        return False
    
    # Validate training arguments
    if args.train_only and not TRAINING_AVAILABLE:
        logger.error("Training dependencies not available. Install with: pip install -r requirements_training.txt")
        return False
    
    # Validate file arguments
    if args.prepare_only and args.functions_file and not os.path.exists(args.functions_file):
        logger.error(f"Functions file not found: {args.functions_file}")
        return False
    
    if args.train_only:
        if not args.train_file:
            logger.error("--train-file required for training mode")
            return False
        if not os.path.exists(args.train_file):
            logger.error(f"Training file not found: {args.train_file}")
            return False
    
    return True

def load_repositories_from_file(repo_file: str) -> List[RepositoryInfo]:
    """Load repositories from JSON file."""
    try:
        with open(repo_file, 'r') as f:
            repo_data = json.load(f)
        
        repositories = []
        
        # Handle different formats
        if isinstance(repo_data, list):
            repo_list = repo_data
        elif isinstance(repo_data, dict):
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
    """Load clone results from JSON file."""
    try:
        with open(clone_file, 'r') as f:
            clone_data = json.load(f)
        
        clone_results = []
        
        # Handle different JSON formats
        if isinstance(clone_data, list):
            results_list = clone_data
        elif isinstance(clone_data, dict):
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

def discover_repositories(args) -> List[RepositoryInfo]:
    """Discover repositories using GitHub API."""
    logger.info("=" * 60)
    logger.info("STEP 1: DISCOVERING REPOSITORIES")
    logger.info("=" * 60)
    
    scraper = GitHubScraper()
    
    if args.use_repos:
        # Use specific repositories
        repositories = []
        for repo_name in args.use_repos:
            try:
                repo_info = scraper.get_repository_info(repo_name)
                if repo_info:
                    repositories.append(repo_info)
                    logger.info(f"✓ Added repository: {repo_name}")
                else:
                    logger.warning(f"✗ Repository not found or not accessible: {repo_name}")
            except Exception as e:
                logger.error(f"✗ Failed to get info for {repo_name}: {e}")
    else:
        # Discover repositories
        repositories = scraper.discover_repositories(
            max_repos=args.max_repos,
            min_stars=args.min_stars
        )
    
    if repositories:
        logger.info(f"✓ Discovered {len(repositories)} repositories")
        scraper.save_repositories(repositories)
        return repositories
    else:
        logger.warning("No repositories discovered")
        return []

def clone_repositories(repositories: List[RepositoryInfo], args) -> List[CloneResult]:
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
        cloner.save_clone_results(clone_results)
        successful = len([r for r in clone_results if r.success])
        logger.info(f"✓ Successfully cloned {successful}/{len(clone_results)} repositories")
        return clone_results
    else:
        logger.warning("No repositories cloned")
        return []

def extract_functions(clone_results: List[CloneResult], args) -> List[ErlangFunction]:
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
        extractor.save_functions(functions)
        logger.info(f"✓ Extracted {len(functions)} functions")
        return functions
    else:
        logger.warning("No functions extracted")
        return []

def prepare_training_data(args) -> tuple:
    """Prepare training data by splitting functions into train/val/test sets."""
    logger.info("=" * 60)
    logger.info("STEP 4: PREPARING TRAINING DATA")
    logger.info("=" * 60)
    
    # Determine functions file
    functions_file = args.functions_file or get_output_path(OUTPUT_CONFIG['functions_file'])
    
    if not os.path.exists(functions_file):
        logger.error(f"Functions file not found: {functions_file}")
        return None
    
    # Determine output directory
    output_dir = args.graphcodebert_output or get_graphcodebert_output_path("")
    
    if args.no_split:
        logger.info("Skipping data split (--no-split specified)")
        return functions_file, None, None
    else:
        # Split data into train/val/test
        logger.info("Splitting functions into train/validation/test sets")
        
        # Use split ratios from config
        train_ratio = GRAPHCODEBERT_CONFIG['data_splits']['train_ratio']
        val_ratio = GRAPHCODEBERT_CONFIG['data_splits']['val_ratio']
        test_ratio = GRAPHCODEBERT_CONFIG['data_splits']['test_ratio']
        
        train_file, val_file, test_file = split_and_save_functions(
            functions_file=functions_file,
            output_dir=output_dir,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=42
        )
        
        logger.info(f"✓ Training data prepared:")
        logger.info(f"  Training: {train_file}")
        logger.info(f"  Validation: {val_file}")
        logger.info(f"  Test: {test_file}")
        
        return train_file, val_file, test_file

def train_model(args) -> bool:
    """Train GraphCodeBERT model."""
    if not TRAINING_AVAILABLE:
        logger.error("Training dependencies not available")
        return False
    
    logger.info("=" * 60)
    logger.info("STEP 5: TRAINING GRAPHCODEBERT MODEL")
    logger.info("=" * 60)
    
    try:
        trainer = GraphCodeBERTTrainer(use_lora=args.use_lora)
        
        trainer.train(
            train_file=args.train_file,
            val_file=args.val_file,
            output_dir=args.model_output_dir
        )
        
        logger.info("✓ Training completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False

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
        
        # Initialize variables
        repositories = []
        clone_results = []
        functions = []
        
        # Execute pipeline steps based on arguments
        if args.train_only:
            # Training only mode
            success = train_model(args)
            return 0 if success else 1
        
        elif args.prepare_only:
            # Prepare training data only
            result = prepare_training_data(args)
            return 0 if result else 1
        
        elif args.extract_only:
            # Extract functions only (requires clone results)
            clone_results_file = get_output_path(OUTPUT_CONFIG['clone_results_file'])
            if os.path.exists(clone_results_file):
                clone_results = load_clone_results_from_file(clone_results_file)
                if clone_results:
                    functions = extract_functions(clone_results, args)
                else:
                    logger.error("No clone results loaded")
                    return 1
            else:
                logger.error(f"Clone results file not found: {clone_results_file}")
                logger.error("Run cloning step first: python main.py --clone-only")
                return 1
        
        elif args.clone_only:
            # Clone repositories only (requires repository list)
            repos_file = get_output_path(OUTPUT_CONFIG['repos_file'])
            if os.path.exists(repos_file):
                repositories = load_repositories_from_file(repos_file)
                if repositories:
                    clone_results = clone_repositories(repositories, args)
                else:
                    logger.error("No repositories loaded")
                    return 1
            else:
                logger.error(f"Repositories file not found: {repos_file}")
                logger.error("Run discovery step first: python main.py --discover-only")
                return 1
        
        elif args.discover_only:
            # Discover repositories only
            repositories = discover_repositories(args)
        
        else:
            # Full pipeline or remaining steps
            
            # Step 1: Discover repositories (if needed)
            repos_file = get_output_path(OUTPUT_CONFIG['repos_file'])
            if os.path.exists(repos_file):
                logger.info(f"Loaded existing repositories from {repos_file}")
                repositories = load_repositories_from_file(repos_file)
            else:
                repositories = discover_repositories(args)
            
            if not repositories:
                logger.error("No repositories available for cloning")
                return 1
            
            # Step 2: Clone repositories (if needed)
            clone_results_file = get_output_path(OUTPUT_CONFIG['clone_results_file'])
            if os.path.exists(clone_results_file):
                logger.info(f"Loaded existing clone results from {clone_results_file}")
                clone_results = load_clone_results_from_file(clone_results_file)
            else:
                clone_results = clone_repositories(repositories, args)
            
            if not clone_results:
                logger.error("No repositories cloned successfully")
                return 1
            
            # Step 3: Extract functions (if needed)
            functions_file = get_output_path(OUTPUT_CONFIG['functions_file'])
            if os.path.exists(functions_file):
                logger.info(f"Loaded existing functions from {functions_file}")
                # Functions already extracted
            else:
                functions = extract_functions(clone_results, args)
                if not functions:
                    logger.error("No functions extracted")
                    return 1
            
            # Step 4: Prepare training data
            train_file, val_file, test_file = prepare_training_data(args)
            if not train_file:
                logger.error("Failed to prepare training data")
                return 1
            
            logger.info("=" * 60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info("Next steps:")
            logger.info("1. Review the prepared training data")
            if TRAINING_AVAILABLE:
                logger.info(f"2. Start training: python main.py --train-only --train-file {train_file} --val-file {val_file}")
            else:
                logger.info("2. Install training dependencies: pip install -r requirements_training.txt")
                logger.info("3. Then start training with the prepared data files")
        
        # Log next steps for partial runs
        if args.discover_only:
            logger.info("Next step: Clone discovered repositories")
            logger.info("Command: python main.py --clone-only")
        elif args.clone_only:
            logger.info("Next step: Extract functions from cloned repositories")
            logger.info("Command: python main.py --extract-only")
        elif args.extract_only:
            logger.info("Next step: Prepare training data")
            logger.info("Command: python main.py --prepare-only")
        elif args.prepare_only:
            if TRAINING_AVAILABLE:
                logger.info("Next step: Start training")
                logger.info("Command: python main.py --train-only --train-file <train_file> --val-file <val_file>")
            else:
                logger.info("Next step: Install training dependencies and start training")
        
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
