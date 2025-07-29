#!/usr/bin/env python3
"""
Erlang GraphCodeBERT Training Pipeline

Complete pipeline for creating Erlang-specialized GraphCodeBERT models:
1. Repository discovery and cloning
2. Function extraction with data flow graphs
3. Training data preparation
4. GraphCodeBERT model training
5. Model evaluation

Updated with clean subcommand interface.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Import configuration
from config import (
    DISCOVERY_CONFIG, CLONE_CONFIG, PARSER_CONFIG,
    OUTPUT_CONFIG, GRAPHCODEBERT_CONFIG, FINETUNING_CONFIG,
    HARDWARE_CONFIG, LOGGING_CONFIG
)

# Import pipeline components
from scrapers.github_scraper import GitHubScraper, RepositoryInfo
from scrapers.repo_cloner import RepoCloner, CloneResult
from parsers.function_extractor import FunctionExtractor, ErlangFunction
from config import get_output_path, get_graphcodebert_output_path

# Import training components
try:
    from train.train import GraphCodeBERTTrainer
    from train.dataset import split_and_save_functions
    TRAINING_AVAILABLE = True
except ImportError as e:
    logger.error(f"Training dependencies not available: {e}")
    logger.error("Install with: pip install -r requirements_training.txt")
    TRAINING_AVAILABLE = False
    TRAINING_IMPORT_ERROR = str(e)

    # For data preparation, we need the splitting function even without full training
    def split_and_save_functions(*args, **kwargs):
        raise ImportError("Training dependencies required for data preparation")

    def GraphCodeBERTTrainer(*args, **kwargs):
        raise ImportError("Training dependencies required")

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
        default_log = log_dir / f"pipeline_{timestamp}.log"
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

def load_repositories_from_file(filename: str) -> List[RepositoryInfo]:
    """Load repositories from JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)

    # Handle different file formats
    if isinstance(data, list):
        # Direct list of repositories
        return [RepositoryInfo(**repo) for repo in data]
    elif isinstance(data, dict) and 'repositories' in data:
        # Wrapped format with metadata
        return [RepositoryInfo(**repo) for repo in data['repositories']]
    else:
        logger.warning(f"Unexpected repositories file format in {filename}")
        return []

def load_clone_results_from_file(filename: str) -> List[CloneResult]:
    """Load clone results from JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)

    # Handle different file formats
    if isinstance(data, list):
        # Direct list of clone results
        return [CloneResult.from_dict(result) for result in data]
    elif isinstance(data, dict) and 'clone_results' in data:
        # Wrapped format with metadata
        return [CloneResult.from_dict(result) for result in data['clone_results']]
    else:
        logger.warning(f"Unexpected clone results file format in {filename}")
        return []

def _add_discovery_args(parser):
    """Add repository discovery arguments."""
    discovery = parser.add_argument_group('Repository Discovery')
    discovery.add_argument('--github-token', help='GitHub API token')
    discovery.add_argument('--max-repos', type=int, default=DISCOVERY_CONFIG['max_total_repos'])
    discovery.add_argument('--min-stars', type=int, default=DISCOVERY_CONFIG['min_stars'])

def _add_clone_args(parser):
    """Add cloning arguments."""
    clone = parser.add_argument_group('Repository Cloning')
    clone.add_argument('--max-concurrent-clones', type=int, default=CLONE_CONFIG['max_concurrent_clones'])
    clone.add_argument('--clone-timeout', type=int, default=CLONE_CONFIG['clone_timeout'])

def _add_extract_args(parser):
    """Add extraction arguments."""
    extract = parser.add_argument_group('Function Extraction')
    extract.add_argument('--max-parser-workers', type=int, default=PARSER_CONFIG['max_concurrent_parsers'])
    extract.add_argument('--min-function-score', type=float, default=0.5)  # From FUNCTION_SCORING if available
    extract.add_argument('--max-code-length', type=int, help='Override max code length')
    extract.add_argument('--max-dfg-length', type=int, help='Override max DFG length')

def _add_prepare_args(parser):
    """Add data preparation arguments."""
    prepare = parser.add_argument_group('Data Preparation')
    prepare.add_argument('--functions-file', help='Use specific functions JSONL file')
    prepare.add_argument('--no-split', action='store_true', help='Skip train/val/test split')

def _add_training_args(parser):
    """Add training arguments."""
    training = parser.add_argument_group('Training')
    training.add_argument('--use-lora', action='store_true', help='Use LoRA adaptation')

def validate_args(args) -> bool:
    """Validate command line arguments."""
    # GitHub token validation for discovery
    if hasattr(args, 'github_token') and args.use_repos is None:
        if not args.github_token and not os.getenv('GITHUB_TOKEN'):
            logger.warning("No GitHub token provided. Discovery may be rate-limited.")

    # Training validation - both training and prepare need the training stack
    if args.command in ['train', 'prepare'] and not getattr(args, "no_split", False):
        if not TRAINING_AVAILABLE:
            logger.error("Training dependencies not available for data preparation:")
            logger.error(f"  {TRAINING_IMPORT_ERROR}")
            logger.error("Install with: pip install -r requirements_training.txt")
            return False

    return True

def discover_repositories(args) -> List[RepositoryInfo]:
    """Discover repositories to process."""
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
                    if scraper._meets_quality_criteria(repo_info):
                        repositories.append(repo_info)
                        logger.info(f"✓ Added repository: {repo_name}")
                    else:
                        logger.warning(f"✗ Repository found but does not meet quality criteria: {repo_name}")
                else:
                    logger.warning(f"✗ Repository not found: {repo_name}")
            except Exception as e:
                logger.error(f"✗ Failed to get info for {repo_name}: {e}")
    else:
        # Discover repositories
        repositories = scraper.discover_repositories(args.max_repos)

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

def prepare_training_data(args) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Prepare training data by splitting functions into train/val/test sets."""
    logger.info("=" * 60)
    logger.info("STEP 4: PREPARING TRAINING DATA")
    logger.info("=" * 60)

    # Determine functions file
    functions_file = getattr(args, "functions_file", None) or get_output_path(OUTPUT_CONFIG['functions_file'])

    if not os.path.exists(functions_file):
        logger.error(f"Functions file not found: {functions_file}")
        return None, None, None

    # Determine output directory
    output_dir = args.graphcodebert_output or get_graphcodebert_output_path("")

    if getattr(args, "no_split", False):
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
        trainer = GraphCodeBERTTrainer(
            use_lora=args.use_lora,
            output_dir=args.model_output_dir
        )

        trainer.train(
            train_file=args.train_file,
            val_file=args.val_file,
        )

        logger.info("✓ Training completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False

def run_full_pipeline(args) -> int:
    """Run the complete pipeline."""
    logger.info("Starting full GraphCodeBERT pipeline")

    # Set GitHub token
    if args.github_token:
        os.environ['GITHUB_TOKEN'] = args.github_token

    # Update GraphCodeBERT config with command line overrides
    if hasattr(args, 'max_code_length') and args.max_code_length:
        GRAPHCODEBERT_CONFIG['max_code_length'] = args.max_code_length
    if hasattr(args, 'max_dfg_length') and args.max_dfg_length:
        GRAPHCODEBERT_CONFIG['max_dfg_length'] = args.max_dfg_length

    try:
        # Step 1: Discover repositories (with smart checkpointing)
        logger.info("=" * 60)
        logger.info("STEP 1: DISCOVERING REPOSITORIES")
        logger.info("=" * 60)

        repositories_file = get_output_path(OUTPUT_CONFIG['repos_file'])
        if os.path.exists(repositories_file) and not args.use_repos and not getattr(args, 'force_refresh', False):
            logger.info(f"Loading existing repositories from {repositories_file}")
            repositories = load_repositories_from_file(repositories_file)
        else:
            if getattr(args, 'force_refresh', False):
                logger.info("Force refresh enabled - rediscovering repositories")
            repositories = discover_repositories(args)

        if not repositories:
            logger.error("No repositories available")
            return 1

        # Step 2: Clone repositories (with smart checkpointing)
        logger.info("=" * 60)
        logger.info("STEP 2: CLONING REPOSITORIES")
        logger.info("=" * 60)

        clone_results_file = get_output_path(OUTPUT_CONFIG['clone_results_file'])
        if os.path.exists(clone_results_file) and not getattr(args, 'force_refresh', False):
            logger.info(f"Loading existing clone results from {clone_results_file}")
            clone_results = load_clone_results_from_file(clone_results_file)
        else:
            if getattr(args, 'force_refresh', False):
                logger.info("Force refresh enabled - re-cloning repositories")
            clone_results = clone_repositories(repositories, args)

        if not clone_results:
            logger.error("No repositories cloned successfully")
            return 1

        # Step 3: Extract functions (with smart checkpointing)
        logger.info("=" * 60)
        logger.info("STEP 3: EXTRACTING FUNCTIONS")
        logger.info("=" * 60)

        functions_file = get_output_path(OUTPUT_CONFIG['functions_file'])
        if os.path.exists(functions_file) and not getattr(args, 'force_refresh', False):
            logger.info(f"Loading existing functions from {functions_file}")
        else:
            if getattr(args, 'force_refresh', False):
                logger.info("Force refresh enabled - re-extracting functions")
            functions = extract_functions(clone_results, args)
            if not functions:
                logger.error("No functions extracted")
                return 1

        # Step 4: Prepare training data
        logger.info("=" * 60)
        logger.info("STEP 4: PREPARING TRAINING DATA")
        logger.info("=" * 60)

        train_file, val_file, test_file = prepare_training_data(args)
        if not train_file:
            logger.error("Failed to prepare training data")
            return 1

        # Step 5: Training (if available)
        if TRAINING_AVAILABLE:
            logger.info("=" * 60)
            logger.info("STEP 5: TRAINING MODEL")
            logger.info("=" * 60)

            # Set up args for training
            args.train_file = train_file
            args.val_file = val_file

            success = train_model(args)
            if success:
                logger.info("✓ Full pipeline completed successfully!")
                return 0
            else:
                logger.error("✗ Training failed")
                return 1
        else:
            logger.info("Training dependencies not available")
            logger.info("Install with: pip install -r requirements_training.txt")
            logger.info(f"Then run: python main.py train --train-file {train_file} --val-file {val_file}")
            return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


def run_data_preparation(args) -> int:
    """Run data preparation only."""
    logger.info("Starting data preparation")

    # Set GitHub token
    if hasattr(args, 'github_token') and args.github_token:
        os.environ['GITHUB_TOKEN'] = args.github_token

    try:
        # Check if we can start from existing functions (unless force refresh)
        functions_file = args.functions_file or get_output_path(OUTPUT_CONFIG['functions_file'])

        if os.path.exists(functions_file) and not getattr(args, 'force_refresh', False):
            logger.info(f"✓ Using existing functions from {functions_file}")
            # Skip to data preparation
            train_file, val_file, test_file = prepare_training_data(args)
            if train_file:
                logger.info("✓ Data preparation completed")
                return 0
            else:
                logger.error("✗ Data preparation failed")
                return 1

        # Need to run discovery/clone/extract pipeline
        if getattr(args, 'force_refresh', False):
            logger.info("Force refresh enabled - running full preparation pipeline")
        else:
            logger.info("Running preparation pipeline...")

        # Step 1: Smart checkpointing for repository discovery
        repositories_file = get_output_path(OUTPUT_CONFIG['repos_file'])
        if os.path.exists(repositories_file) and not args.use_repos and not getattr(args, 'force_refresh', False):
            logger.info(f"Loading existing repositories from {repositories_file}")
            repositories = load_repositories_from_file(repositories_file)
        else:
            repositories = discover_repositories(args)

        if not repositories:
            logger.error("No repositories available")
            return 1

        # Step 2: Smart checkpointing for cloning
        clone_results_file = get_output_path(OUTPUT_CONFIG['clone_results_file'])
        if os.path.exists(clone_results_file) and not getattr(args, 'force_refresh', False):
            logger.info(f"Loading existing clone results from {clone_results_file}")
            clone_results = load_clone_results_from_file(clone_results_file)
        else:
            clone_results = clone_repositories(repositories, args)

        if not clone_results:
            logger.error("No repositories cloned")
            return 1

        # Step 3: Extract functions if needed or force refresh
        if not os.path.exists(functions_file) or getattr(args, 'force_refresh', False):
            functions = extract_functions(clone_results, args)
            if not functions:
                logger.error("No functions extracted")
                return 1

        # Step 4: Prepare training data
        train_file, val_file, test_file = prepare_training_data(args)
        if train_file:
            logger.info("✓ Data preparation completed")
            logger.info(f"Next: python main.py train --train-file {train_file} --val-file {val_file}")
            return 0
        else:
            logger.error("✗ Data preparation failed")
            return 1

    except Exception as e:
        logger.error(f"Data preparation failed: {e}", exc_info=True)
        return 1


def run_training_only(args) -> int:
    """Train model on existing prepared data."""
    logger.info("Starting training on existing data")

    if not TRAINING_AVAILABLE:
        logger.error("Training dependencies not available")
        logger.error("Install with: pip install -r requirements_training.txt")
        return 1

    # Check if we need to prepare data first (if force refresh or missing files)
    if getattr(args, 'force_refresh', False) or not hasattr(args, 'train_file') or not args.train_file:
        logger.info("No training file specified or force refresh enabled - preparing data first")

        # Set default values for data preparation
        if not hasattr(args, 'train_file') or not args.train_file:
            args.graphcodebert_output = args.graphcodebert_output or get_output_path('graphcodebert_data')

        # Run data preparation pipeline
        prep_result = run_data_preparation(args)
        if prep_result != 0:
            logger.error("Data preparation failed - cannot proceed with training")
            return prep_result

        # Set train/val files from preparation output
        graphcodebert_dir = args.graphcodebert_output or get_output_path('graphcodebert_data')
        args.train_file = os.path.join(graphcodebert_dir, 'train.jsonl')
        args.val_file = os.path.join(graphcodebert_dir, 'valid.jsonl')

    # Validate training files exist
    if not os.path.exists(args.train_file):
        logger.error(f"Training file not found: {args.train_file}")
        return 1

    if args.val_file and not os.path.exists(args.val_file):
        logger.error(f"Validation file not found: {args.val_file}")
        return 1

    success = train_model(args)
    return 0 if success else 1


def run_evaluation(args) -> int:
    """Run evaluation on existing checkpoint."""
    logger.info("Starting model evaluation")

    if not TRAINING_AVAILABLE:
        logger.error("Evaluation dependencies not available")
        logger.error("Install with: pip install -r requirements_training.txt")
        return 1

    # Check if model checkpoint exists
    if not os.path.exists(args.model_checkpoint):
        logger.error(f"Model checkpoint not found: {args.model_checkpoint}")
        return 1

    # Check if we need to prepare evaluation data (if force refresh or missing)
    eval_data_dir = args.graphcodebert_output or get_output_path('graphcodebert_data')
    test_file = os.path.join(eval_data_dir, 'full.jsonl')

    if getattr(args, 'force_refresh', False) or not os.path.exists(test_file):
        logger.info("Force refresh enabled or evaluation data missing - preparing data first")

        # Run data preparation to get evaluation data
        prep_result = run_data_preparation(args)
        if prep_result != 0:
            logger.error("Data preparation failed - cannot proceed with evaluation")
            return prep_result

    # Validate evaluation data exists
    if not os.path.exists(test_file):
        logger.error(f"Test file not found: {test_file}")
        logger.error("Run data preparation first: python main.py prepare")
        return 1

    try:
        # Import evaluation functions
        from benchmark import get_evaluator

        logger.info(f"Loading model from checkpoint: {args.model_checkpoint}")
        logger.info(f"Evaluation tasks: {args.eval_tasks}")
        logger.info(f"Test data: {test_file}")

        # Run evaluation for each task
        all_results = {}
        for task in args.eval_tasks:
            logger.info(f"Running {task} evaluation...")

            # Get the appropriate evaluator
            evaluator_class = get_evaluator(task)
            evaluator = evaluator_class(
                model_checkpoint=args.model_checkpoint,
                device="auto"
            )

            # Run evaluation
            results = evaluator.evaluate(
                data_path=test_file,
                batch_size=getattr(args, 'eval_batch_size', 32),
                max_examples=getattr(args, 'max_eval_examples', None)
            )

            # Store results
            for metric, score in results.items():
                all_results[f"{task}_{metric}"] = score

            # Cleanup evaluator resources
            evaluator.cleanup()

        if all_results:
            logger.info("✓ Evaluation completed successfully!")
            # Print results summary
            for metric, score in all_results.items():
                logger.info(f"  {metric}: {score:.4f}")
            return 0
        else:
            logger.error("✗ Evaluation failed - no results")
            return 1

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1

# Also need to update the argument parser to include force_refresh
def create_argument_parser():
    """Create command line argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description='Erlang GraphCodeBERT Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (default behavior)
  python main.py full --github-token YOUR_TOKEN
  python main.py --github-token YOUR_TOKEN  # same as above

  # Force refresh - re-run all stages
  python main.py full --force-refresh --github-token YOUR_TOKEN

  # Prepare validation data from different repo
  python main.py prepare --use-repos elixir-lang/elixir \\
    --graphcodebert-output output/validation_data

  # Train on existing data
  python main.py train --train-file output/graphcodebert_data/train.jsonl \\
    --val-file output/graphcodebert_data/valid.jsonl

  # Train with force refresh (re-prepare data)
  python main.py train --force-refresh

  # Evaluate checkpoint
  python main.py eval --model-checkpoint models/erlang_graphcodebert/checkpoint-best \\
    --graphcodebert-output output/validation_data
        """
    )

    # Create subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Shared arguments for all commands
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument('--debug', action='store_true', help='Enable debug logging')
    shared.add_argument('--quiet', action='store_true', help='Suppress non-error output')
    shared.add_argument('--log-file', help='Log file path')
    shared.add_argument('--use-repos', nargs='+', help='Specific repositories to use (owner/repo)')
    shared.add_argument('--graphcodebert-output', help='GraphCodeBERT output directory')
    shared.add_argument('--model-output-dir', default='models/erlang_graphcodebert')
    shared.add_argument('--force-refresh', action='store_true',
                       help='Force re-running of all pipeline stages, ignoring existing cache files')

    # Full pipeline command (default behavior)
    full_cmd = subparsers.add_parser('full', parents=[shared],
                                    help='Run complete pipeline')
    _add_discovery_args(full_cmd)
    _add_clone_args(full_cmd)
    _add_extract_args(full_cmd)
    _add_prepare_args(full_cmd)
    _add_training_args(full_cmd)

    # Data preparation only
    prep_cmd = subparsers.add_parser('prepare', parents=[shared],
                                    help='Prepare training data only')
    _add_discovery_args(prep_cmd)
    _add_clone_args(prep_cmd)
    _add_extract_args(prep_cmd)
    _add_prepare_args(prep_cmd)

    # Training only
    train_cmd = subparsers.add_parser('train', parents=[shared],
                                     help='Train model only')
    train_cmd.add_argument('--train-file', help='Training JSONL file (optional if using force-refresh)')
    train_cmd.add_argument('--val-file', help='Validation JSONL file')
    _add_training_args(train_cmd)

    # Evaluation only
    eval_cmd = subparsers.add_parser('eval', parents=[shared],
                                    help='Evaluate model')
    eval_cmd.add_argument('--model-checkpoint', required=True,
                         help='Path to model checkpoint')
    eval_cmd.add_argument('--eval-tasks', nargs='+', default=['mlm'],
                         choices=['mlm', 'code_search', 'clone_detection'],
                         help='Evaluation tasks to run')

    return parser

def main() -> int:
    """Main entry point with subcommand routing."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Backward compatibility - default to full pipeline
    if args.command is None:
        args.command = 'full'

    # Setup logging
    setup_logging(debug=args.debug, log_file=args.log_file)
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    logger.info(f"Starting Erlang GraphCodeBERT pipeline - command: {args.command}")
    logger.info(f"Arguments: {vars(args)}")

    # Validate arguments
    if not validate_args(args):
        return 1

    # Route to appropriate handler
    try:
        if args.command == 'full':
            return run_full_pipeline(args)
        elif args.command == 'prepare':
            return run_data_preparation(args)
        elif args.command == 'train':
            return run_training_only(args)
        elif args.command == 'eval':
            return run_evaluation(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
