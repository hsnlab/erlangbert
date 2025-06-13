"""
Configuration file for Erlang corpus scraper.
Contains all constants, settings, and target repositories.
"""

import os
from typing import Dict, List, Any

# GitHub API Configuration
GITHUB_API_BASE = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # Set via environment variable
REQUESTS_PER_HOUR = 5000 if GITHUB_TOKEN else 60  # Authenticated vs anonymous

# Repository Discovery Settings
REPO_DISCOVERY = {
    "min_stars": 10,
    "min_size_kb": 100,
    "max_size_mb": 500,
    "min_erlang_percentage": 0.5,  # At least 50% Erlang code
    "exclude_forks": True,
    "recent_activity_months": 12,  # Active in last 12 months
    "max_repos_per_search": 1000,
    "include_archived": False,
}

# High-quality seed repositories
SEED_REPOSITORIES = [
    # Core Erlang/OTP
    "erlang/otp",
    
    # Web frameworks and servers
    "ninenines/cowboy",
    "phoenixframework/phoenix", 
    "mochi/mochiweb",
    "extend/cowlib",
    "ninenines/gun",
    
    # Message brokers and distributed systems
    "rabbitmq/rabbitmq-server",
    "processone/ejabberd",
    "emqx/emqx",
    "vernemq/vernemq",
    
    # Databases and storage
    "apache/couchdb",
    "basho/riak",
    "basho/riak_core",
    "leo-project/leofs",
    "devinus/poolboy",
    
    # Build tools and utilities
    "erlang/rebar3",
    "rebar/rebar",
    "massemanet/eper",
    "ferd/recon",
    
    # Testing and development
    "manopapad/proper",
    "eproxus/meck",
    "boundary/folsom",
    
    # Libraries and frameworks
    "devinus/hackney",
    "benoitc/hackney",
    "kivra/oauth2",
    "jlouis/graphql-erlang",
    "FlowForwarding/of_protocol",
    
    # Game engines and multimedia
    "jlouis/etorrent",
    "spawngrid/mimetypes",
]

# GitHub search queries for additional repository discovery
GITHUB_SEARCH_QUERIES = [
    "language:erlang stars:>50 size:>1000",
    "language:erlang stars:>20 created:>2020-01-01",
    "language:erlang topic:web stars:>10",
    "language:erlang topic:distributed stars:>10",
    "language:erlang topic:messaging stars:>10",
    "language:erlang topic:database stars:>10",
    "language:erlang topic:framework stars:>10",
]

# File Processing Settings
FILE_PROCESSING = {
    "target_extensions": [".erl", ".hrl", ".escript"],
    "include_test_files": True,
    "include_examples": True,
    "exclude_patterns": [
        "*/deps/*",           # Dependencies
        "*/rebar.lock",       # Lock files
        "*/_build/*",         # Build artifacts
        "*/ebin/*",           # Compiled beam files
        "*/priv/*",           # Private files (usually non-code)
        "*/.git/*",           # Git metadata
        "*/test/ct_*",        # Common test generated files
    ],
    "min_file_size_bytes": 50,
    "max_file_size_bytes": 1024 * 1024,  # 1MB max
}

# Function Extraction Settings
FUNCTION_EXTRACTION = {
    "min_function_lines": 2,
    "max_function_lines": 200,
    "include_private_functions": True,
    "include_exported_only": False,
    "require_documentation": False,
    "extract_type_specs": True,
    "extract_edoc_comments": True,
    "group_function_clauses": True,  # Group multi-clause functions
}

# Documentation Patterns
DOC_PATTERNS = {
    "edoc_patterns": [
        r"%% @doc\s+(.+?)(?=\n%%|\n[^%]|\n$)",
        r"%% @brief\s+(.+?)(?=\n%%|\n[^%]|\n$)",
    ],
    "comment_patterns": [
        r"%%\s+(.+?)(?=\n[^%]|\n$)",  # Standard comments
        r"%\s+(.+?)(?=\n[^%]|\n$)",   # Single % comments
    ],
    "spec_patterns": [
        r"-spec\s+(\w+)\s*\([^)]*\)\s*->\s*[^.]+\.",
    ],
}

# Output Configuration
OUTPUT_CONFIG = {
    "base_directory": "./output",
    "repositories_file": "repositories.json",
    "functions_file": "functions.jsonl", 
    "stats_file": "stats.json",
    "clone_directory": "./cloned_repos",
    "checkpoint_file": "scraper_checkpoint.json",
    "log_file": "scraper.log",
}

# Processing Limits
PROCESSING_LIMITS = {
    "max_repositories": 200,      # Limit total repos to process
    "max_functions_per_repo": 2000,  # Limit functions per repo
    "max_total_functions": 150000,   # Target corpus size
    "parallel_clone_workers": 4,     # Concurrent git clones
    "parallel_parse_workers": 8,     # Concurrent file parsers
    "request_delay_seconds": 0.1,    # Delay between API requests
}

# Retry and Error Handling
ERROR_HANDLING = {
    "max_retries": 3,
    "retry_delay_seconds": 2,
    "skip_on_clone_failure": True,
    "skip_on_parse_failure": True,
    "continue_on_api_error": True,
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "console_output": True,
    "file_output": True,
}

def get_output_path(filename: str) -> str:
    """Get full path for output file."""
    base_dir = OUTPUT_CONFIG["base_directory"]
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, filename)

def get_clone_path(repo_name: str) -> str:
    """Get full path for cloned repository."""
    clone_dir = OUTPUT_CONFIG["clone_directory"]
    os.makedirs(clone_dir, exist_ok=True)
    # Replace / with _ for filesystem compatibility
    safe_name = repo_name.replace("/", "_")
    return os.path.join(clone_dir, safe_name)

def validate_config() -> bool:
    """Validate configuration settings."""
    if not GITHUB_TOKEN:
        print("Warning: GITHUB_TOKEN not set. API rate limits will be severely restricted.")
        
    if PROCESSING_LIMITS["max_repositories"] > 1000:
        print("Warning: Processing over 1000 repositories may take very long.")
        
    return True

if __name__ == "__main__":
    # Test configuration
    validate_config()
    print(f"Configuration loaded successfully!")
    print(f"Seed repositories: {len(SEED_REPOSITORIES)}")
    print(f"Search queries: {len(GITHUB_SEARCH_QUERIES)}")
    print(f"Output directory: {OUTPUT_CONFIG['base_directory']}")
