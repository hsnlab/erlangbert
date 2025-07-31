#!/usr/bin/env python3
"""
Setup script for Erlang corpus scraper.
Creates necessary directories and validates dependencies.
Updated for new project structure and GraphCodeBERT pipeline.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_git_installed():
    """Check if git is installed and accessible."""
    try:
        result = subprocess.run(["git", "--version"],
                                capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Git is installed: {result.stdout.strip()}")
            return True
        else:
            print("✗ Git is not working properly")
            return False
    except FileNotFoundError:
        print("✗ Git is not installed or not in PATH")
        return False

def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    version_info = f"{version.major}.{version.minor}.{version.micro}"
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python version is compatible: {version_info}")
        return True
    else:
        print(f"✗ Python version {version_info} is not supported")
        print("  Requires Python 3.8 or higher")
        return False

def create_directories():
    """Create necessary project directories."""
    directories = [
        "output",
        "output/graphcodebert_data",  # New: GraphCodeBERT output directory
        "cloned_repos",
        "scrapers",
        "parsers",
        "utils",
        "logs",
        "models"  # New: For trained models
    ]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True, parents=True)
        print(f"✓ Created directory: {directory}")

    # Create __init__.py files for Python packages
    init_files = [
        "scrapers/__init__.py",
        "parsers/__init__.py",
        "utils/__init__.py"
    ]

    for init_file in init_files:
        Path(init_file).touch(exist_ok=True)
        print(f"✓ Created: {init_file}")

def install_dependencies():
    """Install Python dependencies."""
    print("Installing core dependencies...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )
        print("✓ Core dependencies installed successfully")

        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install core dependencies")
        return False

def test_imports():
    """Test if core modules can be imported."""
    print("Testing core module imports...")

    # Test core imports
    test_imports = [
        ("config", "Configuration module"),
        ("scrapers.github_scraper", "GitHub scraper"),
        ("scrapers.repo_cloner", "Repository cloner"),
        ("parsers.function_extractor", "Function extractor")
    ]

    success = True
    for module_name, description in test_imports:
        try:
            __import__(module_name)
            print(f"✓ {description} import successful")
        except ImportError as e:
            print(f"✗ {description} import failed: {e}")
            success = False

    return success

def test_erlang_parser():
    """Test if the Erlang parser can be initialized."""
    print("Testing Erlang parser setup...")
    try:
        # Import and test the parser
        sys.path.append(os.getcwd())
        from parsers.erlang_parser import ErlangParser

        # This will automatically download and compile tree-sitter-erlang
        parser = ErlangParser()
        print("✓ Erlang parser initialized successfully")

        # Quick test parse
        test_code = "hello() -> world."
        root = parser.parse_string(test_code)
        if root:
            print("✓ Erlang parser test passed")
            return True
        else:
            print("✗ Erlang parser test failed")
            return False

    except Exception as e:
        print(f"✗ Erlang parser setup failed: {e}")
        print("  Note: This will be automatically set up when first used")
        return True  # Non-blocking for setup

def test_optional_dependencies():
    """Test optional ML dependencies."""
    print("Testing optional ML dependencies...")

    try:
        import torch
        print(f"✓ PyTorch available: {torch.__version__}")
    except ImportError:
        print("  PyTorch not installed (needed for GraphCodeBERT training)")

    try:
        import transformers
        print(f"✓ Transformers available: {transformers.__version__}")
    except ImportError:
        print("  Transformers not installed (needed for GraphCodeBERT)")

    try:
        import peft
        print(f"✓ PEFT available: {peft.__version__}")
    except ImportError:
        print("  PEFT not installed (needed for LoRA fine-tuning)")

def create_env_template():
    """Create environment variable template."""
    env_template = """# Erlang Corpus Scraper Environment Variables
# Copy this file to .env and fill in your values

# GitHub API token (OPTIONAL but strongly recommended for higher rate limits)
# Without token: 60 requests/hour | With token: 5000 requests/hour
# Get one at: https://github.com/settings/tokens
# GITHUB_TOKEN=your_github_token_here

# Repository Discovery Settings
# MAX_REPOSITORIES=200
# MIN_STARS=5
# MAX_REPOS_PER_QUERY=100

# Cloning Settings
# MAX_CONCURRENT_CLONES=4
# CLONE_TIMEOUT=300

# Function Extraction Settings
# MAX_PARSER_WORKERS=8
# MIN_FUNCTION_SCORE=5.0
# MAX_FILE_SIZE=1048576

# GraphCodeBERT Settings
# MAX_CODE_LENGTH=256
# MAX_DFG_LENGTH=64
# MAX_NL_LENGTH=128

# Training Settings (if using training features)
# BATCH_SIZE=32
# LEARNING_RATE=2e-5
# NUM_EPOCHS=10
# USE_LORA=true

# Output Settings
# OUTPUT_DIRECTORY=./output
# GRAPHCODEBERT_OUTPUT=./output/graphcodebert_data
"""

    with open(".env.template", "w") as f:
        f.write(env_template)

    print("✓ Created .env.template")
    print("  Edit .env.template and save as .env to configure your environment")

def main():
    """Main setup function."""
    print("=" * 70)
    print("ERLANG CORPUS SCRAPER & GRAPHCODEBERT PIPELINE SETUP")
    print("=" * 70)

    success = True

    # Check prerequisites
    print("Checking prerequisites...")
    if not check_python_version():
        success = False

    if not check_git_installed():
        success = False

    if not success:
        print("\n✗ Prerequisites not met. Please install required software.")
        return 1

    # Create project structure
    print("\nCreating project structure...")
    create_directories()

    # Install dependencies
    print("\nInstalling dependencies...")
    if not install_dependencies():
        success = False

    # Test core imports
    print("\nTesting core modules...")
    if not test_imports():
        success = False

    # Test Erlang parser
    print("\nTesting Erlang parser...")
    test_erlang_parser()  # Non-blocking

    # Test optional dependencies
    print("\nChecking optional dependencies...")
    test_optional_dependencies()

    # Create configuration files
    print("\nCreating configuration files...")
    create_env_template()

    # Final status
    print("\n" + "=" * 70)
    if success:
        print("✓ SETUP COMPLETED SUCCESSFULLY")
        print("\nNext steps:")
        print("1. [OPTIONAL] Set up GitHub token in .env file for higher rate limits")
        print("   • Without token: 60 requests/hour")
        print("   • With token: 5000 requests/hour")
        print("2. Quick test (no token needed): python main.py --use-repos ninenines/cowboy --clone --extract")
        print("3. Full pipeline: python main.py  # (or add --github-token YOUR_TOKEN)")
        print("4. GraphCodeBERT: python main.py --transform-only")
        print("5. Training: python train_graphcodebert.py --help")
        print("\nSee QUICK_START.md for detailed instructions")
    else:
        print("✗ SETUP COMPLETED WITH ERRORS")
        print("Please fix the issues above before running the scraper.")

    print("=" * 70)
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
