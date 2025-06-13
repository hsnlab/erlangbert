#!/usr/bin/env python3
"""
Setup script for Erlang corpus scraper.
Creates necessary directories and validates dependencies.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_git_installed():
    """Check if git is installed and accessible."""
    try:
        result = subprocess.run(["git", "--version"], capture_output=True, text=True)
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
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python version is compatible: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python version {version.major}.{version.minor}.{version.micro} is not supported")
        print("  Requires Python 3.8 or higher")
        return False

def create_directories():
    """Create necessary project directories."""
    directories = [
        "output",
        "cloned_repos", 
        "scrapers",
        "parsers",
        "utils",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
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
    print("Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install dependencies")
        return False

def setup_tree_sitter_erlang():
    """Download and compile tree-sitter-erlang."""
    print("Setting up tree-sitter-erlang...")
    
    # Check if already exists
    if os.path.exists("tree-sitter-erlang"):
        print("✓ tree-sitter-erlang directory already exists")
        return True
    
    try:
        # Clone WhatsApp's tree-sitter-erlang
        subprocess.check_call([
            "git", "clone", "--depth", "1",
            "https://github.com/WhatsApp/tree-sitter-erlang.git"
        ])
        print("✓ Cloned tree-sitter-erlang repository")
        
        # TODO: Add compilation steps here when we implement the parser
        print("  Note: Parser compilation will be added in the next phase")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to clone tree-sitter-erlang: {e}")
        return False

def create_env_template():
    """Create environment variable template."""
    env_template = """# Erlang Corpus Scraper Environment Variables
# Copy this file to .env and fill in your values

# GitHub API token (strongly recommended for higher rate limits)
# Get one at: https://github.com/settings/tokens
# GITHUB_TOKEN=your_github_token_here

# Optional: Adjust processing limits
# MAX_REPOSITORIES=200
# PARALLEL_CLONE_WORKERS=4

# Optional: Custom output directory
# OUTPUT_DIRECTORY=./output
"""
    
    with open(".env.template", "w") as f:
        f.write(env_template)
    
    print("✓ Created .env.template")
    print("  Edit .env.template and save as .env to configure your environment")

def main():
    """Main setup function."""
    print("=" * 60)
    print("ERLANG CORPUS SCRAPER SETUP")
    print("=" * 60)
    
    success = True
    
    # Check prerequisites
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
    
    # Set up tree-sitter-erlang
    print("\nSetting up tree-sitter-erlang...")
    if not setup_tree_sitter_erlang():
        print("  Warning: tree-sitter-erlang setup failed, but continuing...")
    
    # Create environment template
    print("\nCreating configuration template...")
    create_env_template()
    
    # Final status
    print("\n" + "=" * 60)
    if success:
        print("✓ SETUP COMPLETED SUCCESSFULLY")
        print("\nNext steps:")
        print("1. Set up your GitHub token in .env file (recommended)")
        print("2. Run: python main.py --discover --clone")
        print("3. Or test with: python main.py --discover-only --max-repos 5")
    else:
        print("✗ SETUP COMPLETED WITH ERRORS")
        print("Please fix the issues above before running the scraper.")
    
    print("=" * 60)
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
