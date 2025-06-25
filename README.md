# ErlangBERT: Specialized Embedding Model for Erlang

An end-to-end pipeline for creating specialized code embeddings for the Erlang programming
language, based on the GraphCodeBERT family of models. This project automatically discovers,
clones, and processes high-quality Erlang repositories to create a large-scale corpus for training
domain-specific code embeddings.

## Project Overview

ErlangBERT aims to create specialized embeddings for Erlang code by:

1. **Automated Repository Discovery**: Using GitHub API to find high-quality Erlang repositories
2. **Intelligent Code Extraction**: Tree-sitter based parsing with multi-clause function handling  
3. **GraphCodeBERT Integration**: Extracting tokens, variables, and dataflow graphs for training
4. **Quality Scoring**: Sophisticated scoring system for function quality assessment
5. **Training Pipeline**: Support for both direct fine-tuning and LoRA adaptation

[GraphCodeBERT](<https://arxiv.org/abs/2009.08366>) is a pre-trained model that considers code
structure through data flow graphs.

## Status

- [x] **Phase 1:** Corpus Creation: Build a large-scale Erlang dataset from GitHub by collecting 100K+
      high-quality Erlang functions with matching documentation
- [x] **Phase 2:** Parsing & Data Flow Extraction: Convert Erlang source code to GraphCodeBERT format
  - [x] File Scanning: Find .erl files in cloned repositories
  - [x] Tree-sitter Parsing: Use [WhatsApp's tree-sitter-erlang](<https://github.com/WhatsApp/tree-sitter-erlang>)
  - [x] Function Extraction: Group multi-clause Erlang functions
  - [x] Data Flow Analysis: Extract variable dependencies for Erlang patterns
  - [x] JSONL Generation: Create training data in GraphCodeBERT format
- [ ] **Phase 3:** Model Fine-tuning: Adapt GraphCodeBERT for Erlang
  - [ ] Direct Fine-tuning: Full model fine-tuning on Erlang corpus
  - [ ] LoRA Adaptation: Low-rank adaptation for efficient fine-tuning
- [ ] **Phase 4:** Evaluation: Validate Erlang specialization
  - [ ] Code Search: Natural language → Erlang code retrieval
  - [ ] Code Similarity: Detect functionally similar Erlang code
  - [ ] Pattern Recognition: Understand Erlang-specific constructs

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Git
- GitHub API token (recommended for higher rate limits)

### Installation

``` console
git clone <repository>
cd erlang_corpus_scraper
python -m venv venv
source venv/bin/activate
python setup.py
export GITHUB_TOKEN=<your_github_token>
```

### Basic Usage

**Discover and process repositories:**
```bash
# Discover repositories and extract functions
python main.py --discover --clone --extract

# Process a limited number for testing
python main.py --discover-only --max-repos 5

# Extract from already cloned repositories
python main.py --extract-only
```

**Test the parser:**
```bash
# Test the Erlang parser directly
python parsers/erlang_parser.py

# Test function extraction
python parsers/function_extractor.py
```

## Parsing special Erlang code constructs

Some of Erlang's language constructs make for more complex GraphCodeBERT-style datafow graphs: 

- Pattern Matching: Variables flow into pattern destructuring, creating multiple new variables.
- Guards: Variables flow into boolean conditions that control execution paths.
- Message Passing: Variables flow between separate processes, creating inter-process dependencies.

These create more complex data flow graphs than imperative languages because:

- One input can create multiple outputs (pattern destructuring).
- Execution path depends on data values (guards).
- Variables can flow between different execution contexts (processes).

## Pattern matching

Pattern matching is Erlang's way of destructuring data and controlling program flow
simultaneously. Instead of if statements, you use different function clauses with different
patterns.

``` erlang
%% Multiple clauses for the same function
max(A, B) when A > B -> A;
max(A, B) -> B.
```

GraphCodeBERT representation in the corpus:

```
[CLS] "Returns maximum value" [SEP] max(A,B) when A>B -> A; max(A,B) -> B. [SEP] A B A B [SEP]
```

The ErlangBERT training pipeline treats all clauses of a function as one logical unit in the
corpus.

## Guard Flows

Guards are additional conditions that can be checked after pattern matching succeeds. They're like if conditions but more restricted.

``` erlang
divide(A, B) when B =/= 0 -> A / B;
divide(_, 0) -> error.
```

Data Flow Graph:

```
Variables: A, B, A_clause1, B_clause1, B_guard, A_clause2, result_1, result_2
Edges: [A->A_clause1, B->B_clause1, B_clause1->B_guard, A_clause1->result_1, B_guard->result_1, A->A_clause2, result_2]
```

## Type specs

Add type specs to the code snippet:

```
%% Divides two numbers, returns number or error
%% @spec number(), number() -> number() | error
divide(A, B) when B =/= 0 -> A / B;
```

GraphCodeBERT sees types as natural language context. This fits existing GraphCodeBERT
architecture, but treats types as text, not structured data.

``` json
{
  "docstring": "Divides two numbers.",
  "code": "-spec divide(number(), number()) -> number() | error.;divide(A, B) when B =/= 0 -> A / B;",
  "dfg": {"variables": [...], "edges": [...]}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **GraphCodeBERT**: Based on Microsoft's GraphCodeBERT architecture
- **Tree-sitter**: Using WhatsApp's tree-sitter-erlang grammar
- **Erlang Community**: Thanks to the Erlang ecosystem for high-quality open source projects
- **Claude.ai**: Thanks for Anthropic for contributing to the code.

## 📚 References

- [GraphCodeBERT: Pre-training Code Representations with Data Flow](https://arxiv.org/abs/2009.08366)
- [Tree-sitter Erlang Grammar](https://github.com/WhatsApp/tree-sitter-erlang)
- [CodeSearchNet Dataset](https://github.com/github/CodeSearchNet)


