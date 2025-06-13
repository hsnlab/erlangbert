# ErlangBERT: Pre-trained embedding model for the Erlang programming language

ErlangBERT is a specialized embedding model for the Erlang programming language obtained by
fine-tuning GraphCodeBERT. The goal is to build a model that understands Erlang's unique functional
programming patterns, pattern matching, and message-passing concurrency.

Erlang presents unique challenges for code embedding models:
- **Pattern Matching**: Functions use pattern matching instead of if/else statements
- **Multi-clause Functions**: Single functions have multiple definitions with different patterns
- **Message Passing**: Processes communicate via message passing, not shared memory
- **Functional Programming**: Immutable variables and recursive patterns
- **Concurrency**: Actor model with lightweight processes

[GraphCodeBERT](<https://arxiv.org/abs/2009.08366>) is a pre-trained model that considers code
structure through data flow graphs.

## Status

- [x] **Phase 1:** Corpus Creation: Build a large-scale Erlang dataset from GitHub by collecting 100K+
      high-quality Erlang functions with matching documentation
- [ ] **Phase 2:** Parsing & Data Flow Extraction: Convert Erlang source code to GraphCodeBERT format
  - [ ] File Scanning: Find .erl files in cloned repositories
  - [ ] Tree-sitter Parsing: Use [WhatsApp's tree-sitter-erlang](<https://github.com/WhatsApp/tree-sitter-erlang>)
  - [ ] Function Extraction: Group multi-clause Erlang functions
  - [ ] Data Flow Analysis: Extract variable dependencies for Erlang patterns
  - [ ] JSONL Generation: Create training data in GraphCodeBERT format
- [ ] **Phase 3:** Model Fine-tuning: Adapt GraphCodeBERT for Erlang
  - [ ] Direct Fine-tuning: Full model fine-tuning on Erlang corpus
  - [ ] LoRA Adaptation: Low-rank adaptation for efficient fine-tuning
- [ ] **Phase 4:** Evaluation: Validate Erlang specialization
  - [ ] Code Search: Natural language → Erlang code retrieval
  - [ ] Code Similarity: Detect functionally similar Erlang code
  - [ ] Pattern Recognition: Understand Erlang-specific constructs

## Getting Started

### Prerequisites

- Python 3.8+
- Git
- GitHub token (recommended for higher API limits)

### Setup

``` console
git clone <repository>
cd erlang_corpus_scraper
python setup.py
export GITHUB_TOKEN=<your_github_token>
```

### Scraping

``` console
python main.py --discover-only --max-repos 5       # discovery Only
python main.py --discover --clone                  # full pipeline
python main.py --clone-only                        # clone from existing discovery
python main.py --force-discovery --force-reclone   # force refresh
```

### Parsing: TODO

# Parsing special Erlang code constructs

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

The ErlangBERT training pipeline treats all clauses of a function as one logical unit in the
corpus:

``` json
{
  "idx": "erlang_func_123",
  "url": "github.com/repo/module.erl#max/2", 
  "docstring": "Returns the maximum of two values",
  "code": "max(A, B) when A > B -> A;\nmax(A, B) -> B.",
  "code_tokens": ["max", "(", "A", ",", "B", ")", "when", "A", ">", "B", "->", "A", ";", "max", "(", "A", ",", "B", ")", "->", "B", "."],
  "dfg": [[0, 2], [1, 3], [4, 6], [5, 7]] // Data flow between clauses
}
```

## Guard Flows

Guards are additional conditions that can be checked after pattern matching succeeds. They're like if conditions but more restricted.

``` erlang
divide(A, B) when B =/= 0 -> A / B;
divide(_, 0) -> error.
```

GraphCodeBERT representation in the corpus:

```
[CLS] "Returns maximum value" [SEP] max(A,B) when A>B -> A; max(A,B) -> B. [SEP] A B A B [SEP]
```

Data Flow Graph:

```
Variables: A₁, B₁, A₂, B₂ (indexed by clause)
Edges: [(A₁,return₁), (B₁,guard₁), (B₂,return₂)]
```

## Message Passing

Erlang processes communicate by sending messages to each other. This is Erlang's concurrency model - no shared memory, only message passing.

``` erlang
loop(State) ->
    receive
        {update, NewState} -> loop(NewState);
        stop -> ok
    end.
```

## References

- [GraphCodeBERT Paper](<https://arxiv.org/abs/2009.08366>)
- [WhatsApp tree-sitter-erlang](<https://github.com/WhatsApp/tree-sitter-erlang>)
- [CodeSearchNet Dataset](<https://github.com/github/CodeSearchNet>)
- [Erlang/OTP Documentation](<https://erlang.org/doc/>)

