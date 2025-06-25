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
python -m venv venv
source venv/bin/activate
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

GraphCodeBERT representation in the corpus:

```
[CLS] "Returns maximum value" [SEP] max(A,B) when A>B -> A; max(A,B) -> B. [SEP] A B A B [SEP]
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
  "dfg": {
    # Edges: [A->A_clause1, B->B_clause1, A_clause1->A_guard, B_clause1->B_guard, A_guard->result_1, A->A_clause2, B->B_clause2, B_clause2->result_2]
    "variables": ["A", "B", "A_clause1", "B_clause1", "A_guard", "B_guard", "A_clause2", "B_clause2", "result_1", "result_2"],
    "edges": [[0,2], [1,3], [2,4], [3,5], [4,8], [0,6], [1,7], [7,9]]
  }
}
```

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

## Message Passing

Erlang processes communicate by sending messages to each other. This is Erlang's concurrency model - no shared memory, only message passing.

``` erlang
loop(State) ->
    receive
        {update, NewState} -> loop(NewState);
        stop -> ok
    end.
```


## Type specs

Add type specs to the comments:

```
%% Divides two numbers, returns number or error
%% @spec number(), number() -> number() | error
divide(A, B) when B =/= 0 -> A / B;
```

GraphCodeBERT sees types as natural language context. This fits existing GraphCodeBERT
architecture, but treats types as text, not structured data.

``` json
{
  "docstring": "Divides two numbers. @spec divide(number(), number()) -> number() | error.",
  "code": "divide(A, B) when B =/= 0 -> A / B;",
  "dfg": {"variables": [...], "edges": [...]}
}
```

## References

- [GraphCodeBERT Paper](<https://arxiv.org/abs/2009.08366>)
- [WhatsApp tree-sitter-erlang](<https://github.com/WhatsApp/tree-sitter-erlang>)
- [CodeSearchNet Dataset](<https://github.com/github/CodeSearchNet>)
- [Erlang/OTP Documentation](<https://erlang.org/doc/>)

