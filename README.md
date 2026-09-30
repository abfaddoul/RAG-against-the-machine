*This project has been created as part of the 42 curriculum by abdfaddo.*

# RAG Against the Machine

## Description

RAG Against the Machine is a Retrieval-Augmented Generation (RAG) system that
answers questions about the vLLM codebase.

The project ingests vLLM source code and documentation, splits the files into
searchable chunks, builds a persistent BM25 index, retrieves the most relevant
source locations for a question, and sends their text to the local
`Qwen/Qwen3-0.6B` model to generate a grounded answer.

The project also evaluates retrieval quality with recall@k. The official public
dataset requirements are at least 80% Recall@5 for documentation questions and
50% Recall@5 for source-code questions.

## Features

- Separate chunking strategies for Python and Markdown/text files
- Exact source paths and character-range tracking
- Persistent BM25 index
- Top-k retrieval for one question or a complete dataset
- Structured JSON input and output validated with Pydantic
- Recall@k evaluation with source-span overlap
- Local answer generation with `Qwen/Qwen3-0.6B`
- Batch progress bars with `tqdm`
- Python Fire command-line interface
- Unit tests, Flake8, and Mypy validation

## System Architecture

```text
vLLM corpus
    |
    v
File ingestion
    |
    v
Python or Markdown/text chunking
    |
    v
Tokenization and BM25 indexing
    |
    v
Persistent index in data/processed/
    |
    v
Question -> top-k retrieval -> source locations
    |
    v
Source text extraction and context construction
    |
    v
Prompt construction -> Qwen/Qwen3-0.6B
    |
    v
Grounded answer with retrieved sources
```

The main components are:

- `src/ingestion/`: discovers and reads useful corpus files.
- `src/chunking/`: selects a language-aware splitter and preserves source spans.
- `src/indexing/`: tokenizes chunks, builds BM25, and persists the index.
- `src/retrieval/`: ranks chunks and produces minimal source locations.
- `src/evaluation/`: calculates recall@k using file and span overlap.
- `src/generation/`: builds context and prompts, loads Qwen, and generates answers.
- `src/cli.py`: connects all stages through Python Fire commands.

## Chunking Strategy

The corpus uses two distinct LangChain-based strategies:

- Python files use a Python-aware recursive splitter so code structures are kept
  together when possible.
- Markdown and plain-text files use a Markdown/text recursive splitter that prefers
  natural section, paragraph, and line boundaries.

`--max_chunk_size` is configurable and defaults to 2000 characters, as required by
the subject. The current implementation uses an effective target of:

```text
min(max_chunk_size, 1000)
```

The 1000-character target was selected after retrieval tuning. It improved source
localization while ensuring that no retrieved source exceeds the moulinette limit of
2000 characters.

Each chunk stores:

- Its project-relative file path
- Its first character index
- Its last character index
- Its original text

This metadata allows retrieval results to point back to exact regions of the corpus.

## Retrieval Method

The project uses BM25, implemented with `rank-bm25`.

During indexing:

1. Every selected corpus file is split into chunks.
2. Each chunk is normalized and tokenized.
3. BM25 document statistics are calculated.
4. The BM25 object and chunk metadata are persisted with Joblib.

During retrieval:

1. The question is tokenized with the same tokenizer.
2. BM25 calculates a relevance score for every indexed chunk.
3. Chunks are ranked by score.
4. The top-k chunks are returned as `MinimalSource` objects.

BM25 was chosen because it is fast on CPU, works well with exact identifiers found in
source code, and satisfies the mandatory lexical-retrieval requirement without an
external vector database.

## Data Models and Output

Pydantic models validate the boundaries between all pipeline stages:

- `MinimalSource`
- `UnansweredQuestion`
- `AnsweredQuestion`
- `RagDataset`
- `MinimalSearchResults`
- `MinimalAnswer`
- `StudentSearchResults`
- `StudentSearchResultsAndAnswer`

Search commands write `StudentSearchResults` JSON. Batch answer generation writes
`StudentSearchResultsAndAnswer` JSON. Source paths remain relative to the project root
and match the ingested corpus paths exactly.

## Instructions

### Requirements

- Python `>=3.11,<3.14`
- `uv`
- Enough disk space for PyTorch, Transformers, and model weights
- CPU execution is supported; a GPU is optional

### Installation

```bash
git clone https://github.com/abfaddoul/RAG-against-the-machine.git
cd RAG-against-the-machine
uv sync
```

On systems with a small home partition, the caches can be moved to a larger disk:

```bash
export UV_CACHE_DIR=/goinfre/$USER/.cache/uv
export HF_HOME=/goinfre/$USER/.cache/huggingface
uv sync
```

### Prepare the Data

The expected evaluation layout is:

```text
data/
|-- raw/
|   `-- vllm-0.10.1/
|-- datasets/
|   |-- AnsweredQuestions/
|   `-- UnansweredQuestions/
|-- processed/
`-- output/
    |-- search_results/
    `-- search_results_and_answer/
```

The large corpus, datasets, indexes, generated outputs, and model weights are not
committed to Git. Place the provided vLLM repository under
`data/raw/vllm-0.10.1/` and the provided datasets under the matching
`data/datasets/` directories before running the pipeline.

### Makefile Commands

```bash
make install
make run
make debug
make clean
make lint
make test
```

`make lint` runs the mandatory Flake8 and Mypy checks.

## Example Usage

Every command is executed from the repository root with:

```text
uv run python -m src <command> [options]
```

### 1. Build the Index

```bash
uv run python -m src index \
  --max_chunk_size 2000
```

The index is saved under `data/processed/`.

### 2. Search One Question

```bash
uv run python -m src search \
  "What is LoRA?" \
  --k 5
```

### 3. Search a Dataset

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```

The command writes:

```text
data/output/search_results/UnansweredQuestions/dataset_docs_public.json
```

### 4. Evaluate Retrieval Locally

```bash
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

The official score is calculated by the separate moulinette executable, which is never
imported or called by the project:

```bash
./moulinette evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10 \
  --max_context_length 2000
```

### 5. Answer One Question

```bash
uv run python -m src answer \
  "What is LoRA and how is it used in vLLM?" \
  --k 5 \
  --max_new_tokens 96
```

### 6. Answer a Dataset

```bash
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions \
  --max_new_tokens 96
```

The resulting JSON follows `StudentSearchResultsAndAnswer` and preserves the question,
retrieved source locations, and generated answer.

## Answer Generation

The generation stack is:

```text
Qwen/Qwen3-0.6B
    -> Transformers text-generation pipeline
    -> HuggingFacePipeline
    -> ChatHuggingFace
```

For every question, the system:

1. Reads the exact text spans identified by retrieval.
2. Labels and joins them into a context string.
3. Builds separate system and human chat messages.
4. Instructs Qwen to use only the retrieved context.
5. Generates and cleans the final answer.
6. Returns a validated `MinimalAnswer`.

The model is cached inside a running Python process and reused during batch generation.
Tests inject a fake chat model so the unit-test suite does not download or run Qwen.

## Performance Analysis

The official public-dataset moulinette produced:

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Required Recall@5 |
|---|---:|---:|---:|---:|---:|
| Documentation | 60.0% | 76.0% | 80.0% | 83.0% | 80.0% |
| Source code | 38.4% | 62.6% | 69.7% | 77.8% | 50.0% |

Both mandatory Recall@5 thresholds are reached. The final index contained 29,378 chunks
created from 1,847 corpus files.

Smaller chunks improved recall because a retrieved span was less likely to mix unrelated
topics and was more likely to overlap a precise reference source. The trade-off is a
larger index and more candidate chunks to score.

BM25 retrieval is fast because the corpus statistics are precomputed and loaded from the
persisted index. Generation is the expensive stage on CPU. In one test environment, a
single 96-token answer took about one minute, while a batch smoke test with ten retrieved
sources took about 72 seconds for one question. Times vary by hardware, context length,
and output length.

## Design Decisions

- **BM25 instead of mandatory embeddings:** lexical retrieval is lightweight, fast on
  CPU, and effective for technical identifiers and exact code terms.
- **Language-aware chunking:** Python and documentation have different structural
  boundaries and therefore use different splitters.
- **Tuned 1000-character target:** smaller chunks improved recall while staying safely
  below the 2000-character evaluator limit.
- **Exact source metadata:** every result retains the original path and character span
  required by the evaluator.
- **Pydantic at stage boundaries:** malformed data is rejected before it propagates
  through the pipeline.
- **Persistent BM25 index:** indexing is performed once and retrieval commands reuse the
  saved index.
- **Dependency injection for generation tests:** fake models keep tests deterministic,
  offline, and fast.
- **Grounded prompt:** the model is told to use only retrieved context and to report when
  the sources are insufficient.

## Challenges Faced

### Preserving Exact Character Positions

Recursive splitting can make it difficult to map chunks back to the original file. Chunk
metadata was designed to retain exact start and end positions so paths and spans remain
compatible with the evaluator.

### Improving Recall Without Oversized Sources

Large chunks contain more context but reduce localization quality. Testing showed that an
effective 1000-character target improved Recall@5 while remaining below the 2000-character
limit.

### Matching Exact Corpus Paths

The moulinette compares file paths verbatim. The ingestion and chunking stages therefore
preserve project-relative paths beginning with `data/raw/vllm-0.10.1/`.

### Running Qwen on CPU

The evaluation environment may not provide a GPU. The project uses the small mandatory
Qwen model, a CPU PyTorch installation, process-level model caching, and configurable
generation length. Batch generation remains slower than retrieval.

### Qwen Thinking Markers

Qwen may return empty `<think>...</think>` markers even when non-thinking behavior is
requested. The prompt uses `/no_think`, and generated answers are cleaned before being
stored.

### Limited Home-Directory Storage

PyTorch and model files can require several gigabytes. `UV_CACHE_DIR` and `HF_HOME` can
be redirected to a larger filesystem without changing the project code.

## Testing and Code Quality

Run static analysis:

```bash
make lint
```

Run the unit-test suite:

```bash
make test
```

The current suite contains 59 tests covering models, ingestion, chunking, indexing,
retrieval, evaluation, prompt construction, context extraction, answer cleaning, and
batch generation.

## Resources

- [Retrieval-Augmented Generation paper](https://arxiv.org/abs/2005.11401)
- [BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)
- [rank-bm25](https://github.com/dorianbrown/rank_bm25)
- [vLLM documentation](https://docs.vllm.ai/)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [LangChain documentation](https://docs.langchain.com/)
- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [Python Fire documentation](https://google.github.io/python-fire/)
- [uv documentation](https://docs.astral.sh/uv/)
- [tqdm documentation](https://tqdm.github.io/)

### Use of AI

AI was used as a learning and development assistant for:

- Explaining RAG, chunking, BM25, recall@k, LangChain prompts, and model wrappers
- Reviewing implementation plans and data flow between pipeline stages
- Suggesting unit-test cases and edge cases
- Diagnosing dependency, type-checking, storage, and model-loading issues
- Reviewing CLI integration and helping structure this documentation

All suggested code and explanations were reviewed, understood, adapted, linted, and tested
before inclusion. Retrieval quality was verified independently with the provided
moulinette.

## Main Technologies

- Python
- Pydantic
- LangChain
- Hugging Face Transformers
- PyTorch
- Qwen
- rank-bm25
- Joblib
- Python Fire
- tqdm
- Pytest
- Mypy
- Flake8

## Author

- 42 login: `abdfaddo`
- GitHub: [abfaddoul](https://github.com/abfaddoul)

## Optional Features

Bonus implementation lives in `src/bonus/`, with its tests in `tests/bonus/`.
The mandatory BM25 pipeline and commands keep their original implementation.

| File | Responsibility |
|---|---|
| `embeddings.py` | Load MiniLM on CPU and embed chunk batches |
| `indexing.py` | Build HNSW and index the semantic corpus |
| `storage.py` | Save and load the semantic graph and metadata |
| `search.py` | Embed a query and retrieve nearest chunks |
| `hybrid.py` | Combine BM25 and semantic rankings with RRF |
| `incremental.py` | Detect file changes and update their semantic vectors |
| `cli.py` | Connect the four existing bonus commands |

```bash
uv run python -m src index_semantic
uv run python -m src search_semantic "How does vLLM load LoRA adapters?" --k 5
uv run python -m src search_hybrid "How does vLLM load LoRA adapters?" --k 5
uv run python -m src index_incremental
```

Incremental indexing updates the semantic index only. If the corpus changes and
hybrid search must use current lexical results too, run the mandatory `index`
command to refresh BM25. Incremental indexing requires the state saved by
`index_semantic`; it reports a clear error if that state is missing.

Existing semantic metadata and incremental state remain loadable through two
small compatibility modules in `src/indexing/`. They contain no indexing logic.
Caching and the local HTTP API are still pending. See `REFACTOR_NOTES.md` for the
scope, verification results, and how to apply this refactor to an existing checkout.
