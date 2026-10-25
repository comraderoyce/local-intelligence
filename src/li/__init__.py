"""local-intelligence: abliteration + fine-tuning harness for Apple Silicon."""
import os

# HF tokenizers' Rust thread pool deadlocks when `datasets` forks workers after
# a tokenizer has run (seen as a 0%-CPU hang mid-eval)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
__version__ = "0.1.0"
