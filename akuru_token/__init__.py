from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("bpe-tokenizer")
except PackageNotFoundError:
    # Package not installed
    __version__ = "dev"

from .tokenizer import BPETokenizer
from .trainer import BPETrainer
from .vocab import Vocab
from .pretokenizer import (
    BasePreTokenizer,
    WhitespacePreTokenizer,
    GPT2PreTokenizer,
    GraphemePreTokenizer,
    split_graphemes,
)

__all__ = [
    # Core
    "BPETokenizer",
    "BPETrainer",
    "Vocab",
    # Pre-tokenizers
    "BasePreTokenizer",
    "WhitespacePreTokenizer",
    "GPT2PreTokenizer",
    "GraphemePreTokenizer",
]