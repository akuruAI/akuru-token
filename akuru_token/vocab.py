"""
Vocab: stores token:id mappings and the ordered list of BPE merge rules.

Built-in vocab files live in the `vocabs/` directory inside the package.
The save() method allows writing new files into the vocabs/ directory, 
but will not overwrite an existing file there. Delete manually to overwrite.

Usage examples
--------------
# Load the default built-in vocab (sin_eng):
vocab = Vocab.load("sin_eng")

# Load another built-in vocab by name:
vocab = Vocab.load("my_custom") - looks up vocabs/my_custom.json

# Load from an explicit path:
vocab = Vocab.load("/tmp/my_vocab.json")

# Save to the vocabs dir (only if it doesn't already exist):
vocab.save("new_vocab") - writes to vocabs/new_vocab.json

# Save to an explicit path:
vocab.save("/tmp/my_vocab.json")

# List all built-in vocabs:
Vocab.list_vocabs()  - returns ["sin_eng", ...]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Directory that ships with the package containing built-in vocab files.
_VOCABS_DIR = Path(__file__).parent / "vocabs"


class Vocab:
    """
    Holds the vocabulary (token:id mappings) and BPE merge rules.

    Merge rules are stored in priority order (earliest = highest priority).
    """

    SPECIAL_UNK = "<unk>"
    SPECIAL_PAD = "<pad>"
    SPECIAL_BOS = "<bos>"
    SPECIAL_EOS = "<eos>"

    def __init__(self) -> None:
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []  # ordered merge rules

    def add_token(self, token: str) -> int:
        """Add a token if it doesn't exist; return its id."""
        if token not in self._token_to_id:
            idx = len(self._token_to_id)
            self._token_to_id[token] = idx
            self._id_to_token[idx] = token
        return self._token_to_id[token]

    def token_to_id(self, token: str) -> Optional[int]:
        return self._token_to_id.get(token)

    def id_to_token(self, idx: int) -> Optional[str]:
        return self._id_to_token.get(idx)

    def __len__(self) -> int:
        return len(self._token_to_id)

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_id

    def __repr__(self) -> str:
        return f"Vocab(size={len(self)}, merges={len(self.merges)})"

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def unk_id(self) -> int:
        return self._token_to_id[self.SPECIAL_UNK]
    
    def describe(self) -> None:
        """Print a human-readable summary of the vocabulary to stdout."""
        specials = [
            (tok, self._token_to_id[tok])
            for tok in (self.SPECIAL_UNK, self.SPECIAL_PAD, self.SPECIAL_BOS, self.SPECIAL_EOS)
            if tok in self._token_to_id
        ]
        special_str = "  ".join(f"{tok}={idx}" for tok, idx in specials)
        first_tokens = [self._id_to_token[i] for i in range(min(10, len(self)))]
        first_merges = [f"({a!r}, {b!r}) → {a+b!r}" for a, b in self.merges[:5]]

        print(repr(self))
        print(f"  Special tokens : {special_str or '(none)'}")
        print(f"  First 10 tokens: {', '.join(repr(t) for t in first_tokens)}")
        print(f"  First 5 merges : {', '.join(first_merges) or '(none)'}")

    def tokens(self, start: int = 0, end: Optional[int] = None) -> List[Tuple[str, int]]:
        """
        Return a slice of the vocabulary as (token, id) pairs, ordered by id.

        Parameters
        ----------
        start:
            First id to include (default 0).
        end:
            One past the last id to include (default: end of vocab).
        """
        end = len(self) if end is None else end
        return [
            (self._id_to_token[i], i)
            for i in range(start, end)
            if i in self._id_to_token
        ]


    @staticmethod
    def _resolve_load_path(name_or_path: str | Path) -> Path:
        """
        Resolve a name or path to an existing vocab file.

          - Bare name (e.g. "sin_eng" or "sin_eng.json") - looks in vocabs dir.
          - Anything else - treated as an explicit file path.
        """
        p = Path(name_or_path)

        if p.exists() and p.is_file():
            return p

        if p.parent == Path("."):
            candidate = _VOCABS_DIR / f"{p.stem}.json"
            if candidate.exists():
                return candidate
            raise FileNotFoundError(
                f"No built-in vocab named {p.stem!r}. "
                f"Available: {Vocab.list_builtin()}. "
                f"To load a custom file, supply its full path."
            )

        raise FileNotFoundError(f"Vocab file not found: {p}")

    @staticmethod
    def _resolve_save_path(name_or_path: str | Path) -> Path:
        """
        Resolve a name or path to the destination for saving.

          - Bare name -> vocabs/ directory.
          - Anything else -> treated as an explicit file path.

        Raises FileExistsError if the resolved path already exists inside
        the vocabs/ directory — delete the file manually to replace it.
        """
        p = Path(name_or_path)

        if p.parent == Path("."):
            dest = _VOCABS_DIR / f"{p.stem}.json"
        else:
            dest = p

        if dest.exists() and dest.resolve().parent == _VOCABS_DIR.resolve():
            raise FileExistsError(
                f"{dest.name!r} already exists in the built-in vocabs directory. "
                f"Delete it manually if you want to replace it."
            )

        return dest

    def save(self, name_or_path: str | Path) -> None:
        """
        Save vocab and merges to a JSON file.

        Parameters
        ----------
        name_or_path:
            - A bare name like ``"my_vocab"`` -> saved to the package's
              built-in ``vocabs/`` directory as ``vocabs/my_vocab.json``.
            - A full or relative file path for saving outside the package.

        Raises
        ------
        FileExistsError
            If the resolved path already exists inside the vocabs/ directory.
            Delete the file manually via the filesystem to replace it.
        """
        dest = self._resolve_save_path(name_or_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vocab": self._token_to_id,
            "merges": self.merges,
        }
        dest.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, name_or_path: str | Path = "sin_eng") -> "Vocab":
        """
        Load a vocab from a built-in name or an explicit file path.

        Parameters
        ----------
        name_or_path:
            - A bare name like ``"sin_eng"`` or ``"sin_eng.json"`` -> loaded
              from the package's built-in ``vocabs/`` directory.
            - A full or relative file path to a custom JSON file.
            - Defaults to ``"sin_eng"`` when called with no arguments.
        """
        path = cls._resolve_load_path(name_or_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        vocab = cls()
        for token, idx in data["vocab"].items():
            vocab._token_to_id[token] = idx
            vocab._id_to_token[idx] = token
        vocab.merges = [tuple(pair) for pair in data["merges"]]
        return vocab

    @staticmethod
    def list_vocabs() -> List[str]:
        """
        Return the names of all built-in vocab files shipped with the package.

        Returns stems only (no ``.json`` extension), sorted alphabetically.

        Example
        -------
        >>> Vocab.list_vocabs()
        ['sin_eng']
        """
        return sorted(p.stem for p in _VOCABS_DIR.glob("*.json"))