from __future__ import annotations

import bisect
import csv
import random
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from string import Formatter
from typing import Optional

try:
    from coref import COREF_ROOT
except ImportError:
    COREF_ROOT = Path(__file__).resolve().parents[1]
else:
    COREF_ROOT = Path(COREF_ROOT)

try:
    from coref.models import HF_PATH, get_gemma_tokenizer, get_llama_tokenizer
except ImportError:
    HF_PATH = None

    def get_gemma_tokenizer():
        raise ImportError("coref.models.get_gemma_tokenizer is not available.")

    def get_llama_tokenizer():
        raise ImportError("coref.models.get_llama_tokenizer is not available.")


RAW_DATA_DIR = COREF_ROOT / "coref" / "datasets" / "raw"


def _read_csv_column(path: Path, column: str) -> list[str]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [row[column] for row in csv.DictReader(handle) if row.get(column)]


def _read_country_capital_pairs(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            (row["country"], row["capital"])
            for row in csv.DictReader(handle)
            if row.get("country") and row.get("capital")
        ]


names = _read_csv_column(RAW_DATA_DIR / "new-top-firstNames.csv", "name")
country_capital_pairs = _read_country_capital_pairs(RAW_DATA_DIR / "country-list.csv")
colors = _read_csv_column(RAW_DATA_DIR / "colors.csv", "color")


class Name(IntEnum):
    a = 0
    b = 1
    c = 2
    d = 3
    e = 4
    f = 5


class Attr(IntEnum):
    a = 0
    b = 1
    c = 2
    d = 3
    e = 4
    f = 5


def train_test_split(lst, frac):
    mid = int(len(lst) * frac)
    return lst[:mid], lst[mid:]


class TrackFormatter(Formatter):
    def format(self, format_string, **kwargs):
        """
        Only accepts keyword arguments.

        Returns:
            formatted string : str
            locations : Dict[str, List[Tuple[int, int]]] - denoting start and end positions (inclusive, exclusive)
        """
        locations = defaultdict(list)
        result = []
        run_length = 0
        for literal_text, field_name, format_spec, conversion in self.parse(
            format_string
        ):
            # output the literal text
            if literal_text:
                result.append(literal_text)
                run_length += len(literal_text)

            if field_name is not None:
                # given the field_name, find the object it references
                #  and the argument it came from
                obj, arg_used = self.get_field(field_name, [], kwargs)

                # do any conversion on the resulting object
                obj = self.convert_field(obj, conversion)

                # format the object and append to the result
                final_str = self.format_field(obj, format_spec)
                locations[arg_used].append(
                    Substring(run_length, run_length + len(final_str))
                )
                result.append(final_str)
                run_length += len(final_str)

        return "".join(result), locations


@dataclass(frozen=True)
class Substring:
    start: int
    end: int

    def __iter__(self):
        return iter((self.start, self.end))

    def __getitem__(self, key):
        if key == 0:
            return self.start
        if key == 1:
            return self.end
        raise IndexError(key)

    def to_slice(self):
        return slice(self.start, self.end)

    def __add__(self, num):
        return Substring(self.start + num, self.end + num)


def recursify(func, dtype=Substring, pred=None):
    if pred is None:
        pred = lambda x: isinstance(x, dtype)

    def wrapper(indices, *args, **kwargs):
        if pred(indices):
            return func(indices, *args, **kwargs)
        elif isinstance(indices, dict):
            return {
                key: wrapper(value, *args, **kwargs) for key, value in indices.items()
            }
        elif isinstance(indices, list):
            return [wrapper(value, *args, **kwargs) for value in indices]
        else:
            raise TypeError(f"Unexpected index type: {type(indices).__name__}")

    return wrapper


@recursify
def recursive_align_tokens(indices, offset_mapping):
    # it's unclear what conventions offset_mapping uses
    # but we can say for sure that starting indices are inclusive
    # but my indices are inclusive, exclusive
    start, end = indices
    start = bisect.bisect_right([x for x, _ in offset_mapping], start) - 1
    end = bisect.bisect_right([x for x, _ in offset_mapping], end - 1) - 1
    return Substring(start, end + 1)


@recursify
def recursive_add_offset(indices, offset):
    return indices + offset


def rotate(func, dtype=Substring, pred=None):
    """
    Rotates "List ... x" into "... List x", and calls func on List x

    Returns:
        ... func(List x)
    """
    if pred is None:
        pred = lambda x: isinstance(x, dtype)

    def wrapper(indices, *args, **kwargs):
        assert isinstance(indices, list)
        child = indices[0]
        if pred(child):
            return func(indices, *args, **kwargs)
        elif isinstance(child, dict):
            return {
                key: wrapper([sib[key] for sib in indices], *args, **kwargs)
                for key in child.keys()
            }

        elif isinstance(child, list):
            return [
                wrapper([sib[key] for sib in indices], *args, **kwargs)
                for key in range(len(child))
            ]
        else:
            raise TypeError(f"Unexpected index type: {type(child).__name__}")

    return wrapper


class PromptRng:
    """Small deterministic RNG wrapper with the methods this project uses."""

    def __init__(self, seed: Optional[int]):
        self._random = random.Random(seed)

    def integers(self, low: int, high: Optional[int] = None) -> int:
        if high is None:
            low, high = 0, low
        return self._random.randrange(int(low), int(high))

    def permutation(self, length: int) -> list[int]:
        values = list(range(int(length)))
        self._random.shuffle(values)
        return values

    def choice(self, values, size: Optional[int] = None, replace: bool = True):
        population = list(range(values)) if isinstance(values, int) else list(values)
        if size is None:
            return self._random.choice(population)
        if replace:
            return [self._random.choice(population) for _ in range(int(size))]
        return self._random.sample(population, int(size))


class BaseVocab:
    train_prop = 0.5
    min_items_per_split = 2

    def __init__(self, tokenizer_type, tokenizer=None):
        """
        tokenizer_type: "pythia", "llama", or "gemma"
        """
        if tokenizer_type == "pythia":
            self.LLAMA = False
        elif tokenizer_type == "llama":
            self.LLAMA = True
        elif tokenizer_type == "gemma":
            self.LLAMA = True
        else:
            raise ValueError(f"Unknown tokenizer type {tokenizer_type}")
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif tokenizer_type == "llama":
            self.tokenizer = get_llama_tokenizer()
        elif tokenizer_type == "gemma":
            self.tokenizer = get_gemma_tokenizer()
        else:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                "EleutherAI/pythia-2.8b",
                **(dict(cache_dir=HF_PATH) if HF_PATH is not None else {}),
            )
        self.global_rng = PromptRng(42)

    def filter_names(self, names, length=1):
        return [name for name in names if self.get_length_of_word(name) == length]

    def filter_countries(self, country_capital_pairs, length=1):
        return [
            pair
            for pair in country_capital_pairs
            if self.get_length_of_word(pair[0]) == length
            and self.get_length_of_word(pair[1]) == length
        ]

    def get_rng(self, prompt_id):
        if prompt_id is None:
            return self.global_rng
        return PromptRng(int(prompt_id))

    def generate_permutations(self, prompt_id, lengths):
        rng = self.get_rng(prompt_id)
        return [rng.permutation(l) for l in lengths]

    def get_shuffled_labels(self, prompt_id=None):
        return self.generate_permutations(
            prompt_id,
            [len(self.filtered_names), len(self.filtered_country_capital_pairs)],
        )

    def fetch_shuffled_name(self, name_id, prompt_id):
        names_perm, _ = self.get_shuffled_labels(prompt_id)
        return self.filtered_names[names_perm[name_id]]

    def fetch_shuffled_attr(self, attr_id, prompt_id):
        _, attrs_perm = self.get_shuffled_labels(prompt_id)
        return self.filtered_country_capital_pairs[attrs_perm[attr_id]]

    def fetch_contextualized_shuffled_attr(
        self, attr_id, prompt_id, context=None, query_name=None
    ):
        return self.fetch_shuffled_attr(attr_id, prompt_id)

    def _word_tokens(self, word):
        # Vocab entries are used mid-sentence / as next-token continuations.
        # Some SentencePiece tokenizers split a string that *starts* with a
        # space into a separate space token, so compare against a normal prefix
        # and keep only the continuation tokens for the word.
        prefix = "The"
        full_ids = self.tokenizer(
            f"{prefix} {word}",
            add_special_tokens=False,
        )["input_ids"]
        prefix_ids = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
        if full_ids[: len(prefix_ids)] == prefix_ids:
            return full_ids[len(prefix_ids) :]
        return self.tokenizer(" " + word, add_special_tokens=False)["input_ids"]

    def encode_single_word(self, word):
        stuff = self._word_tokens(word)
        assert len(stuff) == 1
        return stuff[0]

    def get_length_of_word(self, word):
        return len(self._word_tokens(word))

    def simple_train_test_split(self, split):
        def maybe_split(items):
            min_total = self.min_items_per_split * 2
            if len(items) < min_total:
                return items, items
            return train_test_split(items, self.train_prop)

        if split == "train":
            self.filtered_names, _ = maybe_split(self.filtered_names)
            self.filtered_country_capital_pairs, _ = maybe_split(
                self.filtered_country_capital_pairs
            )
        elif split == "test":
            _, self.filtered_names = maybe_split(self.filtered_names)
            _, self.filtered_country_capital_pairs = maybe_split(
                self.filtered_country_capital_pairs
            )
        elif split is not None:
            raise ValueError(
                f"Unknown split {split!r}; expected 'train', 'test', or None."
            )

    @staticmethod
    def default_context(num_entities, entities=None, attributes=None):
        if entities is None:
            entities = range(num_entities)
        if attributes is None:
            attributes = range(num_entities)
        return [Statement(e, s, "normal") for e, s in zip(entities, attributes)]


def tokenize_prompt(*, vocab, prompt, indices, **kwargs):
    tokenized = vocab.tokenizer(prompt, return_offsets_mapping=True)
    aligned_tokens = recursive_align_tokens(indices, tokenized["offset_mapping"])
    return tokenized["input_ids"], aligned_tokens


def stack_tokens(vocab, tokens_list):
    import torch

    longest_length = max(len(s) for s in tokens_list)
    pad_token_id = vocab.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = vocab.tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = getattr(vocab.tokenizer, "pad_token_type_id", 0)
    padded = torch.tensor(
        [
            s + [pad_token_id] * (longest_length - len(s))
            for s in tokens_list
        ]
    )
    return padded


default_mcq_labels = ["A", "B", "C", "D", "E", "F", "G"]


def generate_shuffler(prompt_id, num_entities):
    rng = PromptRng(int(prompt_id))
    perm = rng.permutation(num_entities)
    inv_perm = [0] * num_entities
    for i, p in enumerate(perm):
        inv_perm[p] = i

    def shuffler(ls, invert=False):
        assert len(ls) == num_entities
        if invert:
            return [ls[p] for p in perm]
        return [ls[p] for p in inv_perm]

    return shuffler


def lookup_answer_logits(
    logits,
    answer_tokens,
    *,
    query_position=None,
):
    """Return logits for each candidate answer token at each prompt's answer site."""

    import torch

    if logits.ndim != 3:
        raise ValueError(
            f"Expected logits [batch, seq, vocab], got {tuple(logits.shape)}."
        )
    if answer_tokens.ndim != 2:
        raise ValueError(
            f"Expected answer tokens [batch, candidates], got {tuple(answer_tokens.shape)}."
        )

    batch_size = logits.shape[0]
    if query_position is None:
        query_position = torch.full(
            (batch_size,),
            logits.shape[1] - 1,
            dtype=torch.long,
            device=logits.device,
        )
    else:
        query_position = query_position.to(device=logits.device, dtype=torch.long)

    rows = torch.arange(batch_size, device=logits.device)
    site_logits = logits[rows, query_position]
    answer_tokens = answer_tokens.to(device=logits.device, dtype=torch.long)
    return site_logits.gather(1, answer_tokens)


Statement = namedtuple("Statement", ["name", "attr", "type"])

MCQStatement = namedtuple(
    "MCQStatement", ["label", "option", "type", "raw_label"], defaults=[None]
)

ICLStatement = namedtuple(
    "ICLStatement", ["question", "answer", "label", "type"], defaults=[None]
)
