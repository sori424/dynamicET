import csv
import string
from collections import namedtuple
from pathlib import Path

from .common import BaseVocab
from .common import (
    Substring,
    TrackFormatter,
)


def _load_object_pairs(path: Path) -> list[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            (row["objects"], row["objects"])
            for row in csv.DictReader(handle)
            if row.get("objects")
        ]


OBJECT_PAIRS = _load_object_pairs(Path(__file__).with_name("objects.csv"))


class Vocab(BaseVocab):
    type = "BOXES"

    def __init__(self, tokenizer_type, split=None, tokenizer=None):
        super().__init__(tokenizer_type, tokenizer=tokenizer)

        self.filtered_names = list(string.ascii_uppercase)
        self.filtered_country_capital_pairs = self.filter_countries(OBJECT_PAIRS)
        if not self.filtered_country_capital_pairs:
            raise ValueError(
                "BOXES vocab has no single-token objects for this tokenizer. "
                "The evaluator requires single-token answer objects."
            )
        self.simple_train_test_split(split)

    def sample_swap_pair(self, entities, query_name=None):
        if len(entities) < 2:
            raise ValueError("Need at least two entities to sample a swap pair.")
        if query_name is not None:
            if query_name not in entities:
                raise ValueError(
                    f"Query box {query_name} is not present in entities {entities}."
                )
            other_entities = [entity for entity in entities if entity != query_name]
            partner = other_entities[
                int(self.global_rng.integers(0, len(other_entities)))
            ]
            if bool(self.global_rng.integers(0, 2)):
                return (query_name, partner)
            return (partner, query_name)

        sampled_indices = self.global_rng.choice(len(entities), size=2, replace=False)
        return tuple(entities[int(idx)] for idx in sampled_indices)

    def sample_query_name(self, context, query_name=None):
        if query_name is not None:
            return query_name

        swap_candidates = []
        for statement in context:
            if statement.type == "swap":
                swap_candidates.extend((statement.name, statement.attr))

        if swap_candidates:
            return swap_candidates[
                int(self.global_rng.integers(0, len(swap_candidates)))
            ]

        normal_candidates = [
            statement.name
            for statement in context
            if statement.type in ("normal", "ref")
        ]
        if not normal_candidates:
            raise ValueError("Could not sample a query box from an empty BOXES context.")
        return normal_candidates[
            int(self.global_rng.integers(0, len(normal_candidates)))
        ]

    def default_context(
        self,
        num_entities,
        entities=None,
        attributes=None,
        include_swap=True,
        swap_pair=None,
        query_name=None,
    ):
        if entities is None:
            entities = range(num_entities)
        if attributes is None:
            attributes = range(num_entities)
        entities = list(entities)
        attributes = list(attributes)
        context = [Statement(e, s, "normal") for e, s in zip(entities, attributes)]
        if include_swap and num_entities >= 2:
            if swap_pair is None:
                swap_pair = self.sample_swap_pair(entities, query_name=query_name)
            elif query_name is not None and query_name not in swap_pair:
                raise ValueError(
                    f"Query box {query_name} must be part of swap_pair {swap_pair}."
                )
            context.append(Statement(swap_pair[0], swap_pair[1], "swap"))
        return context

    def fetch_contextualized_shuffled_attr(
        self, attr_id, prompt_id, context=None, query_name=None
    ):
        del query_name
        if context is None:
            return self.fetch_shuffled_attr(attr_id, prompt_id)

        final_attrs_by_box = {
            statement.name: statement.attr
            for statement in context
            if statement.type in ("normal", "ref")
        }
        if attr_id not in final_attrs_by_box:
            return self.fetch_shuffled_attr(attr_id, prompt_id)

        for statement in context:
            if statement.type != "swap":
                continue
            left_box, right_box = statement.name, statement.attr
            if (
                left_box not in final_attrs_by_box
                or right_box not in final_attrs_by_box
            ):
                continue
            final_attrs_by_box[left_box], final_attrs_by_box[right_box] = (
                final_attrs_by_box[right_box],
                final_attrs_by_box[left_box],
            )

        return self.fetch_shuffled_attr(final_attrs_by_box[attr_id], prompt_id)


Statement = namedtuple("Statement", ["name", "attr", "type"])


class TEMPLATES:
    @classmethod
    def lookup(cls, template_type):
        if template_type == "normal":
            return cls.DEFAULT
        raise ValueError(f"unknown template {template_type}")

    class DEFAULT:
        @classmethod
        def lookup(cls, context_type):
            if context_type in {"normal", "ref"}:
                return cls.context_template
            if context_type == "swap":
                return cls.swap_template
            raise ValueError(f"unknown context template {context_type}")

        @classmethod
        def generate_template(cls, *, vocab, template_context, prompt_id, context):
            @lambda f: f(**template_context)
            def ret(query_name, raw_query_name):
                names_perm, _ = vocab.get_shuffled_labels(prompt_id)
                selected_query_name = vocab.sample_query_name(
                    context,
                    query_name=query_name,
                )
                return (
                    cls.template,
                    dict(
                        qn_subject=vocab.filtered_names[names_perm[selected_query_name]]
                        if raw_query_name is None
                        else raw_query_name,
                    ),
                )

            return ret

        @classmethod
        def extract_template_indices(cls, full_output_indices):
            return {
                "qn_subject": full_output_indices["qn_subject"][0],
                "ans_subject": full_output_indices["qn_subject"][-1],
            }

        @classmethod
        def instantiate(cls, vocab, statement, prompt_id, template_context):
            names_perm, attrs_perm = vocab.get_shuffled_labels(prompt_id)
            if statement.type in {"normal", "ref"}:
                context_template = cls.lookup(statement.type)
                name = vocab.filtered_names[names_perm[statement.name]]
                object_word, _ = vocab.filtered_country_capital_pairs[
                    attrs_perm[statement.attr]
                ]
                cur_ctx, ctx_idx_map = TrackFormatter().format(
                    context_template,
                    subject=name,
                    object=object_word,
                )
                return (
                    cur_ctx,
                    {
                        "subject": ctx_idx_map["subject"][0],
                        "country": ctx_idx_map["object"][0],
                        "object": ctx_idx_map["object"][0],
                        "sentence": Substring(0, len(cur_ctx)),
                    },
                )
            elif statement.type == "swap":
                context_template = cls.lookup(statement.type)
                left_box = vocab.filtered_names[names_perm[statement.name]]
                right_box = vocab.filtered_names[names_perm[statement.attr]]
                cur_ctx, ctx_idx_map = TrackFormatter().format(
                    context_template,
                    left_box=left_box,
                    right_box=right_box,
                )
                return (
                    cur_ctx,
                    {
                        "left_box": ctx_idx_map["left_box"][0],
                        "right_box": ctx_idx_map["right_box"][0],
                        "sentence": Substring(0, len(cur_ctx)),
                    },
                )
            else:
                raise ValueError(f"unknown statement type {statement.type}")

        template = (
            "Answer the question based on the context below. Answer with the single object word only.\n\n"
            "Context:{context}\n\n"
            "Question: Which item does Box {qn_subject} contain?\n\n"
            "Answer:"
        )

        context_template = """ Box {subject} contains the {object}."""
        swap_template = """ Swap the items of Box {left_box} and Box {right_box}."""
