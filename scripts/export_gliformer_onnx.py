#!/usr/bin/env python3
"""Export a GLiFormer checkpoint to local ONNX files.

The official forward takes a Python ``classes_mapping`` and cannot be traced
as one graph. This script writes a split runtime under ``models/<checkpoint>/``:

* ``onnx/encoder.onnx`` — token embeddings
* ``onnx/ner.onnx`` — BIO logits from gathered word, label, and parent embeddings
* ``onnx/classification.onnx`` — class logits, CLS-pooled
* ``onnx/relations.onnx`` — joint relation scores for caller-supplied spans
* ``onnx/structuring.onnx`` — record anchors, span membership, and anchor relations when the head is multi-level

Rust gathers prompt features between the encoder and the heads. Span selection
stays out of these graphs. The script fails if ONNX logits diverge from PyTorch.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download

REPO_ID = "knowledgator/gliformer-base-v1"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "models" / "gliformer-base-v1"
ONNX_DIR = OUT_DIR / "onnx"
OPSET = 18
ATOL = 1e-3


class EncoderExport(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.layer(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            return output.last_hidden_state
        if isinstance(output, (tuple, list)):
            return output[0]
        raise TypeError(f"unexpected encoder output {type(output)}")


class NerExport(nn.Module):
    def __init__(self, head: nn.Module, rnn: nn.Module):
        super().__init__()
        self.head = head
        self.rnn = rnn

    def forward(
        self,
        words: torch.Tensor,
        word_mask: torch.Tensor,
        children: torch.Tensor,
        child_mask: torch.Tensor,
        parent: torch.Tensor,
    ) -> torch.Tensor:
        words = self.rnn(words, word_mask)
        flat = SimpleNamespace(
            words_embedding=words,
            mask=word_mask,
            child_embedding=children,
            child_mask=child_mask,
            parent_embedding=parent,
        )
        scores, _, _, _, _ = self.head._compute_bio_scores(flat, {})
        return scores.squeeze(1)


class ClassificationExport(nn.Module):
    def __init__(self, head: nn.Module, rnn: nn.Module):
        super().__init__()
        self.head = head
        self.rnn = rnn

    def forward(
        self,
        words: torch.Tensor,
        word_mask: torch.Tensor,
        children: torch.Tensor,
        child_mask: torch.Tensor,
        parent: torch.Tensor,
        cls_embed: torch.Tensor,
    ) -> torch.Tensor:
        words = self.rnn(words, word_mask)
        anchor_rep, anchor_mask = self.head._generate_anchors(
            parent,
            words,
            feature_mask=word_mask,
        )
        anchor_rep = self.head._refine_anchors(
            anchor_rep,
            words,
            memory_mask=word_mask,
            anchor_mask=anchor_mask,
        )
        fused = self.head._reduce_fused_anchors(
            self.head._model_anchors(
                anchor_rep,
                children,
                anchor_mask=anchor_mask,
                child_mask=child_mask,
            ),
            anchor_mask,
        )
        return self.head.scorer(cls_embed, fused)


class RelationsExport(nn.Module):
    def __init__(self, head: nn.Module, rnn: nn.Module):
        super().__init__()
        self.head = head
        self.rnn = rnn

    def forward(
        self,
        words: torch.Tensor,
        word_mask: torch.Tensor,
        span_idx: torch.Tensor,
        span_mask: torch.Tensor,
        relations: torch.Tensor,
    ) -> torch.Tensor:
        words = self.rnn(words, word_mask)
        span_mask_bool = span_mask.to(torch.bool)
        safe_idx = span_idx * span_mask_bool.unsqueeze(-1).long()
        span_rep = self.head.rel_span_rep_layer(words, safe_idx)
        span_rep = span_rep * span_mask_bool.unsqueeze(-1).to(span_rep.dtype)
        # This checkpoint scores every directed pair. A dense (span, span, relation)
        # tensor keeps the span count as an input axis instead of a data-dependent pair list.
        head = span_rep.unsqueeze(2).expand(-1, span_rep.shape[1], span_rep.shape[1], -1)
        tail = span_rep.unsqueeze(1).expand(-1, span_rep.shape[1], span_rep.shape[1], -1)
        pair_rep = self.head.pair_rep_layer(torch.cat((head, tail), dim=-1))
        scores = torch.einsum("bijd,bcd->bijc", pair_rep, relations)
        valid = span_mask_bool.unsqueeze(2) & span_mask_bool.unsqueeze(1)
        index = torch.arange(span_rep.shape[1], device=span_rep.device)
        diagonal = index.unsqueeze(0) == index.unsqueeze(1)
        valid = valid & ~diagonal.unsqueeze(0)
        return scores * valid.unsqueeze(-1).to(scores.dtype)


class StructuringExport(nn.Module):
    def __init__(self, head: nn.Module, rnn: nn.Module, export_relations: bool = False):
        super().__init__()
        self.head = head
        self.rnn = rnn
        self.export_relations = export_relations

    def forward(
        self,
        words: torch.Tensor,
        word_mask: torch.Tensor,
        span_idx: torch.Tensor,
        span_mask: torch.Tensor,
        parent: torch.Tensor,
    ):
        words = self.rnn(words, word_mask)
        span_mask_bool = span_mask.to(torch.bool)
        safe_idx = span_idx * span_mask_bool.unsqueeze(-1).long()
        span_rep = self.head.entity_span_rep_layer(words, safe_idx)
        span_rep = span_rep * span_mask_bool.unsqueeze(-1).to(span_rep.dtype)
        flat = SimpleNamespace(
            words_embedding=words,
            mask=word_mask,
            parent_embedding=parent,
        )
        anchors, anchor_mask = self.head._record_anchors(flat, {})
        membership = self.head._score_anchor_membership(
            span_rep,
            span_mask_bool,
            anchors,
            anchor_mask,
        )
        objectness = self.head.objectness_head(anchors).squeeze(-1)
        anchor_mask = anchor_mask.to(membership.dtype)
        if not self.export_relations:
            return membership, objectness, anchor_mask
        relations = self.head._score_anchor_relations(anchors, anchor_mask.bool())
        return membership, objectness, anchor_mask, relations


def _export(module: nn.Module, args, path: Path, input_names, output_names, dynamic_axes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module,
        args,
        str(path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        dynamo=False,
    )


def _max_abs(left, right) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def _check(name: str, left, right) -> None:
    delta = _max_abs(left, right)
    print(f"  {name}: max abs {delta:.3e} shape {tuple(np.asarray(left).shape)}")
    if delta > ATOL:
        raise SystemExit(f"{name} diverged from PyTorch (max abs {delta})")


def _load_model():
    from gliformer import GLiFormer

    model = GLiFormer.from_pretrained(REPO_ID, load_tokenizer=True)
    model.cpu().eval()
    return model


def _copy_checkpoint_files() -> None:
    snapshot = Path(snapshot_download(REPO_ID, allow_patterns=["tokenizer.json", "gliner_config.json", "tokenizer_config.json"]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "gliner_config.json", "tokenizer_config.json"):
        source = snapshot / name
        if source.exists():
            shutil.copyfile(source, OUT_DIR / name)


def _official_batch(model, text: str, **tasks):
    tokens, _, _ = model.prepare_inputs([text])
    examples = model._build_inference_input(tokens, **tasks)
    collator_cls = model.data_collator_class
    collator = collator_cls(
        model.config,
        data_processor=model.data_processor,
        return_tokens=True,
        prepare_labels=False,
    )
    batch = collator(examples)
    allowed = {
        "input_ids",
        "attention_mask",
        "words_mask",
        "text_lengths",
        "token_type_ids",
    }
    tensors = {key: value for key, value in batch.items() if key in allowed and torch.is_tensor(value)}
    with torch.no_grad():
        output = model.model(**tensors, classes_mapping=batch["classes_mapping"])
    return batch, output


def _gather(token_embeds, input_ids, words_mask, token_id: int):
    positions = (input_ids[0] == token_id).nonzero(as_tuple=False).flatten()
    if positions.numel() == 0:
        return token_embeds.new_zeros((1, 0, token_embeds.shape[-1]))
    return token_embeds[0, positions].unsqueeze(0)


def _gather_words(token_embeds, words_mask):
    mask = words_mask[0]
    word_count = int(mask.max().item()) if mask.numel() else 0
    hidden = token_embeds.shape[-1]
    words = token_embeds.new_zeros((1, word_count, hidden))
    for position, word_id in enumerate(mask.tolist()):
        if word_id > 0:
            words[0, int(word_id) - 1] = token_embeds[0, position]
    return words


def _run(session, feed):
    names = {item.name for item in session.get_inputs()}
    missing = names.difference(feed)
    if missing:
        raise SystemExit(f"ONNX session missing feeds {sorted(missing)}")
    outputs = session.run(None, {name: feed[name] for name in names})
    return outputs


def _relax_structuring_trace_checks(head: nn.Module) -> None:
    """Drop shape assertions that compare traced tensor sizes with Python ints."""

    positions = getattr(head, "record_anchor_refine_positions", None)
    if positions is None:
        return
    position_cls = type(positions)

    @staticmethod
    def coordinate_batch(coordinates, *, batch_size, count, dimensions, device, name):
        del batch_size, count, dimensions, name
        if coordinates.dim() == 2:
            coordinates = coordinates.unsqueeze(0)
        return coordinates.to(device=device)

    @staticmethod
    def mask(mask, shape, *, device, name):
        del name
        if mask is None:
            return torch.ones(shape, dtype=torch.bool, device=device)
        return mask.to(device=device).bool()

    position_cls._coordinate_batch = coordinate_batch
    position_cls._mask = mask

    from gliformer.layers import attention_bias

    def coordinates(value, *, name, length, device):
        del name, length
        if value.dim() == 2:
            value = value.unsqueeze(0)
        return value.to(device=device, dtype=torch.float32)

    def broadcast(queries, keys):
        return queries, keys

    attention_bias._coordinates = coordinates
    attention_bias._broadcast_coordinate_batches = broadcast


def main() -> None:
    import onnxruntime as ort

    global REPO_ID, OUT_DIR, ONNX_DIR
    parser = argparse.ArgumentParser(description="Export a GLiFormer checkpoint to local ONNX files.")
    parser.add_argument(
        "repo_id",
        nargs="?",
        default="knowledgator/gliformer-base-v1",
        help="Hugging Face repo id (default: knowledgator/gliformer-base-v1)",
    )
    args = parser.parse_args()
    REPO_ID = args.repo_id
    OUT_DIR = ROOT / "models" / REPO_ID.split("/")[-1]
    ONNX_DIR = OUT_DIR / "onnx"

    print(f"loading {REPO_ID}")
    model = _load_model()
    inner = model.model
    _relax_structuring_trace_checks(inner.heads["structuring"])
    ner_head = inner.heads["ner"]
    rnn = inner.rnn
    _copy_checkpoint_files()
    broken = ONNX_DIR / "model.onnx"
    if broken.exists():
        broken.unlink()

    text = "Alice works at Acme in London."
    labels = ["person", "organization", "location"]
    batch, official = _official_batch(model, text, entities=labels)
    class_batch, class_official = _official_batch(
        model, text, classes=["news", "biography"]
    )
    relation_batch, _ = _official_batch(
        model,
        text,
        joint_relations={
            "relations": {
                "entities": ["person", "organization"],
                "relations": ["works_at"],
            }
        },
    )
    struct_batch, _ = _official_batch(
        model, text, structures={"profile": ["person", "organization"]}
    )
    def prepared(task_batch):
        task_ids = task_batch["input_ids"]
        task_attn = task_batch["attention_mask"]
        task_words_mask = task_batch["words_mask"]
        embeds = EncoderExport(inner.token_rep_layer)(task_ids, task_attn)
        task_words = _gather_words(embeds, task_words_mask)
        return {
            "ids": task_ids,
            "attn": task_attn,
            "words": task_words,
            "word_mask": torch.ones(task_words.shape[:2], dtype=torch.float32),
            "parent": _gather(embeds, task_ids, task_words_mask, inner.config.parent_token_index)[:, :1].squeeze(1),
            "embeds": embeds,
            "cls": embeds[:, 0],
        }

    with torch.no_grad():
        ner_pack = prepared(batch)
        class_pack = prepared(class_batch)
        relation_pack = prepared(relation_batch)
        struct_pack = prepared(struct_batch)
        ids, attn = ner_pack["ids"], ner_pack["attn"]
        words, word_mask, parent = ner_pack["words"], ner_pack["word_mask"], ner_pack["parent"]
        token_embeds = ner_pack["embeds"]
        entity_children = _gather(token_embeds, ids, batch["words_mask"], inner.config.class_token_index)
        class_children = _gather(
            class_pack["embeds"],
            class_pack["ids"],
            class_batch["words_mask"],
            inner.config.classification_config.cat_token_index,
        )
        relation_children = _gather(
            relation_pack["embeds"],
            relation_pack["ids"],
            relation_batch["words_mask"],
            inner.config.joint_relex_config.rel_token_index,
        )
        ner_ref = NerExport(ner_head, rnn)(
            words,
            word_mask,
            entity_children,
            torch.ones(entity_children.shape[:2]),
            parent,
        )
        cls_ref = ClassificationExport(inner.heads["classification"], rnn)(
            class_pack["words"],
            class_pack["word_mask"],
            class_children,
            torch.ones(class_children.shape[:2]),
            class_pack["parent"],
            class_pack["cls"],
        )

    print("exporting encoder")
    _export(
        EncoderExport(inner.token_rep_layer).eval(),
        (ids, attn),
        ONNX_DIR / "encoder.onnx",
        ["input_ids", "attention_mask"],
        ["token_embeds"],
        {
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_embeds": {0: "batch", 1: "sequence"},
        },
    )
    print("exporting ner")
    _export(
        NerExport(ner_head, rnn).eval(),
        (words, word_mask, entity_children, torch.ones(entity_children.shape[:2]), parent),
        ONNX_DIR / "ner.onnx",
        ["words", "word_mask", "children", "child_mask", "parent"],
        ["ner_logits"],
        {
            "words": {1: "words"},
            "word_mask": {1: "words"},
            "children": {1: "classes"},
            "child_mask": {1: "classes"},
            "ner_logits": {1: "words", 2: "classes"},
        },
    )
    print("exporting classification")
    _export(
        ClassificationExport(inner.heads["classification"], rnn).eval(),
        (
            class_pack["words"],
            class_pack["word_mask"],
            class_children,
            torch.ones(class_children.shape[:2]),
            class_pack["parent"],
            class_pack["cls"],
        ),
        ONNX_DIR / "classification.onnx",
        ["words", "word_mask", "children", "child_mask", "parent", "cls_embed"],
        ["class_logits"],
        {
            "words": {1: "words"},
            "word_mask": {1: "words"},
            "children": {1: "classes"},
            "child_mask": {1: "classes"},
            "class_logits": {1: "classes"},
        },
    )

    span_idx = torch.tensor([[[0, 0], [3, 3]]], dtype=torch.long)
    span_mask = torch.ones(1, 2)
    if relation_children.shape[1] == 0:
        raise SystemExit("relation prompt produced no [RELATION] embeddings")
    relation_spans = span_idx[:, : relation_pack["words"].shape[1]].clamp(max=relation_pack["words"].shape[1] - 1)
    struct_spans = span_idx[:, : struct_pack["words"].shape[1]].clamp(max=struct_pack["words"].shape[1] - 1)
    structuring_head = inner.heads["structuring"]
    export_relations = bool(getattr(structuring_head, "multi_level", False)) and getattr(
        structuring_head, "anchor_relations_rep_layer", None
    ) is not None
    with torch.no_grad():
        rel_ref = RelationsExport(inner.heads["joint_relex"], rnn)(
            relation_pack["words"],
            relation_pack["word_mask"],
            relation_spans,
            span_mask,
            relation_children,
        )
        struct_ref = StructuringExport(structuring_head, rnn, export_relations)(
            struct_pack["words"],
            struct_pack["word_mask"],
            struct_spans,
            span_mask,
            struct_pack["parent"],
        )

    print("exporting relations")
    _export(
        RelationsExport(inner.heads["joint_relex"], rnn).eval(),
        (
            relation_pack["words"],
            relation_pack["word_mask"],
            relation_spans,
            span_mask,
            relation_children,
        ),
        ONNX_DIR / "relations.onnx",
        ["words", "word_mask", "span_idx", "span_mask", "relations"],
        ["relation_logits"],
        {
            "words": {1: "words"},
            "word_mask": {1: "words"},
            "span_idx": {1: "spans"},
            "span_mask": {1: "spans"},
            "relations": {1: "relations"},
            "relation_logits": {1: "spans", 2: "spans", 3: "relations"},
        },
    )
    print("exporting structuring")
    structure_outputs = ["membership", "objectness", "anchor_mask"]
    structure_axes = {
        "words": {1: "words"},
        "word_mask": {1: "words"},
        "span_idx": {1: "spans"},
        "span_mask": {1: "spans"},
        "membership": {2: "spans"},
    }
    if export_relations:
        structure_outputs.append("anchor_relations")
        structure_axes["anchor_relations"] = {1: "anchors", 2: "anchors"}
    _export(
        StructuringExport(structuring_head, rnn, export_relations).eval(),
        (
            struct_pack["words"],
            struct_pack["word_mask"],
            struct_spans,
            span_mask,
            struct_pack["parent"],
        ),
        ONNX_DIR / "structuring.onnx",
        ["words", "word_mask", "span_idx", "span_mask", "parent"],
        structure_outputs,
        structure_axes,
    )

    providers = ["CPUExecutionProvider"]
    encoder = ort.InferenceSession(ONNX_DIR / "encoder.onnx", providers=providers)
    ner = ort.InferenceSession(ONNX_DIR / "ner.onnx", providers=providers)
    classification = ort.InferenceSession(ONNX_DIR / "classification.onnx", providers=providers)
    relations = ort.InferenceSession(ONNX_DIR / "relations.onnx", providers=providers)
    structuring = ort.InferenceSession(ONNX_DIR / "structuring.onnx", providers=providers)

    print("parity")
    embeds = _run(encoder, {"input_ids": ids.numpy(), "attention_mask": attn.numpy()})[0]
    _check("encoder embeddings", embeds, token_embeds.detach().numpy())
    ner_out = _run(
        ner,
        {
            "words": words.detach().numpy(),
            "word_mask": word_mask.numpy(),
            "children": entity_children.detach().numpy(),
            "child_mask": np.ones(entity_children.shape[:2], dtype=np.float32),
            "parent": parent.detach().numpy(),
        },
    )[0]
    _check("ner logits", ner_out, ner_ref.detach().numpy())
    if official.ner_logits is not None:
        _check("ner vs official forward", ner_out, official.ner_logits.detach().numpy())
    cls_out = _run(
        classification,
        {
            "words": class_pack["words"].detach().numpy(),
            "word_mask": class_pack["word_mask"].numpy(),
            "children": class_children.detach().numpy(),
            "child_mask": np.ones(class_children.shape[:2], dtype=np.float32),
            "parent": class_pack["parent"].detach().numpy(),
            "cls_embed": class_pack["cls"].detach().numpy(),
        },
    )[0]
    _check("classification logits", cls_out, cls_ref.detach().numpy())
    if class_official.cat_logits is not None:
        _check("classification vs official forward", cls_out, class_official.cat_logits.detach().numpy())

    rel_feed = {
        "words": relation_pack["words"].detach().numpy(),
        "word_mask": relation_pack["word_mask"].numpy(),
        "span_idx": relation_spans.numpy(),
        "span_mask": span_mask.numpy(),
        "relations": relation_children.detach().numpy(),
    }
    rel_out = _run(relations, rel_feed)
    _check("relation logits", rel_out[0], rel_ref.detach().numpy())

    struct_feed = {
        "words": struct_pack["words"].detach().numpy(),
        "word_mask": struct_pack["word_mask"].numpy(),
        "span_idx": struct_spans.numpy(),
        "span_mask": span_mask.numpy(),
        "parent": struct_pack["parent"].detach().numpy(),
    }
    struct_out = _run(structuring, struct_feed)
    _check("structuring membership", struct_out[0], struct_ref[0].detach().numpy())
    _check("structuring objectness", struct_out[1], struct_ref[1].detach().numpy())
    if export_relations:
        _check("structuring anchor relations", struct_out[3], struct_ref[3].detach().numpy())

    other = "Bob met Carol."
    other_batch, _ = _official_batch(model, other, entities=["person"])
    other_ids = other_batch["input_ids"]
    other_attn = other_batch["attention_mask"]
    with torch.no_grad():
        other_ref = EncoderExport(inner.token_rep_layer)(other_ids, other_attn)
    other_out = _run(encoder, {"input_ids": other_ids.numpy(), "attention_mask": other_attn.numpy()})[0]
    _check("encoder dynamic length", other_out, other_ref.detach().numpy())

    words2 = words[:, :4].contiguous()
    children2 = entity_children[:, :2].contiguous()
    with torch.no_grad():
        ner_dyn_ref = NerExport(ner_head, rnn)(
            words2,
            torch.ones(1, 4),
            children2,
            torch.ones(1, 2),
            parent,
        )
    ner_dyn = _run(
        ner,
        {
            "words": words2.detach().numpy(),
            "word_mask": np.ones((1, 4), dtype=np.float32),
            "children": children2.detach().numpy(),
            "child_mask": np.ones((1, 2), dtype=np.float32),
            "parent": parent.detach().numpy(),
        },
    )[0]
    _check("ner dynamic shapes", ner_dyn, ner_dyn_ref.detach().numpy())
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
