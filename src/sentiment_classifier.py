#!/usr/bin/env python
# coding: utf-8
"""Aspect-based sentiment classifier wrapper.

Provides a BERT/DeBERTa-based aspect-based sentiment analysis (ABSA) classifier
that, given a (sentence, aspect_name) pair, predicts positive | negative | neutral
for *that specific aspect* in the sentence. This is more accurate than the legacy
keyword-counting approach in aspect_inference.py (it handles negation, sarcasm,
and words outside a fixed keyword list).

The default model is aspect-aware:
    yangheng/deberta-v3-base-absa-v1.1
which takes the sentence as text and the aspect term as text_pair, and outputs
one of {Negative, Neutral, Positive}.

A sentence-level fallback model is also supported:
    cardiffnlp/twitter-roberta-base-sentiment-latest

The classifier is lazy-loaded so importing this module is cheap; the heavy
transformers/torch import only happens on first classify() call.
"""

from __future__ import annotations

from typing import Iterable

# Canonical labels used across the project.
POSITIVE = "pos"
NEGATIVE = "neg"
NEUTRAL = "neu"

# Maps raw model labels (varies per checkpoint) to our canonical 3-way labels.
_LABEL_MAP = {
    "positive": POSITIVE,
    "pos": POSITIVE,
    "label_2": POSITIVE,
    "negative": NEGATIVE,
    "neg": NEGATIVE,
    "label_0": NEGATIVE,
    "neutral": NEUTRAL,
    "neu": NEUTRAL,
    "label_1": NEUTRAL,
}

DEFAULT_ABSA_MODEL = "yangheng/deberta-v3-base-absa-v1.1"
DEFAULT_SENTENCE_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def _normalize_label(raw_label: str) -> str:
    key = str(raw_label).strip().lower()
    return _LABEL_MAP.get(key, NEUTRAL)


class SentimentClassifier:
    """Lazy-loaded transformers sentiment classifier.

    Parameters
    ----------
    model_name:
        HuggingFace model id. Defaults to an aspect-based (ABSA) model.
    aspect_aware:
        If True, the aspect name is passed as the text_pair so the model scores
        the sentiment *toward that aspect*. If False, only the sentence is used
        (whole-sentence sentiment).
    device:
        'cpu', 'cuda', or 'auto' (use cuda when available).
    batch_size:
        Max sentences per forward pass in classify_batch.
    """

    def __init__(self,
                 model_name: str = DEFAULT_ABSA_MODEL,
                 aspect_aware: bool = True,
                 device: str = "auto",
                 batch_size: int = 16):
        self.model_name = model_name
        self.aspect_aware = aspect_aware
        self.device_pref = device
        self.batch_size = batch_size
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._device = None
        self._id2label = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import (AutoModelForSequenceClassification,
                                       AutoTokenizer)
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise SystemExit(
                "Missing transformers dependencies for the BERT sentiment "
                "backend. Install them with:\n"
                "  uv pip install -r requirements_abstractive.txt") from exc

        self._torch = torch
        target_device = "cpu"
        if self.device_pref != "cpu" and torch.cuda.is_available():
            target_device = "cuda"
        self._device = target_device

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name)
        model.to(target_device)
        model.eval()
        self._model = model
        self._id2label = model.config.id2label

    def classify(self, sentence: str, aspect_name: str = "") -> tuple[str, float]:
        """Classify one sentence. Returns (canonical_label, confidence)."""
        labels = self.classify_batch([sentence], [aspect_name])
        return labels[0]

    def classify_batch(self,
                       sentences: list[str],
                       aspect_names: Iterable[str] | None = None
                       ) -> list[tuple[str, float]]:
        """Classify a list of sentences.

        aspect_names: parallel list of aspect display names (one per sentence).
        If None or shorter than sentences, missing entries are treated as "".
        Returns a list of (canonical_label, confidence) tuples.
        """
        if not sentences:
            return []
        self._ensure_loaded()
        torch = self._torch

        names = list(aspect_names) if aspect_names is not None else []
        if len(names) < len(sentences):
            names = names + [""] * (len(sentences) - len(names))

        results: list[tuple[str, float]] = []
        for start in range(0, len(sentences), self.batch_size):
            batch_sents = sentences[start:start + self.batch_size]
            batch_aspects = names[start:start + self.batch_size]

            if self.aspect_aware and any(a for a in batch_aspects):
                encoded = self._tokenizer(
                    batch_sents,
                    [a or "" for a in batch_aspects],
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=256,
                )
            else:
                encoded = self._tokenizer(
                    batch_sents,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=256,
                )
            encoded = {k: v.to(self._device) for k, v in encoded.items()}

            with torch.inference_mode():
                logits = self._model(**encoded).logits
                probs = torch.softmax(logits, dim=-1)
                conf, idx = torch.max(probs, dim=-1)

            for row_conf, row_idx in zip(conf.tolist(), idx.tolist()):
                raw_label = self._id2label.get(
                    row_idx, self._id2label.get(str(row_idx), str(row_idx)))
                results.append((_normalize_label(raw_label), float(row_conf)))

        return results


_SHARED_CLASSIFIER: SentimentClassifier | None = None


def get_classifier(model_name: str = DEFAULT_ABSA_MODEL,
                   aspect_aware: bool = True,
                   device: str = "auto",
                   batch_size: int = 16) -> SentimentClassifier:
    """Return a process-wide shared classifier (loaded once)."""
    global _SHARED_CLASSIFIER
    if _SHARED_CLASSIFIER is None:
        _SHARED_CLASSIFIER = SentimentClassifier(
            model_name=model_name,
            aspect_aware=aspect_aware,
            device=device,
            batch_size=batch_size,
        )
    return _SHARED_CLASSIFIER
